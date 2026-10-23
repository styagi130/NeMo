# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming request metadata propagation for vLLM-Omni 0.26.

The upstream resume path does not merge ``additional_information`` into
``model_intermediate_buffer``. Preserve model-generated state while forwarding
per-chunk EasyMagpie text and conditioning metadata.
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
from easymagpie_vllm_omni.batched_feedback import feedback_merge_status, try_merge_feedback
from easymagpie_vllm_omni.feedback_snapshot import is_owned_feedback
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker import gpu_ar_worker, gpu_generation_worker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker

CPU_QUERY_VECTOR_ENABLED = os.environ.get("EASYMAGPIE_CPU_QUERY_VECTOR", "0") == "1"


def merge_streaming_additional_information(
    cached: dict[str, Any],
    incoming: dict[str, Any],
    accumulated_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge one streaming chunk without dropping persistent model state."""
    accumulated_keys = accumulated_keys or set()
    merged = dict(cached)

    for key, value in incoming.items():
        if not isinstance(value, dict):
            merged[key] = value
            continue

        old_value = merged.get(key)
        merged_sub = dict(old_value) if isinstance(old_value, dict) else {}
        for subkey, subvalue in value.items():
            if (key, subkey) in accumulated_keys and isinstance(subvalue, torch.Tensor):
                new_tensor = subvalue.detach().to("cpu").contiguous()
                old_tensor = merged_sub.get(subkey)
                merged_sub[subkey] = new_tensor if old_tensor is None else torch.cat((old_tensor, new_tensor), dim=0)
            else:
                merged_sub[subkey] = subvalue
        merged[key] = merged_sub

    meta = dict(merged.get("meta", {}))
    meta["num_processed_tokens"] = 0
    meta["resumable"] = True
    merged["meta"] = meta
    return merged


class EasyMagpieGPUARModelRunner(GPUARModelRunner):
    """GPU AR runner that restores streaming chunk metadata propagation."""

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._bulk_fresh_prefill_inputs = None
        value = os.environ.get("EASYMAGPIE_BULK_FRESH_PREFILL_INPUTS", "0")
        if value not in ("0", "1"):
            raise ValueError("EASYMAGPIE_BULK_FRESH_PREFILL_INPUTS must be 0 or 1")
        if value == "1":
            from .bulk_fresh_prefill_inputs import BulkFreshInputs
            from .easymagpie import EasyMagpieTTSForConditionalGeneration

            model = self.get_model()
            if type(model) is not EasyMagpieTTSForConditionalGeneration:
                raise TypeError("bulk fresh input preparation requires the EasyMagpie talker")
            self._bulk_fresh_prefill_inputs = BulkFreshInputs(self, model)
        return result

    def _query_starts_cpu(self, count: int, site: str) -> list[int]:
        """Read the existing CPU int32 buffer once; return a fresh Python list."""
        host = self.query_start_loc.cpu
        if CPU_QUERY_VECTOR_ENABLED:
            counters = getattr(self, "_cpu_query_vector_counts", None)
            if counters is None:
                counters = self._cpu_query_vector_counts = Counter()
            # Metadata-only guards preserve the legacy behavior for any
            # unexpected frontend/buffer type, dtype, rank or short capacity.
            if (
                isinstance(host, torch.Tensor)
                and host.device.type == "cpu"
                and host.dtype == torch.int32
                and host.ndim == 1
                and 0 <= count <= host.numel()
            ):
                result = host[:count].tolist()
                counters[site + "_calls"] += 1
                counters[site + "_rows"] += count
                return result
            counters["fallback_calls"] += 1
        return [int(host[index]) for index in range(count)]

    def _maybe_run_batch_preprocess(self, req_ids: list[str], device: torch.device) -> None:
        super()._maybe_run_batch_preprocess(req_ids, device)
        prepare = getattr(self.model, "prepare_decode_batch", None)
        if callable(prepare):
            starts = self._query_starts_cpu(len(req_ids), "batch_preprocess")
            counts = [int(self._omni_num_scheduled_tokens_np[index]) for index in range(len(req_ids))]
            use_device_state = getattr(self.model, "_device_decode_state", None) is not None
            if use_device_state or getattr(self.model, "_prefill_graphs_enabled", False):
                self.model.prepare_query_dispatch(starts, counts, device)
            rows, request_keys = [], []
            for index, (req_id, start, count) in enumerate(zip(req_ids, starts, counts, strict=True)):
                request = self.requests[req_id]
                computed = int(self.input_batch.num_computed_tokens_cpu[index])
                if count == 1 and computed >= len(request.prompt_token_ids or []):
                    rows.append((start, self.model_intermediate_buffer.get(req_id, {})))
                    if use_device_state:
                        request_keys.append((req_id, id(self.requests[req_id])))
            if use_device_state:
                if getattr(self.model, "_retain_live_decode_leases", False):
                    # Keep the preceding snapshot alive until after pruning: a
                    # completed object cannot have its Python id recycled for
                    # a same-ID admission while its old lease still exists.
                    live_requests = dict(self.requests)
                    live_keys = [(key, id(request)) for key, request in live_requests.items()]
                    prepare(rows, device, request_keys=request_keys, live_request_keys=live_keys)
                    self._dds_live_request_objects = live_requests
                else:
                    prepare(rows, device, request_keys=request_keys)
            else:
                prepare(rows, device)

    def _try_complete_decode_preprocess(self, input_ids, inputs_embeds, counts) -> bool:
        if self._try_complete_q1_preprocess(input_ids, inputs_embeds, counts):
            return True
        fresh = getattr(self, "_bulk_fresh_prefill_inputs", None)
        return fresh is not None and fresh.try_prepare(input_ids, inputs_embeds, counts)

    def _try_complete_q1_preprocess(self, input_ids, inputs_embeds, counts) -> bool:
        """Consume an eligible DDS batch without per-row tensor copy launches.

        Called by the isolated pinned-runtime hook after the base runner has
        prepared connector metadata, positions and graph-compatible buffers.
        Every guard runs before any update is consumed. The old loop handles
        the entire batch if one row cannot use this path.
        """
        from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

        model = self.get_model()
        if type(model) is not EasyMagpieTTSForConditionalGeneration or not getattr(
            model, "_batched_decode_inputs_enabled", False
        ):
            return False
        self._batched_decode_input_attempts = getattr(self, "_batched_decode_input_attempts", 0) + 1
        if (
            getattr(model, "_device_decode_state", None) is None
            or getattr(self, "has_talker_mtp", False)
            or getattr(self, "enable_prompt_embeds", False)
            or getattr(self, "speculative_config", None) is not None
            or getattr(self.model_config, "is_encoder_decoder", False)
            or getattr(self.parallel_config, "pipeline_parallel_size", 1) != 1
            or inputs_embeds is None
            or input_ids is None
            or input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.storage_offset() != 0
        ):
            return False
        req_ids = self.input_batch.req_ids
        if not req_ids or len(counts) != len(req_ids) or any(int(count) != 1 for count in counts):
            return False
        starts = self._query_starts_cpu(len(req_ids), "decode_fastpath")
        if starts != list(range(len(req_ids))) or inputs_embeds.shape[0] < len(req_ids):
            return False
        prepared = model._prepared_decode_updates
        if set(prepared) != set(starts):
            return False
        metadata = []
        for index, (req_id, start) in enumerate(zip(req_ids, starts, strict=True)):
            request = self.requests.get(req_id)
            prompt = getattr(request, "prompt_token_ids", None)
            if request is None or prompt is None:
                return False
            computed = int(self.input_batch.num_computed_tokens_cpu[index])
            if computed < len(prompt):
                return False
            update = prepared[start]
            if set(update) - {"decode_offset", "text_tokens"} or any(
                isinstance(value, torch.Tensor) for value in update.values()
            ):
                return False
            metadata.append((req_id, start, computed, len(prompt)))

        # The normal EasyMagpie single-token hook returns input_ids unchanged
        # and one zero embedding. Preserve padded rows exactly as the old loop.
        inputs_embeds[: len(req_ids)].zero_()
        for req_id, start, computed, prompt_len in metadata:
            info = self.model_intermediate_buffer.get(req_id, {})
            info.update(
                request_id=req_id,
                duplex_token_offset=computed,
                duplex_prompt_len=prompt_len,
                _omni_prompt_len=prompt_len,
                _omni_num_computed_tokens=computed,
                _omni_is_prefill=False,
            )
            self._update_intermediate_buffer(req_id, prepared.pop(start))
        self._batched_decode_input_hits = getattr(self, "_batched_decode_input_hits", 0) + 1
        self._batched_decode_input_requests = getattr(self, "_batched_decode_input_requests", 0) + len(req_ids)
        return True

    def get_prepost_status(self) -> dict[str, Any]:
        """Read-only on-demand counters; no device reads or synchronization."""
        model = self.get_model()
        return {
            "cpu_query_vector": {
                "enabled": CPU_QUERY_VECTOR_ENABLED,
                **dict(getattr(self, "_cpu_query_vector_counts", {})),
            },
            "device_decode_state": getattr(model, "_device_decode_state", None) is not None,
            "retain_live_decode_leases": getattr(model, "_retain_live_decode_leases", False),
            "decode_lease_status": (
                model._device_decode_state.lease_status()
                if getattr(model, "_device_decode_state", None) is not None
                else None
            ),
            "feedback_merge": feedback_merge_status(self),
            "batched_owned_feedback": getattr(model, "_batched_owned_feedback_enabled", False),
            "batched_decode_inputs": getattr(model, "_batched_decode_inputs_enabled", False),
            "batch_input_attempts": getattr(self, "_batched_decode_input_attempts", 0),
            "batch_input_hits": getattr(self, "_batched_decode_input_hits", 0),
            "batch_input_requests": getattr(self, "_batched_decode_input_requests", 0),
            "owned_feedback_view_stores": getattr(self, "_owned_feedback_view_stores", 0),
        }

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        try:
            return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
        finally:
            clear = getattr(self.model, "clear_decode_batch", None)
            if callable(clear):
                clear()

    def _process_additional_information_updates(
        self,
        hidden_states,
        multimodal_outputs,
        num_scheduled_tokens_np,
        scheduler_output,
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        req_ids_filter=None,
        req_ids=None,
        query_start_loc_cpu=None,
    ):
        if try_merge_feedback(
            self,
            hidden_states,
            multimodal_outputs,
            num_scheduled_tokens_np,
            combined_hidden_states=combined_hidden_states,
            combined_multimodal_outputs=combined_multimodal_outputs,
            req_ids_filter=req_ids_filter,
            req_ids=req_ids,
            query_start_loc_cpu=query_start_loc_cpu,
        ):
            return None
        return super()._process_additional_information_updates(
            hidden_states,
            multimodal_outputs,
            num_scheduled_tokens_np,
            scheduler_output,
            combined_hidden_states,
            combined_multimodal_outputs,
            req_ids_filter=req_ids_filter,
            req_ids=req_ids,
            query_start_loc_cpu=query_start_loc_cpu,
        )

    def _build_omni_async_snapshot_payload(
        self,
        *,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
    ) -> dict[str, Any]:
        payload = super()._build_omni_async_snapshot_payload(
            hidden_states=hidden_states,
            staged_hidden_states_cpu=staged_hidden_states_cpu,
            multimodal_outputs=multimodal_outputs,
        )
        # Carry the padded request axis without copying hidden values. Upstream
        # needs this length to slice per-request codes; an explicit count would
        # avoid using a temporary (padded_batch, 0) tensor as metadata.
        payload.setdefault("hidden_states", hidden_states[:, :0])
        return payload

    def _store_value(self, dest: dict, key: str, value: Any, gpu_keys: set) -> None:
        # Only postprocess views certified against immutable output snapshots
        # bypass cloning. Ordinary tensors keep upstream ownership semantics.
        if key in gpu_keys and is_owned_feedback(value, key):
            self._owned_feedback_view_stores = getattr(self, "_owned_feedback_view_stores", 0) + 1
            dest[key] = value.detach()
            return
        return super()._store_value(dest, key, value, gpu_keys)

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not isinstance(upd, dict) or not upd:
            return
        request = self.requests.get(req_id)
        if request is None:
            return

        gpu_keys = getattr(getattr(self, "model", None), "gpu_resident_buffer_keys", set())
        top_level_gpu_keys = {key for key in gpu_keys if isinstance(key, str)}
        nested_gpu_keys = {key for key in gpu_keys if isinstance(key, tuple) and len(key) == 2}
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for key, value in upd.items():
            if isinstance(value, dict):
                existing_sub = existing.setdefault(key, {})
                resident_qualifiers = {qualifier for type_key, qualifier in nested_gpu_keys if type_key == key}
                for qualifier, subvalue in value.items():
                    self._store_value(existing_sub, qualifier, subvalue, resident_qualifiers)
            else:
                self._store_value(existing, key, value, top_level_gpu_keys)
        request.additional_information_cpu = existing

    def _update_streaming_request(self, req_id, new_req_data):
        payload = getattr(new_req_data, "additional_information", None)
        incoming = deserialize_additional_information(payload)
        if isinstance(incoming, dict) and incoming:
            model = getattr(self, "model", None)
            accumulated_keys = getattr(model, "streaming_accumulated_keys", set())
            cached = self.model_intermediate_buffer.get(req_id, {})
            merged = merge_streaming_additional_information(cached, incoming, accumulated_keys)
            self.model_intermediate_buffer[req_id] = merged
            setattr(self.requests[req_id], "additional_information_cpu", merged)

        return super()._update_streaming_request(req_id, new_req_data)


class EasyMagpieGPUARWorker(GPUARWorker):
    """GPU AR worker that constructs :class:`EasyMagpieGPUARModelRunner`."""

    def compile_or_warm_up_model(self):
        # Capture only after upstream kernel warmup, graph capture and cache
        # binding have completed, before the engine admits its first request.
        from .mamba_prefill_islands import prewarm_model

        result = super().compile_or_warm_up_model()
        prewarm_model(self.model_runner.model, self.model_runner)
        self._mamba_prefill_startup_complete = True
        return result

    def get_mamba_prefill_status(self):
        """Read-only worker RPC; graph preparation never happens on this path."""
        from .mamba_prefill_islands import graph_status

        status = graph_status(self.model_runner.model)
        completed = getattr(self, "_mamba_prefill_startup_complete", False)
        return {
            **status,
            "worker_startup_complete": completed,
            "ready": completed and status["enabled"] and status["frozen"] and status["entries"] == 112,
            "startup_hook": "EasyMagpieGPUARWorker.after_ordinary_warmup",
        }

    def init_device(self):
        # GPUARWorker hardcodes its module-level GPUARModelRunner symbol rather
        # than exposing a runner-class hook. Swap it only while the base method
        # constructs this worker's runner; each worker lives in its own process.
        original_runner_cls = gpu_ar_worker.GPUARModelRunner
        gpu_ar_worker.GPUARModelRunner = EasyMagpieGPUARModelRunner
        try:
            return super().init_device()
        finally:
            gpu_ar_worker.GPUARModelRunner = original_runner_cls


def batch_waveforms_to_cpu(outputs: Any) -> Any:
    """Copy a per-request waveform list to CPU in one transfer."""
    if not isinstance(outputs, list) or len(outputs) < 2 or not all(isinstance(x, torch.Tensor) for x in outputs):
        return outputs
    if len({(x.device, x.dtype) for x in outputs}) != 1:
        return outputs

    sizes = [x.numel() for x in outputs]
    shapes = [x.shape for x in outputs]
    packed = torch.cat([x.detach().reshape(-1) for x in outputs]).to("cpu").contiguous()
    return [part.view(shape) for part, shape in zip(packed.split(sizes), shapes, strict=True)]


class EasyMagpieCodecGPUGenerationModelRunner(GPUGenerationModelRunner):
    """Batch Stage-1 waveform D2H before upstream output handling."""

    def codec_async_status(self):
        from .codec_async_output import codec_async_output_enabled

        config = self.vllm_config.scheduler_config
        cls = config.scheduler_cls
        return {
            "enabled": codec_async_output_enabled(),
            "async_scheduling": self.use_async_scheduling,
            "scheduler_cls": (
                cls if isinstance(cls, str) else f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}"
            ),
            "counters": dict(getattr(self, "_codec_async_stats", {})),
            "copy_stream_created": getattr(self, "_codec_pcm_copy_stream", None) is not None,
        }

    def sample_tokens(self, grammar_output=None):
        state = self.execute_model_state
        from .codec_async_output import (
            build_codec_async_output,
            codec_async_fallback_reason,
            codec_async_output_enabled,
        )

        if codec_async_output_enabled():
            if not hasattr(self, "_codec_async_stats"):
                self._codec_async_stats = Counter()
                self._codec_async_stats_lock = threading.Lock()
            reason = codec_async_fallback_reason(self, state)
            if grammar_output is not None:
                reason = "grammar"
            if reason is None:
                result = build_codec_async_output(self, state)
                self._codec_async_stats["deferred_outputs"] += 1
                return result
            self._codec_async_stats["fallback:" + reason] += 1
        if state is not None and isinstance(state.multimodal_outputs, Mapping):
            multimodal_outputs = dict(state.multimodal_outputs)
            outputs = multimodal_outputs.get("model_outputs")
            batched = batch_waveforms_to_cpu(outputs)
            if batched is not outputs:
                multimodal_outputs["model_outputs"] = batched
                self.execute_model_state = state._replace(multimodal_outputs=multimodal_outputs)
        return super().sample_tokens(grammar_output)


class EasyMagpieCodecGPUGenerationWorker(GPUGenerationWorker):
    """Construct the codec runner through the upstream worker."""

    def init_device(self):
        original_runner_cls = gpu_generation_worker.GPUGenerationModelRunner
        gpu_generation_worker.GPUGenerationModelRunner = EasyMagpieCodecGPUGenerationModelRunner
        try:
            return super().init_device()
        finally:
            gpu_generation_worker.GPUGenerationModelRunner = original_runner_cls
