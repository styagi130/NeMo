# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""vLLM-Omni 0.24 streaming-input compatibility classes.

vLLM-Omni 0.24 no longer merges a resumed request's
``additional_information`` into ``model_intermediate_buffer``. Consequently,
per-chunk EasyMagpie ``text_token`` payloads reach the scheduler but not the
model runner. The custom runner restores the merge performed by 0.21 while
preserving model-generated state such as ``decode_offset`` and ``text_tokens``.
"""
from __future__ import annotations

import inspect
import os
import textwrap
from typing import Any

import torch
from vllm.logger import init_logger
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker import gpu_ar_worker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker

logger = init_logger(__name__)


class EasyMagpieCodecGPUWorker(GPUGenerationWorker):
    """Apply an optional MPS SM cap before Stage 1 creates its CUDA context."""

    def init_device(self):
        percentage = os.environ.get("EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE")
        if percentage:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = percentage
            logger.info("EasyMagpie Stage-1 MPS active-thread cap: %s%%.", percentage)
        return super().init_device()


def _install_multimodal_only_pooler_patch() -> bool:
    """Avoid the unused Stage-0 hidden-state D2H pooler payload.

    EasyMagpie's native codec consumes only ``codes.audio``.  The generic
    asynchronous runner otherwise snapshots hidden states on the CPU for every
    acoustic frame before invoking the transfer callback.
    """
    from vllm_omni.worker import gpu_ar_model_runner

    runner_cls = gpu_ar_model_runner.GPUARModelRunner
    if getattr(runner_cls, "_easymagpie_multimodal_only_pooler_patched", False):
        return True
    # vLLM-Omni 0.24 exposes the model-level selector used directly by
    # ``_build_omni_model_runner_output_from_snapshot``.  The EasyMagpie model
    # sets it to False, so no source rewrite is required on this API layout.
    if hasattr(runner_cls, "_model_omni_pooler_payload_include_hidden"):
        runner_cls._easymagpie_multimodal_only_pooler_patched = True
        logger.info("EasyMagpie disabled the Stage-0 hidden-state pooler payload through vLLM-Omni's public model hook.")
        return True
    original = runner_cls.sample_tokens
    try:
        source = textwrap.dedent(inspect.getsource(original))
    except (OSError, TypeError):
        logger.warning("EasyMagpie could not inspect GPUARModelRunner.sample_tokens; retaining hidden pooler output.")
        return False

    setup = "    hidden_states_cpu = None\n    req_hidden_states_cpu: dict[str, torch.Tensor] | None = None\n"
    replacement = (
        "    include_pooler_hidden = bool(getattr(self.model, \"omni_pooler_payload_include_hidden\", True))\n"
        "    hidden_states_cpu = None\n    req_hidden_states_cpu: dict[str, torch.Tensor] | None = None\n"
    )
    if source.count(setup) != 1 or source.count("    if needs_pooler_payload:\n") < 1:
        logger.warning("EasyMagpie vLLM pooler patch did not match this vLLM-Omni version; retaining hidden pooler output.")
        return False
    source = source.replace(setup, replacement, 1)
    source = source.replace("    if needs_pooler_payload:\n", "    if needs_pooler_payload and include_pooler_hidden:\n", 1)
    payload_start = "            if req_hidden_states_cpu is not None and combined_hidden_states is None:\n"
    payload_end = "            payload: dict[str, object] = {\"hidden\": req_hidden_states}\n"
    start = source.find(payload_start)
    end = source.find(payload_end, start)
    if start < 0 or end < 0:
        logger.warning("EasyMagpie vLLM pooler payload block did not match; retaining hidden pooler output.")
        return False
    end += len(payload_end)
    payload_block = source[start:end]
    source = source[:start] + (
        "            if not include_pooler_hidden:\n"
        "                payload: dict[str, object] = {}\n"
        "            else:\n" + textwrap.indent(payload_block, "    ")
    ) + source[end:]

    namespace: dict[str, Any] = {}
    try:
        filename = inspect.getsourcefile(original) or "<easymagpie_pooler_patch>"
        exec(compile(source, filename, "exec"), gpu_ar_model_runner.__dict__, namespace)
        patched = namespace.get("sample_tokens")
        if not callable(patched):
            raise RuntimeError("generated sample_tokens patch is not callable")
        runner_cls.sample_tokens = patched
    except Exception:
        logger.warning("EasyMagpie could not install the multimodal-only pooler patch.", exc_info=True)
        return False
    runner_cls._easymagpie_multimodal_only_pooler_patched = True
    logger.info("EasyMagpie installed a multimodal-only Stage-0 pooler patch; hidden-state D2H is disabled.")
    return True


def _build_mm_cpu_preserving_stage0_codes(multimodal_outputs: dict[str, Any]) -> dict[str, object]:
    """Retain the Stage-0 ``codes.audio`` leaf on CUDA for a local connector."""
    from vllm_omni.utils.mm_outputs import _to_cpu

    if not isinstance(multimodal_outputs, dict):
        return {}
    prepared: dict[str, object] = {}
    for key, value in multimodal_outputs.items():
        # vLLM-Omni 0.24 marks two-stage audio-sparse outputs with this flat
        # alias. Preserve it on CUDA as well as the structured leaf below.
        if key == "audio_codes" and isinstance(value, torch.Tensor) and value.is_cuda:
            prepared[key] = value.detach()
            continue
        if key == "codes.audio" and isinstance(value, torch.Tensor) and value.is_cuda:
            prepared[key] = value.detach()
            continue
        if key == "codes" and isinstance(value, dict):
            audio = value.get("audio")
            if isinstance(audio, torch.Tensor) and audio.is_cuda:
                codes = {subkey: _to_cpu(subvalue) for subkey, subvalue in value.items() if subkey != "audio"}
                codes["audio"] = audio.detach()
                prepared[key] = codes
                continue
        cpu_value = _to_cpu(value)
        if cpu_value is not None:
            prepared[key] = cpu_value
    return prepared


def install_stage0_cuda_payload_patch() -> bool:
    """Install the Stage-0-only CUDA payload hook, failing closed on API drift."""
    from vllm_omni.worker import gpu_ar_model_runner

    already_patched = getattr(gpu_ar_model_runner, "_easymagpie_cuda_payload_patched", False)
    _install_multimodal_only_pooler_patch()
    gpu_ar_model_runner.build_mm_cpu = _build_mm_cpu_preserving_stage0_codes
    if not already_patched:
        original_prefix_merge = gpu_ar_model_runner.GPUARModelRunner._maybe_get_combined_prefix_cache_tensors

        def _maybe_get_combined_prefix_cache_tensors(
            self,
            hidden_states,
            hidden_states_cpu,
            multimodal_outputs,
            num_scheduled_tokens,
        ):
            if getattr(self.model, "bypass_omni_prefix_cache_for_pooler_payload", False):
                from vllm_omni.data_entry_keys import flatten_payload

                flat = flatten_payload(multimodal_outputs) if multimodal_outputs else {}
                codes = flat.get("codes.audio")
                if not isinstance(codes, torch.Tensor):
                    return None, None
                starts = self.query_start_loc.cpu()
                per_request: dict[str, torch.Tensor] = {}
                for index, req_id in enumerate(self.input_batch.req_ids):
                    start = int(starts[index])
                    end = start + int(num_scheduled_tokens.get(req_id, 0))
                    if end > start:
                        per_request[req_id] = codes[start:end]
                return None, {"codes.audio": per_request}
            return original_prefix_merge(
                self,
                hidden_states,
                hidden_states_cpu,
                multimodal_outputs,
                num_scheduled_tokens,
            )

        gpu_ar_model_runner.GPUARModelRunner._maybe_get_combined_prefix_cache_tensors = (
            _maybe_get_combined_prefix_cache_tensors
        )
    gpu_ar_model_runner._easymagpie_cuda_payload_patched = True
    if not already_patched:
        logger.info("EasyMagpie installed the CUDA-preserving Stage-0 pooler-payload hook.")
    return True


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

    def init_device(self):
        # GPUARWorker hardcodes its module-level GPUARModelRunner symbol rather
        # than exposing a runner-class hook. Swap it only while the base method
        # constructs this worker's runner; each worker lives in its own process.
        if os.environ.get("EASYMAGPIE_STAGE0_CUDA_PAYLOAD", "0") == "1":
            install_stage0_cuda_payload_patch()
        original_runner_cls = gpu_ar_worker.GPUARModelRunner
        gpu_ar_worker.GPUARModelRunner = EasyMagpieGPUARModelRunner
        try:
            return super().init_device()
        finally:
            gpu_ar_worker.GPUARModelRunner = original_runner_cls
