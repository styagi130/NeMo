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
"""Batch fresh 71-token input construction without reusing learned execution.

The native Q1 preparation runs first. Unsupported, mixed, partial and cold-cache
prefills use the complete original preprocessing loop before any state changes.
"""
import math
import os
import threading
import weakref
from collections import Counter

import numpy as np
import torch

MAX_BATCH = 115
COUNTERS = (
    "bulk_batches",
    "bulk_rows",
    "attempted_prefill_batches",
    "candidate_fresh_rows",
    "native_fallback_batches",
    "native_fallback_rows",
    "mixed_batches",
    "mixed_fresh_rows",
    "partial_batches",
    "partial_rows",
    "singleton_batches",
    "padded_batches",
    "padded_tail_rows_untouched",
    "errors",
)


def _function(value):
    return getattr(value, "__func__", value)


def _pure_embedding(value):
    return (
        type(value) is torch.nn.Embedding
        and value.max_norm is None
        and _function(value.forward) is torch.nn.Embedding.forward
        and not value._forward_hooks
        and not value._forward_pre_hooks
        and not value._backward_hooks
        and getattr(value, "_compiled_call_impl", None) is None
    )


class BulkFreshInputs:
    def __init__(self, runner, model):
        from .easymagpie import EasyMagpieTTSForConditionalGeneration
        from .runner import EasyMagpieGPUARModelRunner

        if type(model) is not EasyMagpieTTSForConditionalGeneration:
            raise TypeError("bulk fresh inputs require the EasyMagpie talker")
        if runner.requests or runner.input_batch.num_reqs or runner.input_batch.req_ids:
            raise RuntimeError("bulk fresh inputs must initialize before traffic")
        methods = {}
        for name in (
            "preprocess",
            "_preprocess_prefill",
            "_build_prefill_embeds",
            "_build_text_prefill_embeds",
            "_embed_phoneme",
            "_maybe_set_lt_sampling_params",
        ):
            method = _function(getattr(model, name))
            if method is not getattr(EasyMagpieTTSForConditionalGeneration, name):
                raise RuntimeError("bulk fresh inputs require native model preprocessing")
            methods[name] = method
        if _function(runner._update_intermediate_buffer) is not EasyMagpieGPUARModelRunner._update_intermediate_buffer:
            raise RuntimeError("bulk fresh inputs require the native metadata update")
        self.runner, self.model = weakref.ref(runner), weakref.ref(model)
        self.pid = os.getpid()
        self.actor_tid = None
        self.closed = False
        self.close_calls = 0
        self.counts = Counter(dict.fromkeys(COUNTERS, 0))
        self.fallback_reasons = Counter()
        self.methods = methods
        self.update = _function(runner._update_intermediate_buffer)

    def binding(self):
        runner, model = self.runner(), self.model()
        return (
            runner is not None
            and model is not None
            and os.getpid() == self.pid
            and runner.get_model() is model
            and getattr(runner, "_bulk_fresh_prefill_inputs", None) is self
        )

    def _fallback(self, reason, rows):
        self.fallback_reasons[reason] += 1
        self.counts["native_fallback_batches"] += 1
        self.counts["native_fallback_rows"] += rows
        return False

    def _guard(self, runner, model, input_ids, inputs_embeds, counts, metadata):
        b = len(metadata)
        if not 2 <= b <= MAX_BATCH:
            return "batch_width"
        if (
            not self.binding()
            or model.training
            or torch.is_grad_enabled()
            or self.actor_tid is not None
            and self.actor_tid != threading.get_ident()
            or getattr(runner, "has_talker_mtp", False)
            or getattr(runner, "enable_prompt_embeds", False)
            or getattr(runner, "speculative_config", None) is not None
            or getattr(runner.model_config, "is_encoder_decoder", False)
            or getattr(runner.parallel_config, "pipeline_parallel_size", 1) != 1
        ):
            return "runner_or_execution_context"
        if (
            any(_function(getattr(model, name)) is not method for name, method in self.methods.items())
            or _function(runner._update_intermediate_buffer) is not self.update
        ):
            return "native_method_changed"
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype != torch.int32
            or input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.storage_offset() != 0
            or not isinstance(inputs_embeds, torch.Tensor)
            or inputs_embeds.ndim != 2
            or not inputs_embeds.is_contiguous()
            or inputs_embeds.storage_offset() != 0
            or inputs_embeds.shape != (input_ids.numel(), model.embedding_dim)
            or input_ids.device != inputs_embeds.device
            or input_ids.numel() < b * 71
            or inputs_embeds.dtype not in (torch.float16, torch.float32)
            or inputs_embeds.dtype != model._combined_embeddings.dtype
            or inputs_embeds.device != model._combined_embeddings.device
            or input_ids.data_ptr() != runner.input_ids.gpu.data_ptr()
            or inputs_embeds.data_ptr() != runner.inputs_embeds.gpu.data_ptr()
        ):
            return "owned_input_buffers"
        if (
            model.arch.text_prefill_num != 4
            or not _pure_embedding(model.text_embedding)
            or tuple(model.text_embedding.weight.shape) != (model.text_embedding.num_embeddings, model.embedding_dim)
            or model.text_embedding.weight.device != inputs_embeds.device
            or model.text_embedding.weight.dtype not in (torch.float16, torch.float32)
        ):
            return "embedding_configuration"
        hooks = torch.nn.modules.module
        if any(
            getattr(hooks, name, None)
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        ):
            return "embedding_hooks"
        if model.has_phoneme and (
            not 0 <= model.phonemes_delay < 4
            or model.arch.phoneme_stacking_factor != len(model.phoneme_embeddings)
            or not model.phoneme_embeddings
            or any(
                not _pure_embedding(value)
                or not 0 <= model.phoneme_bos_id < value.num_embeddings
                or tuple(value.weight.shape) != (value.num_embeddings, model.embedding_dim)
                or value.weight.device != inputs_embeds.device
                or value.weight.dtype not in (torch.float16, torch.float32)
                for value in model.phoneme_embeddings
            )
        ):
            return "phoneme_configuration"
        conditioning = None
        for index, (req_id, request, info, span, start, computed, prompt_len) in enumerate(metadata):
            if (
                span != 71
                or start != index * 71
                or computed != 0
                or prompt_len != 71
                or type(info) is not dict
                or getattr(request, "mm_features", None)
                or "additional_information" in info
                or type(info.get("text_tokens")) is not list
                or not info["text_tokens"]
                or type(info.get("prefill_offset", 0)) not in (int, type(None))
                or info.get("prefill_offset") not in (None, 0)
                or type(info.get("text_prefill_num")) is not int
                or info["text_prefill_num"] != 4
            ):
                return "fresh_metadata"
            prefix = info.get("prefill_text_tokens")
            if (
                type(prefix) not in (list, tuple)
                or len(prefix) > 4
                or any(
                    type(value) is not int or not 0 <= value < model.text_embedding.num_embeddings for value in prefix
                )
            ):
                return "prefix_metadata"
            temperature, top = info.get("temperature"), info.get("top_k", info.get("topk"))
            if (
                temperature is not None
                and (type(temperature) not in (int, float) or not math.isfinite(temperature))
                or top is not None
                and type(top) is not int
            ):
                return "sampling_metadata"
            speaker, context, task = info.get("speaker_id"), info.get("context_text"), info.get("task_mode_id", 0)
            if (
                type(speaker) is not str
                or not speaker
                or type(context) not in (str, type(None))
                or info.get("speaker_embedding") is not None
                or type(task) not in (int, type(None))
                or task not in (None, 0)
                or info.get("text") is not None
                or info.get("text_token")
                or info.get("speaker_reference_prefilled")
                or info.get("user_audio_prefilled")
            ):
                return "conditioning_metadata"
            context = context or "[EN]"
            task = 0
            key = (task, speaker, context, str(input_ids.device))
            if conditioning is None:
                conditioning = key
            elif key != conditioning:
                return "mixed_conditioning"
        if type(model._prefill_cache) is not dict:
            return "native_context_cache"
        base = model._prefill_cache.get(conditioning)
        if (
            type(base) is not torch.Tensor
            or tuple(base.shape) != (67, model.embedding_dim)
            or base.device != inputs_embeds.device
            or base.dtype != inputs_embeds.dtype
        ):
            return "context_not_warm"
        return base

    def try_prepare(self, input_ids, inputs_embeds, counts):
        runner, model = self.runner(), self.model()
        if self.closed or runner is None or model is None:
            return False
        req_ids = runner.input_batch.req_ids
        host_starts = getattr(getattr(runner, "query_start_loc", None), "cpu", None)
        computed_host = getattr(runner.input_batch, "num_computed_tokens_cpu", None)
        if (
            type(req_ids) is not list
            or not req_ids
            or any(type(key) is not str for key in req_ids)
            or len(set(req_ids)) != len(req_ids)
            or runner.input_batch.num_reqs != len(req_ids)
            or type(counts) is not np.ndarray
            or counts.dtype != np.int32
            or counts.ndim != 1
            or len(counts) != len(req_ids)
            or type(computed_host) is not np.ndarray
            or computed_host.dtype != np.int32
            or computed_host.ndim != 1
            or len(computed_host) < len(req_ids)
            or not isinstance(host_starts, torch.Tensor)
            or host_starts.device.type != "cpu"
            or host_starts.dtype != torch.int32
            or host_starts.ndim != 1
            or host_starts.numel() < len(req_ids) + 1
        ):
            return False
        metadata = []
        for index, req_id in enumerate(req_ids):
            request = runner.requests.get(req_id)
            if request is None:
                return False
            prompt = getattr(request, "prompt_token_ids", None)
            if type(prompt) is not list:
                return False
            metadata.append(
                (
                    req_id,
                    request,
                    runner.model_intermediate_buffer.get(req_id, {}),
                    int(counts[index]),
                    int(host_starts[index]),
                    int(computed_host[index]),
                    len(prompt),
                )
            )
        if len({id(row[1]) for row in metadata}) != len(metadata) or len({id(row[2]) for row in metadata}) != len(
            metadata
        ):
            return self._fallback("aliased_request_or_info", len(metadata))
        prefills = [row for row in metadata if row[5] < row[6]]
        if not prefills:
            return False
        fresh = sum(row[3] == 71 and row[5] == 0 and row[6] == 71 for row in prefills)
        self.counts["attempted_prefill_batches"] += 1
        self.counts["candidate_fresh_rows"] += fresh
        if len(prefills) != len(metadata):
            self.counts["mixed_batches"] += 1
            self.counts["mixed_fresh_rows"] += fresh
        partial = len(prefills) - fresh
        if partial:
            self.counts["partial_batches"] += 1
            self.counts["partial_rows"] += partial
        if len(metadata) == 1:
            self.counts["singleton_batches"] += 1
        try:
            base = self._guard(runner, model, input_ids, inputs_embeds, counts, metadata)
            if not isinstance(base, str) and int(host_starts[len(req_ids)]) != len(req_ids) * 71:
                base = "query_span_end"
            if (
                not isinstance(base, str)
                and input_ids.device.type == "cuda"
                and torch.cuda.is_current_stream_capturing()
            ):
                base = "capture"
        except Exception:
            base = "unsupported_metadata"
        if isinstance(base, str):
            return self._fallback(base, len(metadata))
        self.actor_tid = threading.get_ident()
        try:
            # Metadata validation above precedes ALL original request/sampler
            # mutation. Preserve the native per-owner order for valid inputs.
            for req_id, request, info, span, start, computed, prompt_len in metadata:
                info.update(
                    request_id=req_id,
                    duplex_token_offset=computed,
                    duplex_prompt_len=prompt_len,
                    _omni_prompt_len=prompt_len,
                    _omni_num_computed_tokens=computed,
                    _omni_is_prefill=True,
                )
                model._maybe_set_lt_sampling_params(info)
            prefixes = [list(row[2]["prefill_text_tokens"]) for row in metadata]
            ids = torch.tensor(
                [prefix + [0] * (4 - len(prefix)) for prefix in prefixes], dtype=torch.long, device=input_ids.device
            )
            target = model.text_embedding(ids).to(inputs_embeds.dtype)
            if any(len(prefix) != 4 for prefix in prefixes):
                mask = torch.tensor(
                    [[index < len(prefix) for index in range(4)] for prefix in prefixes],
                    dtype=torch.bool,
                    device=input_ids.device,
                )
                target = torch.where(mask.unsqueeze(-1), target, 0.0)
            if model.has_phoneme:
                bos = torch.full(
                    (1, model.arch.phoneme_stacking_factor),
                    model.phoneme_bos_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                # Native sequential stack additions/division, then same cast
                # and one per-element addition at the native BOS position.
                target[:, model.phonemes_delay] += model._embed_phoneme(bos)[0].to(inputs_embeds.dtype)
            b = len(metadata)
            destination = inputs_embeds[: b * 71].view(b, 71, model.embedding_dim)
            destination[:, :67].copy_(base.unsqueeze(0).expand(b, -1, -1))
            destination[:, 67:].copy_(target)
            input_ids[: b * 71].fill_(0)
            for req_id, request, info, span, start, computed, prompt_len in metadata:
                runner._update_intermediate_buffer(req_id, {"decode_offset": 4})
            self.counts["bulk_batches"] += 1
            self.counts["bulk_rows"] += b
            if input_ids.numel() > b * 71:
                self.counts["padded_batches"] += 1
                self.counts["padded_tail_rows_untouched"] += input_ids.numel() - b * 71
            return True
        except BaseException:
            self.counts["errors"] += 1
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.close_calls += 1
        return self.status()

    def status(self):
        return dict(
            enabled=True,
            closed=self.closed,
            close_calls=self.close_calls,
            binding_matched=self.binding(),
            original_Q1_first=True,
            counts={name: self.counts[name] for name in COUNTERS},
            fallback_reasons=dict(self.fallback_reasons),
            retained_requests=0,
            retained_tensors=0,
            batch_range=[2, MAX_BATCH],
            fresh_query_tokens=71,
        )
