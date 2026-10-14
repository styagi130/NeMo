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
"""Opt-in deferred PCM readback for the native codec's final audio stage.

The output wrapper is resolved by vLLM's executor before IPC serialization. Its
host tensors must never be handed to a consumer until the copy event completes.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm_omni.outputs import OmniModelRunnerOutput


def codec_async_output_enabled():
    return os.environ.get("EASYMAGPIE_CODEC_ASYNC_D2H", "0").lower() in ("1", "true", "yes", "on")


class DeferredCodecOutput(AsyncModelRunnerOutput):
    """Own both ends of one D2H until the executor requests completed output."""

    def __init__(self, output, ready, source, on_complete=None):
        self._output = output
        self._ready = ready
        self._source = source
        self._lock = threading.Lock()
        self._complete = False
        self._on_complete = on_complete

    def get_output(self):
        with self._lock:
            if not self._complete:
                self._ready.synchronize()
                self._complete = True
                self._source = None
                self._ready = None
                if self._on_complete is not None:
                    self._on_complete()
                    self._on_complete = None
            return self._output


def codec_async_fallback_reason(runner, state):
    """A narrow supported contract; fall back before modifying runner state."""
    if state is None:
        return "no_execution"
    model = getattr(runner, "model", None)
    if (
        type(model).__name__ != "EasyMagpieCodecForConditionalGeneration"
        or type(model).__module__ != "easymagpie_vllm_omni.codec.model"
    ):
        return "model"
    if not getattr(runner, "_async_chunk", False):
        return "not_chunked"
    if getattr(runner, "speculative_config", None) is not None:
        return "speculative"
    if getattr(runner, "supports_mm_inputs", False) or getattr(runner, "routed_experts_initialized", False):
        return "extra_outputs"
    if getattr(runner, "kv_connector_output", None) is not None or state.ec_connector_output is not None:
        return "connector_output"
    if runner._should_accumulate_full_payload_output():
        return "accumulation"
    parallel = runner.vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
    ):
        return "parallel"
    if getattr(runner, "use_async_scheduling", False):
        cls = runner.vllm_config.scheduler_config.scheduler_cls
        name = cls if isinstance(cls, str) else f"{getattr(cls, '__module__', '')}.{getattr(cls, '__name__', '')}"
        if name != "easymagpie_vllm_omni.scheduler.EasyMagpieAsyncCodecScheduler":
            return "unfenced_scheduler"
    outputs = state.multimodal_outputs
    count = runner.input_batch.num_reqs
    if not isinstance(outputs, Mapping) or set(outputs) != {"model_outputs", "sr"} or count < 1:
        return "payload_keys"
    audio, rates = outputs["model_outputs"], outputs["sr"]
    if not isinstance(audio, list) or not isinstance(rates, list) or len(audio) != count or len(rates) != count:
        return "payload_count"
    if any(
        not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim != 1
        or not x.is_contiguous()
        for x in audio
    ):
        return "pcm_layout"
    if any(x.device != audio[0].device for x in audio):
        return "pcm_device"
    if any(not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.numel() != 1 for x in rates):
        return "sample_rate"
    if len(runner.input_batch.req_ids) != count or set(state.scheduler_output.num_scheduled_tokens) != set(
        runner.input_batch.req_ids
    ):
        return "request_layout"
    return None


def build_codec_async_output(runner, state):
    """Stage one owning PCM buffer, then construct metadata without reading it."""
    audio = state.multimodal_outputs["model_outputs"]
    sizes = [x.numel() for x in audio]
    # torch.cat is deliberate ownership: source PCM may alias graph replay data.
    source = torch.cat([x.detach() for x in audio])
    host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
    stream = getattr(runner, "_codec_pcm_copy_stream", None)
    if stream is None:
        stream = runner._codec_pcm_copy_stream = torch.cuda.Stream(device=source.device)
    produced = torch.cuda.Event()
    produced.record(torch.cuda.current_stream(source.device))
    ready = torch.cuda.Event()
    with torch.cuda.stream(stream):
        stream.wait_event(produced)
        host.copy_(source, non_blocking=True)
        source.record_stream(stream)
        ready.record(stream)
    per_request = [
        {"model_outputs": part, "sr": rate.detach().clone().contiguous()}
        for part, rate in zip(host.split(sizes), state.multimodal_outputs["sr"], strict=True)
    ]
    # All supported keys are final client payloads; upstream partition_payload_list
    # produces inter_stage_outputs=None for exactly this schema.
    output = OmniModelRunnerOutput(
        req_ids=runner.input_batch.req_ids.copy(),
        req_id_to_index=runner.input_batch.req_id_to_index.copy(),
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=per_request,
        inter_stage_outputs=None,
        kv_connector_output=None,
        num_nans_in_logits={},
        cudagraph_stats=state.cudagraph_stats,
        ec_connector_output=None,
    )
    output.omni_connector_output = runner.get_omni_connector_output()
    output.routed_experts = None
    # Clear only after the deferred result owns every value needed by consumers.
    runner.execute_model_state = None
    runner.kv_connector_output = None
    stats = getattr(runner, "_codec_async_stats", None)
    on_complete = None
    if stats is not None:
        lock = runner._codec_async_stats_lock
        with lock:
            stats["pending_outputs"] += 1
            stats["peak_pending_outputs"] = max(stats["peak_pending_outputs"], stats["pending_outputs"])
            stats["pcm_bytes"] += source.numel() * source.element_size()

        def on_complete():
            with lock:
                stats["pending_outputs"] -= 1
                stats["completed_outputs"] += 1

    return DeferredCodecOutput(output, ready, source, on_complete)
