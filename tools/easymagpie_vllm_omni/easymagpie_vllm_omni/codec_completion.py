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
"""Preserve codec terminal outputs regardless of frontend input timing."""

from functools import wraps


def install_codec_completion() -> None:
    """Attach the codec-only compatibility hook before stage pools are built."""
    from vllm_omni.engine.stage_pool import StagePool

    _install_terminal_prewarm()
    original_init = StagePool.__init__
    if getattr(original_init, "_easymagpie_codec_completion", False) is True:
        return

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        model_config = getattr(self.stage_vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if getattr(hf_config, "model_type", None) == "easymagpie_codec" and self.output_processor is not None:
            _patch_processor(self.output_processor)

    init._easymagpie_codec_completion = True
    StagePool.__init__ = init


def _patch_processor(processor):
    original_update = processor._update_stats_from_output
    if getattr(original_update, "_easymagpie_codec_completion", False) is True:
        return

    @wraps(original_update)
    def update_stats(request_state, output, *args, **kwargs):
        result = original_update(request_state, output, *args, **kwargs)
        # This callback runs per raw output, even with logging disabled, before
        # upstream creates the output and performs its normal terminal cleanup.
        # The codec's true final can arrive before the frontend's final input.
        if (
            request_state.streaming_input
            and output.finish_reason is not None
            and getattr(output, "is_segment_finished", None) is False
        ):
            request_state.streaming_input = False
        return result

    update_stats._easymagpie_codec_completion = True
    processor._update_stats_from_output = update_stats


def _install_terminal_prewarm():
    from vllm_omni.engine.orchestrator import Orchestrator

    original = Orchestrator._prewarm_async_chunk_stages
    if getattr(original, "_easymagpie_terminal_prewarm", False) is True:
        return

    @wraps(original)
    async def prewarm(self, request_id, stage0_request, request_state):
        if (
            self.async_chunk
            and len(self.stage_pools) == 2
            and request_state.final_stage_id == 1
            and request_state.streaming.enabled
            and getattr(stage0_request, "resumable", None) is False
        ):
            model_config = getattr(self.stage_pools[1].stage_vllm_config, "model_config", None)
            config = getattr(model_config, "hf_config", None)
            if getattr(config, "model_type", None) == "easymagpie_codec":
                # The connector delivers the codec's final control. Another
                # dummy request could recreate state after it has completed.
                return
        return await original(self, request_id, stage0_request, request_state)

    prewarm._easymagpie_terminal_prewarm = True
    Orchestrator._prewarm_async_chunk_stages = prewarm
