# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Codec history must cover the LM context, including terminal padding."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.worker.gpu_input_batch as input_batch_module
import yaml
from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from easymagpie_vllm_omni.stage_processors import talker2code2wav_async_chunk
from vllm import SamplingParams
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

_PROFILES = [EASYMAGPIE_ROOT / "deploy" / "easymagpie.yaml"]
_H100 = EASYMAGPIE_ROOT / "deploy" / "easymagpie_h100.yaml"
if _H100.is_file():
    _PROFILES.append(_H100)


@pytest.mark.parametrize("path", _PROFILES, ids=lambda path: path.stem)
def test_codec_history_covers_lm_context_and_terminal_chunk(path):
    profile = yaml.safe_load(path.read_text())
    lm, codec = profile["stages"]
    chunks = [connector["extra"]["codec_chunk_frames"] for connector in profile["connectors"].values()]
    required = lm["max_model_len"] + max(chunks)
    assert codec["max_model_len"] >= required
    assert codec["enable_chunked_prefill"] is False
    SchedulerConfig(
        max_model_len=codec["max_model_len"],
        is_encoder_decoder=False,
        max_num_seqs=codec["max_num_seqs"],
        max_num_batched_tokens=codec["max_num_batched_tokens"],
        enable_chunked_prefill=False,
    )


@pytest.mark.parametrize("path", _PROFILES, ids=lambda path: path.stem)
def test_profile_worker_accepts_long_codec_history(path, monkeypatch):
    profile = yaml.safe_load(path.read_text())
    lm, codec = profile["stages"]
    required = lm["max_model_len"] + max(
        connector["extra"]["codec_chunk_frames"] for connector in profile["connectors"].values()
    )
    for rows in (513, 521, 2046, required - 1, required):
        runner = _runner(monkeypatch, codec["max_num_seqs"], codec["max_model_len"], codec["max_num_batched_tokens"])
        _update(runner, rows)
        assert int(runner.input_batch.num_prompt_tokens[0]) == rows


@pytest.mark.parametrize("limit,budget", [(512, 512), (520, 520), (520, 1536)])
def test_upstream_copy_rejects_history_overflow(monkeypatch, limit, budget):
    _update(_runner(monkeypatch, 32, limit, budget), limit)
    with pytest.raises(ValueError, match="could not broadcast input array"):
        _update(_runner(monkeypatch, 32, limit, budget), limit + 1)


@pytest.mark.parametrize("budget", [512, 520, 1536])
def test_atomic_scheduler_rejects_budget_smaller_than_history(budget):
    with pytest.raises(ValueError, match="smaller than max_model_len"):
        SchedulerConfig(
            max_model_len=4104,
            is_encoder_decoder=False,
            max_num_seqs=32,
            max_num_batched_tokens=budget,
            enable_chunked_prefill=False,
        )


@pytest.mark.parametrize("cap", [514, 515, 2048])
@pytest.mark.parametrize("segment_size", [1, 5])
def test_producer_counts_prefill_delay_and_terminal_padding(cap, segment_size):
    manager = SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=5)),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 8, "codec_startup_chunk_frames": [6, 6]}}),
        code_prompt_token_ids=defaultdict(list),
    )
    request = SimpleNamespace(
        external_req_id="history",
        finished=False,
        resumable=True,
        additional_information={"text_prefill_num": 4},
    )
    request.is_finished = lambda: request.finished
    assert talker2code2wav_async_chunk(manager, {"audio_codes": torch.zeros(1, 2, dtype=torch.long)}, request) is None
    chunks = []
    for frame in range(1, cap):  # One sampled prefill token has no acoustic row.
        request.finished = frame % segment_size == 0 or frame == cap - 1
        request.resumable = frame != cap - 1
        payload = talker2code2wav_async_chunk(
            manager,
            {"audio_codes": torch.tensor([[frame, frame]])},
            request,
        )
        if payload is not None and payload.codes.audio.numel():
            chunks.append(payload.codes.audio)
            if request.resumable:
                assert chunks[-1].shape[0] == (6 if len(chunks) <= 2 else 8)
                assert bool((chunks[-1] >= 0).all())
        if request.finished:
            manager.code_prompt_token_ids.pop(request.external_req_id, None)
    rows = torch.cat(chunks)
    real = rows[(rows >= 0).all(dim=1)]
    torch.testing.assert_close(real[:, 0], torch.arange(2, cap))
    assert len(real) == cap - 2  # Prefill plus the one remaining warm-up row.
    assert 0 <= len(rows) - len(real) < 8
    assert request.external_req_id not in manager._emp_seen_frames
    terminal = talker2code2wav_async_chunk(manager, None, request)
    assert terminal is None or terminal.codes.audio.numel() == 0


def test_upstream_ws_resume_retains_cumulative_lm_history():
    scheduler = EasyMagpieARAsyncScheduler.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    scheduler.chunk_transfer_adapter = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler.log_stats = False
    request = Request("history", [0] * 71, SamplingParams(max_tokens=2048), None, resumable=True)
    outputs = 0
    for count in (1, 5, 512, 511, 1000):
        request.append_output_token_ids([0] * count)
        request.num_computed_tokens = request.num_tokens - 1
        outputs += count
        computed = request.num_computed_tokens
        update = SimpleNamespace(
            prompt_token_ids=[0],
            mm_features=[],
            arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=count),
            max_tokens=count,
            additional_information={"text_token": [1]},
            model_intermediate_buffer=None,
        )
        scheduler._update_request_as_session(request, update)
        assert request.num_computed_tokens == computed == 71 + outputs - 1
        assert request.num_prompt_tokens == computed + 1
        assert request.num_output_tokens == 0
    assert request.num_computed_tokens > 512


def _runner(monkeypatch, max_seqs, limit, budget):
    # Driver-free CPU fixture: preserve the actual upstream allocation dimensions.
    monkeypatch.setattr(input_batch_module, "PIN_MEMORY", False)
    batch = InputBatch(
        max_num_reqs=max_seqs,
        max_model_len=limit,
        max_num_batched_tokens=budget,
        device=torch.device("cpu"),
        vocab_size=1025,
        block_sizes=[],
        kernel_block_sizes=[],
        max_num_blocks_per_req=[],
    )
    state = CachedRequestState(
        req_id="history",
        prompt_token_ids=[0] * 508,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=(),
        num_computed_tokens=508,
        output_token_ids=[],
    )
    batch.add_request(state)
    runner = GPUGenerationModelRunner.__new__(GPUGenerationModelRunner)
    runner.input_batch, runner.requests, runner.uses_mrope = batch, {state.req_id: state}, False
    return runner


def _update(runner, rows):
    cached = SimpleNamespace(
        resumed_req_ids=set(),
        req_ids=["history"],
        prompt_token_ids={"history": [0] * rows},
    )
    runner._update_request_states(
        SimpleNamespace(finished_req_ids=set(), num_scheduled_tokens={"history": 1}, scheduled_cached_reqs=cached)
    )
