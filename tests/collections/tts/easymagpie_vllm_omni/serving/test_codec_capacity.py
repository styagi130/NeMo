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
"""Stateful codec capacity exercises real scheduling, polling and release."""

import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.model import EasyMagpieCodecForConditionalGeneration
from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE
from easymagpie_vllm_omni.scheduler import EasyMagpieCodecScheduler, _poll_native_codec_chunk
from vllm import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    initialized = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        pytest.fail("codec scheduling must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("capacity", [1, 32, 128])
@pytest.mark.parametrize("release", ["legacy-false", "terminal", "abort"])
def test_parked_codec_streams_retain_capacity(capacity, release):
    codec_stage = next(stage for stage in EASYMAGPIE_PIPELINE.stages if stage.stage_id == 1)
    legacy = release == "legacy-false"
    scheduler, adapter, payloads, loads = _scheduler(
        capacity, False if legacy else codec_stage.retains_state_across_chunks
    )
    groups = {name: {f"{name}{index}" for index in range(capacity)} for name in ("a", "b", "c")}

    # A computed its first six rows. Without retention accounting, parking A
    # admits B and then C, even though the predecessors still own codec state.
    for name in ("b", "c"):
        output = _step(scheduler)
        assert set(output.num_scheduled_tokens) == (groups[name] if legacy else set())
    for request_id in dict.fromkeys(loads):
        request = scheduler.requests[request_id]
        if request.num_computed_tokens:
            _deliver(adapter, request, payloads, frames=6)
            assert request.num_computed_tokens == 6
            assert len(request.prompt_token_ids) == 12

    output = _step(scheduler)
    assert set(output.num_scheduled_tokens) == groups["b" if legacy else "a"]
    assert set(output.num_scheduled_tokens.values()) == {6}
    if legacy:
        assert {req.request_id for req in scheduler.waiting} == groups["a"]
        assert all(scheduler.requests[rid].status == RequestStatus.PREEMPTED for rid in groups["a"])
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == dict.fromkeys(groups["a"], 12)
        with pytest.raises(ValueError, match="scheduled 12 placeholders.*payload has 6 frames"):
            _validate_payloads(scheduler, output)
        return

    assert {req.request_id for req in scheduler.waiting} == groups["b"] | groups["c"]
    assert not _step(scheduler).num_scheduled_tokens
    for request_id in groups["a"]:
        request = scheduler.requests[request_id]
        _deliver(adapter, request, payloads, frames=8)
        assert request.num_computed_tokens == 12
        assert len(request.prompt_token_ids) == 20
    assert _step(scheduler).num_scheduled_tokens == dict.fromkeys(groups["a"], 8)
    assert not _step(scheduler).num_scheduled_tokens

    # Free actual parked requests, not just their queue entries. Terminal input
    # follows native polling and zero-token completion; abort uses finish_requests.
    requests = [scheduler.requests[rid] for rid in groups["a"]]
    if release == "terminal":
        for request in requests:
            _deliver(adapter, request, payloads, frames=None)
            assert not request.resumable
        assert not _step(scheduler).num_scheduled_tokens
        assert all(request.status == RequestStatus.FINISHED_STOPPED for request in requests)
    else:
        scheduler.finish_requests(groups["a"], RequestStatus.FINISHED_ABORTED)
        assert all(request.status == RequestStatus.FINISHED_ABORTED for request in requests)
    assert not groups["a"] & scheduler.requests.keys()
    for manager in (scheduler.kv_cache_manager, scheduler.encoder_cache_manager):
        assert {call.args[0].request_id for call in manager.free.call_args_list} == groups["a"]
        assert manager.free.call_count == capacity
    assert _step(scheduler).num_scheduled_tokens == dict.fromkeys(groups["b"], 6)
    assert {req.request_id for req in scheduler.waiting} == groups["c"]


def _step(scheduler):
    output = scheduler.schedule()
    _validate_payloads(scheduler, output)
    request_ids = list(output.num_scheduled_tokens)
    runner_output = SimpleNamespace(
        sampled_token_ids=[],
        req_id_to_index={rid: index for index, rid in enumerate(request_ids)},
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        routed_experts=None,
    )
    scheduler.update_from_output(output, runner_output)
    assert all(request.num_in_flight_tokens == 0 for request in scheduler.requests.values())
    return output


def _validate_payloads(scheduler, output):
    entries = {req.req_id: req.additional_information for req in output.scheduled_new_reqs}
    entries.update(output.scheduled_cached_reqs.additional_information)
    infos, spans, expected = [], [], []
    offset = 0
    for request_id, count in output.num_scheduled_tokens.items():
        info = entries[request_id]
        audio = scheduler.requests[request_id].additional_information["codes"]["audio"]
        assert info["codes"]["audio"] is audio
        infos.append(info)
        expected.append(audio)
        spans.append((offset, offset + count))
        offset += count
    codec = object.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(codec)
    codec.config = EasyMagpieCodecConfig(input_dim=5, num_codebooks=1)
    packed, frames, valid_samples = codec._payload_codes(infos, torch.device("cpu"), spans)
    assert frames == list(output.num_scheduled_tokens.values())
    assert valid_samples == [count * codec.config.samples_per_frame for count in frames]
    if expected:
        torch.testing.assert_close(packed, torch.cat(expected), rtol=0, atol=0)


def _audio_rows(request_id, start, frames):
    identity = (ord(request_id[0]) - ord("a")) * 128 + int(request_id[1:])
    positions = torch.arange(start, start + frames)
    return torch.stack((torch.full_like(positions, identity), positions), dim=1)


def _deliver(adapter, request, payloads, frames):
    key = f"{request.request_id}_0_{adapter.get_req_chunk[request.request_id]}"
    if frames is None:
        payloads[key] = ({"meta": {"finished": True}}, 1)
    else:
        audio = _audio_rows(request.request_id, request.num_computed_tokens, frames)
        payloads[key] = ({"codes": {"audio": audio}}, audio.numel() * audio.element_size())
    assert _poll_native_codec_chunk(adapter, request)
    if frames is not None:
        assert request.additional_information["codes"]["audio"] is audio


def _scheduler(capacity, retains_state):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter.receives_chunks = True
    adapter.scheduler_max_num_seqs = capacity
    adapter._active_window = 0
    for name in ("_held_non_active", "waiting_for_chunk_waiting_requests", "waiting_for_chunk_running_requests"):
        setattr(adapter, name, deque())
    for name in (
        "_active_streams",
        "requests_origin_status",
        "request_ids_mapping",
        "request_payload",
        "code_prompt_token_ids",
        "_pending_streaming_prefills",
    ):
        setattr(adapter, name, {})
    for name in ("get_req_chunk", "put_req_chunk", "requests_num_chunks_sent", "ramp_chunk_count"):
        setattr(adapter, name, defaultdict(int))
    for name in (
        "requests_with_ready_chunks",
        "_finished_load_reqs",
        "_cancelled_load_reqs",
        "upstream_exhausted_requests",
        "segment_finished_requests",
    ):
        setattr(adapter, name, set())
    adapter.model_mode = "generation"
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
    payloads, loads = {}, []
    adapter.connector = SimpleNamespace(stage_id=1, get=lambda source, target, key: payloads.pop(key, None))
    adapter.load_async = lambda request: loads.append(request.request_id)

    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler._codec_startup_wait_s = scheduler._codec_busy_wait_s = 0
    scheduler.chunk_transfer_adapter = adapter
    scheduler.max_num_running_reqs = capacity
    scheduler.max_num_scheduled_tokens = 4104
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler._retains_state_across_chunks = retains_state
    scheduler._pending_finish_reqs = []
    scheduler.input_coordinator = scheduler._latest_omni_connector_output = None
    scheduler.perf_metrics = scheduler.finished_req_ids_dict = None
    scheduler.log_stats = scheduler.use_v2_model_runner = scheduler.use_pp = False
    scheduler.num_lookahead_tokens = scheduler.num_waiting_for_streaming_input = 0
    scheduler.needs_kv_cache_zeroing = scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler._inflight_prefills = set()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.connector = scheduler.ec_connector = None
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler.kv_cache_manager = SimpleNamespace(
        new_step_starts=lambda: None,
        allocate_slots=lambda *args, **kwargs: SimpleNamespace(get_block_ids=lambda **kwargs: ([],)),
        get_num_common_prefix_blocks=lambda request_id: [],
        take_events=lambda: None,
        free=Mock(),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [], free=Mock())
    scheduler.requests, scheduler.running = {}, []
    scheduler.waiting, scheduler.skipped_waiting = FCFSRequestQueue(), FCFSRequestQueue()
    for group in ("a", "b", "c"):
        for index in range(capacity):
            request_id = f"{group}{index}"
            request = Request(request_id, [0] * 6, SamplingParams(max_tokens=65536), None, resumable=True)
            request.external_req_id = request_id
            audio = _audio_rows(request_id, 0, 6)
            request.additional_information = {"codes": {"audio": audio}}
            request.status = RequestStatus.RUNNING if group == "a" else RequestStatus.WAITING
            request.num_computed_tokens = 6 if group == "a" else 0
            scheduler.requests[request_id] = request
            if group == "a":
                scheduler.running.append(request)
            else:
                scheduler.waiting.add_request(request)
                adapter.requests_with_ready_chunks.add(request_id)
    return scheduler, adapter, payloads, loads
