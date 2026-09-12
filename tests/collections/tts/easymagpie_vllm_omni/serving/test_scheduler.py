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
"""Tests for EasyMagpie streaming accounting on vLLM-Omni 0.26."""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.scheduler import (
    EasyMagpieARAsyncScheduler,
    EasyMagpieCodecScheduler,
    _poll_native_codec_chunk,
)
from easymagpie_vllm_omni.stage_processors import talker2code2wav_async_chunk
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def test_segment_stop_accounting_uses_upstream_026():
    assert EasyMagpieARAsyncScheduler.update_from_output is OmniARAsyncScheduler.update_from_output
    assert EasyMagpieARAsyncScheduler._update_request_with_output is OmniARAsyncScheduler._update_request_with_output


def test_final_streaming_sentinel_marks_session_non_resumable(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque([None]))

    def fake_handle_stopped_request(self, req):
        assert req.resumable is False
        return True

    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", fake_handle_stopped_request)

    assert scheduler._handle_stopped_request(request) is True
    assert request.resumable is False


def test_empty_streaming_queue_remains_resumable_while_waiting(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque())
    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", lambda self, req: False)

    assert scheduler._handle_stopped_request(request) is False
    assert request.resumable is True


@pytest.mark.parametrize("tail_frames", [0, 1, 3])
def test_late_streaming_sentinel_flushes_codec_tail_without_resuming(monkeypatch, tail_frames):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    request = Request("late-close", [0], SamplingParams(max_tokens=35), None, resumable=True)
    request.external_req_id = request.request_id
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.streaming_queue = deque()
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue([request])
    scheduler.num_waiting_for_streaming_input = 1
    saved = []
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        save_async=lambda multimodal_output, request: saved.append(
            {"multimodal_output": multimodal_output, "request": request}
        )
    )
    monkeypatch.setattr(OmniARAsyncScheduler, "finish_requests", Scheduler.finish_requests)
    monkeypatch.setattr(scheduler, "_free_request", lambda req, **kwargs: scheduler.requests.pop(req.request_id))

    manager = SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=0)),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 8}}),
        code_prompt_token_ids=defaultdict(list),
    )
    # Hold a segment payload as if the asynchronous sender had not consumed it yet.
    frame = {"audio_codes": torch.tensor([[7, 8]])}
    for _ in range(max(0, tail_frames - 1)):
        assert talker2code2wav_async_chunk(manager, frame, request) is None

    sentinel = Request(request.request_id, [0], SamplingParams(max_tokens=1), None, resumable=False)
    scheduler.add_request(sentinel)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert scheduler.num_waiting_for_streaming_input == 0
    assert not scheduler.requests and not scheduler.running
    assert not scheduler.waiting and not scheduler.skipped_waiting
    assert len(saved) == 1
    terminal = saved[0]["request"]
    assert terminal.is_finished() and not terminal.resumable
    # Closing must not turn queued segment callbacks into premature terminal flushes.
    assert request.resumable
    segment = talker2code2wav_async_chunk(manager, frame if tail_frames else None, request, is_finished=True)
    if tail_frames:
        assert segment is None
    else:
        assert segment.codes.audio.numel() == 0
    payload = talker2code2wav_async_chunk(manager, saved[0]["multimodal_output"], terminal, is_finished=True)
    assert bool(payload.meta.finished)
    assert payload.codes.audio.numel() == (16 if tail_frames else 0)
    if tail_frames:
        torch.testing.assert_close(payload.codes.audio[:tail_frames], torch.tensor([[7, 8]] * tail_frames))
    assert request.external_req_id not in manager._emp_frame_buffer


@pytest.mark.parametrize("kind", ["new", "running", "resume", "abort"])
def test_late_close_override_leaves_other_admissions_unchanged(monkeypatch, kind):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    request = Request(
        "request",
        [0],
        SamplingParams(max_tokens=1),
        None,
        resumable=kind == "resume",
        abort_immediately=kind == "abort",
    )
    status = RequestStatus.RUNNING if kind == "running" else RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.requests = (
        {}
        if kind == "new"
        else {request.request_id: SimpleNamespace(status=status, resumable=True, request_id=request.request_id)}
    )
    if kind == "abort":
        monkeypatch.setattr(
            scheduler, "finish_requests", lambda *args: pytest.fail("abort must be delegated upstream")
        )
    forwarded = []
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", lambda self, req: forwarded.append(req))

    scheduler.add_request(request)

    assert forwarded == [request]


@pytest.mark.parametrize("outstanding", [0, 1, 2, 4])
def test_resume_uses_exact_discard_count_and_forwards_chunk_metadata(monkeypatch, outstanding):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    session = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=outstanding,
        num_tokens=23,
        max_tokens=1,
        additional_information={"text_token": [1]},
    )
    update = SimpleNamespace(max_tokens=5, additional_information={"text_token": [2, 3]})

    def fake_update_request_as_session(self, req, streaming_update):
        req.async_tokens_to_discard = int(req.num_output_placeholders > 0)
        req.num_computed_tokens -= req.num_output_placeholders
        req.num_output_placeholders = 0

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_as_session", fake_update_request_as_session)

    scheduler._update_request_as_session(session, update)

    assert session.async_tokens_to_discard == outstanding
    assert session.num_computed_tokens == 20 - outstanding
    assert session.max_tokens == 5
    assert session.additional_information == {"text_token": [2, 3]}


@pytest.mark.parametrize("computed", [5, 6])
def test_resume_recomputes_last_token_without_replacing_codec_metadata(monkeypatch, computed):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=1))
    session = SimpleNamespace(
        num_computed_tokens=computed,
        num_output_placeholders=0,
        num_tokens=5,
        max_tokens=7,
        additional_information={"codes": {"audio": "cached"}},
    )
    update = SimpleNamespace(max_tokens=None, additional_information={"text_token": [2]})
    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_as_session", lambda *args: None)

    scheduler._update_request_as_session(session, update)

    assert session.num_computed_tokens == 4
    assert session.max_tokens == 7
    assert session.additional_information == {"codes": {"audio": "cached"}}


def test_native_codec_chunk_appends_prompt_without_resetting_state(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter.get_req_chunk = {"request": 1}
    request = SimpleNamespace(
        prompt_token_ids=[0, 0],
        request_id="request",
        _all_token_ids=[0, 0],
        num_computed_tokens=2,
        num_prompt_tokens=2,
        additional_information=None,
        update_block_hashes=lambda: None,
    )

    def fake_poll(self, req):
        req.prompt_token_ids = [0]
        req._all_token_ids[:] = []
        req.num_computed_tokens = 0
        req.additional_information = {"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}}
        return True

    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", fake_poll)

    assert _poll_native_codec_chunk(adapter, request) is True
    assert request.prompt_token_ids == [0, 0, 0, 0, 0]
    assert request._all_token_ids == [0, 0, 0, 0, 0]
    assert request.num_prompt_tokens == 5
    assert request.num_computed_tokens == 2
    assert request.additional_information["codes"]["audio"].shape == (3, 2)

    prewarm_request = SimpleNamespace(
        prompt_token_ids=[0],
        request_id="request",
        _all_token_ids=[0],
        num_computed_tokens=0,
        num_prompt_tokens=1,
        additional_information=None,
        update_block_hashes=lambda: None,
    )
    assert _poll_native_codec_chunk(adapter, prewarm_request) is True
    assert prewarm_request.prompt_token_ids == [0, 0, 0]
    assert prewarm_request._all_token_ids == [0, 0, 0]
    assert prewarm_request.num_computed_tokens == 0


def test_native_codec_empty_segment_marker_does_not_reset_state(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    request = SimpleNamespace(
        prompt_token_ids=[0, 0, 0],
        request_id="request",
        _all_token_ids=[0, 0, 0],
        num_computed_tokens=3,
        num_prompt_tokens=3,
        additional_information={"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}},
        update_block_hashes=lambda: None,
    )

    def fake_poll(self, req):
        # The base adapter resets these fields before deciding that an empty
        # non-terminal segment marker is not a schedulable codec chunk.
        req.prompt_token_ids = []
        req._all_token_ids[:] = []
        req.num_prompt_tokens = 0
        req.num_computed_tokens = 0
        req.additional_information = {"meta": {"is_segment_finished": True}}
        return False

    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", fake_poll)

    assert _poll_native_codec_chunk(adapter, request) is False
    assert request.prompt_token_ids == [0, 0, 0]
    assert request._all_token_ids == [0, 0, 0]
    assert request.num_prompt_tokens == 3
    assert request.num_computed_tokens == 3


def test_native_codec_loaded_empty_segment_marker_is_not_scheduled(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter._finished_load_reqs = {"request"}
    adapter.upstream_exhausted_requests = set()
    request = SimpleNamespace(
        prompt_token_ids=[0, 0, 0],
        request_id="request",
        _all_token_ids=[0, 0, 0],
        num_computed_tokens=3,
        num_prompt_tokens=3,
        additional_information={"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}},
        update_block_hashes=lambda: None,
    )

    def fake_poll(self, req):
        req.prompt_token_ids = []
        req._all_token_ids[:] = []
        req.num_prompt_tokens = 0
        req.num_computed_tokens = 0
        req.additional_information = {"meta": {"is_segment_finished": True}}
        return True

    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", fake_poll)

    assert _poll_native_codec_chunk(adapter, request) is False
    assert request.prompt_token_ids == [0, 0, 0]
    assert request._all_token_ids == [0, 0, 0]
    assert request.num_prompt_tokens == 3
    assert request.num_computed_tokens == 3
    assert adapter._finished_load_reqs == set()


@pytest.mark.parametrize("marker", ["finished", "is_segment_finished"])
@pytest.mark.parametrize("audio_shape", [None, (0,), (0, 2)], ids=["absent", "empty-flat", "empty-rows"])
@pytest.mark.parametrize("queue_method", ["_process_chunk_queue_legacy", "_process_chunk_queue"])
def test_native_codec_control_marker_queue_liveness(marker, audio_shape, queue_method):
    payload = {"meta": {marker: True}}
    if audio_shape is not None:
        payload["codes"] = {"audio": torch.empty(audio_shape, dtype=torch.long)}
    adapter, request = _codec_poll_state([payload])
    ref = request.additional_information["codes"]["ref"]
    terminal = marker == "finished"

    received = _poll_native_codec_chunk(adapter, request)
    assert (request.request_id in adapter.upstream_exhausted_requests) is terminal
    assert request.resumable is not terminal
    assert request.prompt_token_ids == [0, 0, 0]
    assert request._all_token_ids == [0, 0, 0]
    assert request.num_prompt_tokens == 3
    assert request.num_computed_tokens == 3
    audio = request.additional_information["codes"].get("audio")
    assert audio is None or audio.numel() == 0
    assert request.additional_information["codes"]["ref"] is ref
    assert request.additional_information["meta"]["chunk_seq"] == 1

    # Exercise the real upstream queue gate, not just the poll return value.
    running, parked = [request], deque()
    getattr(adapter, queue_method)(running, parked, RequestStatus.RUNNING, adapter._finished_load_reqs)
    assert running == ([request] if terminal else [])
    assert list(parked) == ([] if terminal else [request])
    assert request.status == (RequestStatus.RUNNING if terminal else RequestStatus.WAITING_FOR_CHUNK)
    assert received is terminal
    assert request.request_id not in adapter._finished_load_reqs
    # A terminal marker is runnable for completion, with no codec tokens to execute.
    assert len(request.prompt_token_ids) - request.num_computed_tokens == 0


@pytest.mark.parametrize("queue_method", ["_process_chunk_queue_legacy", "_process_chunk_queue"])
def test_native_codec_segment_then_audio_then_terminal_keeps_state(queue_method):
    frames = torch.tensor([[1, 2], [3, 4]])
    adapter, request = _codec_poll_state(
        [{"meta": {"is_segment_finished": True}}, {"codes": {"audio": frames}}, {"meta": {"finished": True}}]
    )
    assert _poll_native_codec_chunk(adapter, request) is False
    assert request.request_id not in adapter._finished_load_reqs

    assert _poll_native_codec_chunk(adapter, request) is True
    assert request.prompt_token_ids == [0] * 5
    assert request._all_token_ids == [0] * 5
    assert request.num_computed_tokens == 3
    assert torch.equal(request.additional_information["codes"]["audio"], frames)
    running, parked = [request], deque()
    getattr(adapter, queue_method)(running, parked, RequestStatus.RUNNING, adapter._finished_load_reqs)
    assert running == [request]
    assert not parked

    # Simulate consuming the actual two-frame payload before the final control arrives.
    request.num_computed_tokens = 5
    request.status = RequestStatus.WAITING_FOR_CHUNK
    adapter.requests_with_ready_chunks.clear()
    adapter.segment_finished_requests.clear()
    assert _poll_native_codec_chunk(adapter, request) is True
    getattr(adapter, queue_method)(running, parked, RequestStatus.RUNNING, adapter._finished_load_reqs)
    assert running == [request]
    assert not parked
    assert request.status == RequestStatus.RUNNING
    assert request.num_prompt_tokens == request.num_computed_tokens == 5
    assert request.prompt_token_ids == request._all_token_ids == [0] * 5
    assert "audio" not in request.additional_information["codes"]
    assert request.resumable is False


def test_native_codec_poll_without_arrival_preserves_metadata(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    info = {"codes": {"audio": torch.ones(3, 2)}, "meta": {"chunk_seq": 1}}
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0, 0, 0],
        _all_token_ids=[0, 0, 0],
        num_computed_tokens=3,
        num_prompt_tokens=3,
        additional_information=info,
        update_block_hashes=lambda: None,
    )
    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", lambda *args: False)

    assert _poll_native_codec_chunk(adapter, request) is False
    assert request.additional_information is info
    assert request.num_computed_tokens == 3
    assert request.prompt_token_ids == [0, 0, 0]


def test_native_codec_streaming_update_preserves_state_and_resumes_polling():
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(segment_finished_requests={"request"})
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.log_stats = False

    class _SkippedWaiting(list):
        def remove_requests(self, requests):
            for request in requests:
                self.remove(request)

    original_prompt = [0, 0, 0, 0, 0]
    original_all_tokens = [0, 0, 0, 0, 0]
    session = SimpleNamespace(
        request_id="request",
        prompt_token_ids=original_prompt,
        _all_token_ids=original_all_tokens,
        num_prompt_tokens=5,
        num_computed_tokens=5,
        additional_information={"codes": {"audio": torch.ones((2, 2), dtype=torch.long)}},
        arrival_time=1.0,
        sampling_params="old",
        _output_token_ids=[],
        update_block_hashes=lambda: None,
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
    )
    scheduler.skipped_waiting = _SkippedWaiting([session])
    enqueued = []
    scheduler._enqueue_waiting_request = enqueued.append
    update = SimpleNamespace(
        prompt_token_ids=[0],
        additional_information=None,
        arrival_time=2.0,
        sampling_params="new",
    )

    scheduler._update_request_as_session(session, update)

    assert session.prompt_token_ids is original_prompt
    assert session._all_token_ids is original_all_tokens
    assert session.num_prompt_tokens == 5
    assert session.num_computed_tokens == 5
    assert session.additional_information["codes"]["audio"].shape == (2, 2)
    assert session.arrival_time == 2.0
    assert session.sampling_params == "new"
    assert session.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()
    assert scheduler.skipped_waiting == []
    assert enqueued == [session]


@pytest.mark.parametrize(
    ("status", "queue_name", "num_waiting"),
    [
        (RequestStatus.WAITING, "waiting", 0),
        (RequestStatus.WAITING_FOR_STREAMING_REQ, "skipped_waiting", 1),
    ],
)
def test_native_codec_segment_resume_stays_on_cached_request_path(status, queue_name, num_waiting):
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(segment_finished_requests={"request"})
    scheduler.num_waiting_for_streaming_input = num_waiting
    scheduler.running = []

    class _RequestQueue(list):
        def remove_requests(self, requests):
            for request in requests:
                self.remove(request)

    session = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 6,
        _all_token_ids=[0] * 6,
        num_prompt_tokens=6,
        num_computed_tokens=6,
        status=status,
    )
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue()
    getattr(scheduler, queue_name).append(session)

    scheduler._resume_codec_after_segment(session)

    assert scheduler.waiting == []
    assert scheduler.skipped_waiting == []
    assert scheduler.running == [session]
    assert session.status == RequestStatus.RUNNING
    assert session.prompt_token_ids == [0] * 6
    assert session._all_token_ids == [0] * 6
    assert session.num_prompt_tokens == 6
    assert session.num_computed_tokens == 6
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()


def _codec_poll_state(payloads):
    """Use the actual upstream poll and queues with only connector I/O stubbed."""
    payloads = deque(payloads)
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter._finished_load_reqs = set()
    adapter.get_req_chunk = defaultdict(int)
    adapter.request_ids_mapping = {}
    adapter.model_mode = "generation"
    adapter.upstream_exhausted_requests = set()
    adapter.segment_finished_requests = set()
    adapter.requests_with_ready_chunks = set()
    adapter.requests_origin_status = {}
    adapter._active_window = 32
    adapter._active_streams = {}
    adapter.connector = SimpleNamespace(stage_id=1, get=lambda *_args: (payloads.popleft(), 1))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0, 0, 0],
        _all_token_ids=[0, 0, 0],
        num_computed_tokens=3,
        num_prompt_tokens=3,
        resumable=True,
        prefill_stats=None,
        status=RequestStatus.WAITING_FOR_CHUNK,
        additional_information={
            "codes": {"audio": torch.ones((3, 2), dtype=torch.long), "ref": torch.tensor([0.1, -0.1])},
            "meta": {"chunk_seq": 1},
        },
        update_block_hashes=lambda: None,
    )
    return adapter, request
