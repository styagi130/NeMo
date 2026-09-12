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
from easymagpie_vllm_omni import scheduler as scheduler_module
from easymagpie_vllm_omni.scheduler import (
    EasyMagpieARAsyncScheduler,
    EasyMagpieCodecScheduler,
    _poll_native_codec_chunk,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import RequestStatus
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
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
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
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
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
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
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
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
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


@pytest.mark.parametrize(
    ("key", "maximum", "codec", "attribute"),
    [
        ("stage0_admission_coalesce_ms", 50, False, "_admission_wait_s"),
        ("codec_startup_coalesce_ms", 2, True, "_codec_startup_wait_s"),
        ("codec_busy_coalesce_ms", 4, True, "_codec_busy_wait_s"),
    ],
)
def test_coalescing_wait_bounds_and_disabled_defaults(monkeypatch, key, maximum, codec, attribute):
    assert getattr(_policy_scheduler(monkeypatch, codec=codec), attribute) == 0
    for value in (0, maximum):
        scheduler = _policy_scheduler(monkeypatch, codec=codec, extra={key: value})
        assert getattr(scheduler, attribute) == value / 1000
    for value in (-1, maximum + 0.001, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=key):
            _policy_scheduler(monkeypatch, codec=codec, extra={key: value})


@pytest.mark.parametrize("target", [None, 3])
@pytest.mark.parametrize("wait_ms", [10, 50])
def test_admission_coalescing_has_one_bounded_deadline(monkeypatch, wait_ms, target):
    extra = {"stage0_admission_coalesce_ms": wait_ms}
    if target is not None:
        extra["stage0_admission_batch_target"] = target
    scheduler = _policy_scheduler(monkeypatch, extra=extra)
    scheduler.waiting = [SimpleNamespace(num_computed_tokens=0)]
    deadline = 1.0 + wait_ms / 1000
    times = iter((1.0, deadline - 0.001, deadline, 2.0, 3.0))
    monkeypatch.setattr(scheduler_module, "monotonic", lambda: next(times), raising=False)

    assert scheduler._should_defer_waiting_admission() is True
    scheduler.waiting.append(SimpleNamespace(num_computed_tokens=0))
    assert scheduler._should_defer_waiting_admission() is True
    assert scheduler._admission_deadline == deadline
    assert scheduler._should_defer_waiting_admission() is False
    assert scheduler._should_defer_waiting_admission() is False
    scheduler.waiting.clear()
    assert scheduler._should_defer_waiting_admission() is False
    scheduler.waiting.append(SimpleNamespace(num_computed_tokens=0))
    assert scheduler._should_defer_waiting_admission() is True
    assert scheduler._admission_deadline == 3.0 + wait_ms / 1000


@pytest.mark.parametrize("target", [1, 16, 64, 128])
def test_admission_batch_target_is_capped_without_changing_capacity(monkeypatch, target):
    scheduler = _policy_scheduler(monkeypatch, extra={"stage0_admission_batch_target": target}, max_requests=64)
    assert scheduler._admission_batch_target == min(target, 64)
    assert scheduler.max_num_running_reqs == 64
    assert _policy_scheduler(monkeypatch)._admission_batch_target == 4


@pytest.mark.parametrize("target", [0, -1, True, False, 1.5, "16", None])
def test_admission_batch_target_rejects_nonpositive_or_noninteger_values(monkeypatch, target):
    with pytest.raises(ValueError, match="stage0_admission_batch_target"):
        _policy_scheduler(monkeypatch, extra={"stage0_admission_batch_target": target})


def test_admission_target_releases_early_without_consuming_or_reordering_requests(monkeypatch):
    scheduler = _policy_scheduler(
        monkeypatch,
        extra={"stage0_admission_coalesce_ms": 50, "stage0_admission_batch_target": 16},
        max_requests=64,
    )
    monkeypatch.setattr(scheduler_module, "monotonic", lambda: 1.0)
    scheduler.waiting = [SimpleNamespace(num_computed_tokens=0) for _ in range(15)]
    assert scheduler._should_defer_waiting_admission() is True
    scheduler.waiting.extend(SimpleNamespace(num_computed_tokens=0) for _ in range(4))
    requests = scheduler.waiting.copy()
    assert scheduler._should_defer_waiting_admission() is False
    assert all(a is b for a, b in zip(scheduler.waiting, requests, strict=True))
    assert scheduler._admission_deadline == 0.0
    scheduler.waiting[:] = requests[:1]  # Cancellation must not restart a released wait.
    assert scheduler._should_defer_waiting_admission() is False
    scheduler.waiting.clear()
    assert scheduler._should_defer_waiting_admission() is False
    scheduler.waiting[:] = requests[:1]
    assert scheduler._should_defer_waiting_admission() is True


@pytest.mark.parametrize(("target", "count", "deferred"), [(16, 15, True), (16, 16, False), (64, 16, True)])
def test_admission_target_preserves_upstream_abort_and_exception_restoration(monkeypatch, target, count, deferred):
    scheduler = _policy_scheduler(
        monkeypatch,
        extra={"stage0_admission_coalesce_ms": 50, "stage0_admission_batch_target": target},
        max_requests=64,
    )
    monkeypatch.setattr(scheduler_module, "monotonic", lambda: 1.0)
    scheduler.waiting = [SimpleNamespace(num_computed_tokens=0, status=RequestStatus.WAITING) for _ in range(count)]
    original = scheduler.waiting
    original.append(SimpleNamespace(num_computed_tokens=0, status=RequestStatus.FINISHED_ABORTED))
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.requests = {}
    scheduler.input_coordinator = None
    events = []
    scheduler._consume_pending_connector_output = lambda **kwargs: events.append("consume")
    scheduler._process_pending_input_timeouts = lambda: events.append("timeouts")
    scheduler.chunk_transfer_adapter.process_pending_chunks = lambda *args, **kwargs: events.append("poll")
    scheduler.chunk_transfer_adapter.restore_queues = lambda *args, **kwargs: events.append("restore")

    def base_schedule(self, *args):
        assert events == ["consume", "timeouts", "poll"]
        assert len(original) == count
        assert len(self.waiting) == (0 if deferred else count)
        assert (self.waiting is original) is not deferred
        raise RuntimeError("base schedule failed")

    monkeypatch.setattr(AsyncScheduler, "schedule", base_schedule)
    with pytest.raises(RuntimeError, match="base schedule failed"):
        scheduler.schedule()
    assert scheduler.waiting is original
    assert len(original) == count
    assert events == ["consume", "timeouts", "poll", "restore"]


@pytest.mark.parametrize("target", [None, 1])
@pytest.mark.parametrize("bypass", ["disabled", "empty", "running", "resumed", "full"])
@pytest.mark.parametrize("wait_ms", [10, 50])
def test_admission_coalescing_bypasses_active_resume_and_full_batches(monkeypatch, bypass, wait_ms, target):
    wait_ms = 0 if bypass == "disabled" else wait_ms
    extra = {"stage0_admission_coalesce_ms": wait_ms}
    if target is not None:
        extra["stage0_admission_batch_target"] = target
    scheduler = _policy_scheduler(monkeypatch, extra=extra)
    scheduler.waiting = [SimpleNamespace(num_computed_tokens=int(bypass == "resumed"))]
    scheduler._admission_deadline = 100.0
    if bypass == "empty":
        scheduler.waiting.clear()
    elif bypass == "running":
        scheduler.running = [object()]
    elif bypass == "full":
        scheduler.waiting *= scheduler.max_num_running_reqs

    assert scheduler._should_defer_waiting_admission() is False
    assert scheduler._admission_deadline == (0.0 if bypass == "full" else None)
    if bypass == "full":
        scheduler.waiting.pop()
        assert scheduler._should_defer_waiting_admission() is False
        assert scheduler._admission_deadline == 0.0


@pytest.mark.parametrize(
    ("running", "waiting", "ready", "startup", "busy"),
    [
        ([], [0], {"w0"}, True, False),
        ([], [], set(), False, False),
        ([8, 8], [], {"r0"}, False, True),
        ([8, 8], [0], {"r0", "w0"}, False, False),
        ([8, 8], [], {"r0", "r1"}, False, False),
        ([8], [], {"r0"}, False, False),
        ([8, 8], [], set(), False, False),
    ],
)
def test_codec_coalescing_waits_only_for_eligible_batches(monkeypatch, running, waiting, ready, startup, busy):
    scheduler = _policy_scheduler(
        monkeypatch, codec=True, extra={"codec_startup_coalesce_ms": 2, "codec_busy_coalesce_ms": 4}
    )
    for prefix, values, target in (("r", running, scheduler.running), ("w", waiting, scheduler.waiting)):
        target.extend(
            SimpleNamespace(
                request_id=f"{prefix}{index}",
                num_computed_tokens=tokens,
                status=RequestStatus.WAITING_FOR_CHUNK,
            )
            for index, tokens in enumerate(values)
        )
    scheduler.chunk_transfer_adapter._finished_load_reqs = ready

    assert scheduler._should_coalesce_codec_startup() is startup
    assert scheduler._should_coalesce_codec_busy() is busy
    scheduler._codec_startup_wait_s = scheduler._codec_busy_wait_s = 0
    assert scheduler._should_coalesce_codec_startup() is False
    assert scheduler._should_coalesce_codec_busy() is False


def test_codec_startup_wait_does_not_hold_payload_lock(monkeypatch):
    scheduler = _policy_scheduler(monkeypatch, codec=True, extra={"codec_startup_coalesce_ms": 2})
    scheduler.waiting = [
        SimpleNamespace(request_id="new", num_computed_tokens=0, status=RequestStatus.WAITING_FOR_CHUNK)
    ]
    adapter = scheduler.chunk_transfer_adapter
    adapter._finished_load_reqs.add("new")
    waits = []

    def wait(duration):
        assert adapter._easymagpie_chunk_lock.acquire(blocking=False)
        adapter._easymagpie_chunk_lock.release()
        waits.append(duration)

    monkeypatch.setattr(scheduler_module, "sleep", wait, raising=False)
    monkeypatch.setattr(OmniGenerationScheduler, "schedule", lambda *args, **kwargs: "scheduled")
    assert scheduler.schedule() == "scheduled"
    assert waits == [0.002]


@pytest.mark.parametrize("arrival", ["terminal", "first_chunk"])
def test_codec_busy_wait_releases_lock_and_ready_arrival_wakes_it(monkeypatch, arrival):
    scheduler = _policy_scheduler(monkeypatch, codec=True, extra={"codec_busy_coalesce_ms": 4})
    payload = {"meta": {"finished": True}} if arrival == "terminal" else {"codes": {"audio": torch.ones(1, 2)}}
    adapter, request = _codec_poll_state([payload])
    adapter._finished_load_reqs.add("ready")
    scheduler.chunk_transfer_adapter = adapter
    scheduler.running = [SimpleNamespace(request_id="ready", num_computed_tokens=3), request]
    if arrival == "first_chunk":
        request.num_computed_tokens = 0
        scheduler.running[-1] = SimpleNamespace(request_id="stalled", num_computed_tokens=3)
        scheduler.waiting = [request]
    # Give thread synchronization headroom; production validation caps this at 4 ms.
    scheduler._codec_busy_wait_s = 2.0
    condition = adapter._easymagpie_chunk_ready
    wait_started = threading.Event()
    original_wait = condition.wait
    published = []

    def wait(timeout):
        wait_started.set()
        notified = original_wait(timeout)
        assert notified, "ready arrival must notify the waiter, not just reach its timeout"
        return notified

    def publish_chunk():
        if wait_started.wait(2):
            published.append(_poll_native_codec_chunk(adapter, request))

    def schedule(*args, **kwargs):
        assert adapter._easymagpie_chunk_lock.locked()
        return "scheduled"

    monkeypatch.setattr(condition, "wait", wait)
    monkeypatch.setattr(OmniGenerationScheduler, "schedule", schedule)
    publisher = threading.Thread(target=publish_chunk, daemon=True)
    publisher.start()
    try:
        assert scheduler.schedule() == "scheduled"
    finally:
        publisher.join(3)
    assert not publisher.is_alive()
    assert published == [True]
    assert request.request_id in adapter._finished_load_reqs
    if arrival == "terminal":
        assert request.request_id in adapter.upstream_exhausted_requests
        assert request.num_prompt_tokens == request.num_computed_tokens == 3
        assert "audio" not in request.additional_information["codes"]
    else:
        assert request.num_prompt_tokens == 1
        assert request.num_computed_tokens == 0
        assert "stalled" not in adapter._finished_load_reqs


def test_codec_busy_wait_timeout_keeps_requests_and_state(monkeypatch):
    scheduler = _policy_scheduler(monkeypatch, codec=True, extra={"codec_busy_coalesce_ms": 4})
    scheduler.running = [SimpleNamespace(request_id=rid, num_computed_tokens=8) for rid in ("a", "b")]
    adapter = scheduler.chunk_transfer_adapter
    adapter._finished_load_reqs = {"a"}
    waits = []

    def wait_for(predicate, timeout):
        assert not predicate()
        waits.append(timeout)
        return False

    monkeypatch.setattr(adapter._easymagpie_chunk_ready, "wait_for", wait_for)
    monkeypatch.setattr(OmniGenerationScheduler, "schedule", lambda *args, **kwargs: "scheduled")
    assert scheduler.schedule() == "scheduled"
    assert waits == [0.004]
    assert [request.num_computed_tokens for request in scheduler.running] == [8, 8]
    assert adapter._finished_load_reqs == {"a"}


@pytest.mark.parametrize("previous", [None, (True, False)])
@pytest.mark.parametrize("fail", [False, True])
def test_talker_sender_restores_nested_finish_context(monkeypatch, previous, fail):
    adapter = _policy_scheduler(monkeypatch).chunk_transfer_adapter
    adapter._easymagpie_send_finish = previous
    outer = {"is_finished": False, "is_segment_finished": True}
    inner = {"is_finished": True, "is_segment_finished": False}
    seen = []

    def send(self, task):
        seen.append(self._easymagpie_send_finish)
        if task is outer:
            try:
                return self._send_single_request(inner)
            finally:
                assert self._easymagpie_send_finish == (False, True)
        if fail:
            raise RuntimeError("send failed")
        return "sent"

    monkeypatch.setattr(OmniChunkTransferAdapter, "_send_single_request", send)
    if fail:
        with pytest.raises(RuntimeError, match="send failed"):
            adapter._send_single_request(outer)
    else:
        assert adapter._send_single_request(outer) == "sent"
    assert seen == [(False, True), (True, False)]
    assert adapter._easymagpie_send_finish is previous


def _policy_scheduler(monkeypatch, *, codec=False, extra=None, max_requests=4):
    cls = EasyMagpieCodecScheduler if codec else EasyMagpieARAsyncScheduler
    base = OmniGenerationScheduler if codec else OmniARAsyncScheduler
    monkeypatch.setattr(base, "__init__", lambda *args, **kwargs: None)
    scheduler = object.__new__(cls)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=int(codec),
            stage_connector_config={"extra": extra or {}},
            hf_config=SimpleNamespace(num_stacked_codebooks=2),
        )
    )
    scheduler.chunk_transfer_adapter = SimpleNamespace(_finished_load_reqs=set())
    scheduler.running = []
    scheduler.waiting = []
    scheduler.max_num_running_reqs = max_requests
    scheduler.__init__()
    return scheduler


def _codec_poll_state(payloads):
    """Use the actual upstream poll and queues with only connector I/O stubbed."""
    payloads = deque(payloads)
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
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
