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
from __future__ import annotations

import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from easymagpie_vllm_omni.stage_processors import talker2code2wav_async_chunk
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


class _Request:
    external_req_id = "request-0"

    def __init__(self):
        self.finished = False
        self.resumable = True
        self.output_token_ids = []

    def is_finished(self):
        return self.finished


def _manager():
    return SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=2)),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                }
            }
        ),
        code_prompt_token_ids=defaultdict(list),
    )


def _output(value: int):
    return {"audio_codes": torch.tensor([[value, value + 100]], dtype=torch.long)}


def test_async_codec_state_stays_continuous_across_resumable_segments():
    manager = _manager()
    request = _Request()

    # Warm-up is counted over the whole request, including segment boundaries.
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    warmup_flush = talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True)
    assert warmup_flush.codes.audio.numel() == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    # The framework buffer is reset per segment, but the processor's request
    # state retains the real acoustic frames and its emission high-water mark.
    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    first = talker2code2wav_async_chunk(manager, _output(4), request, is_finished=True)
    torch.testing.assert_close(first.codes.audio, torch.tensor([[3, 103], [4, 104]]))
    assert first.meta.left_context_size == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(5), request) is None
    request.output_token_ids = [0, 0]
    second = talker2code2wav_async_chunk(manager, _output(6), request)
    torch.testing.assert_close(second.codes.audio, torch.tensor([[5, 105], [6, 106]]))
    assert second.meta.left_context_size == 0

    # Repeated segment flushes at the same length must not duplicate audio.
    request.finished = True
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None

    # Terminal completion releases the request-persistent state.
    request.resumable = False
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None
    assert request.external_req_id not in manager._emp_seen_frames
    assert request.external_req_id not in manager._emp_request_speech_delay
    assert request.external_req_id not in manager._emp_emitted_frames
    assert request.external_req_id not in manager._emp_emitted_chunks
    assert request.external_req_id not in manager._emp_frame_buffer_base
    assert request.external_req_id not in manager._emp_frame_buffer


def test_resumable_segment_stop_does_not_flush_partial_codec_chunk():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4})
    request = _Request()

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.finished = True
    assert talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True) is None

    request.finished = False
    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    full = talker2code2wav_async_chunk(manager, _output(4), request)
    assert full is not None
    torch.testing.assert_close(full.codes.audio, torch.tensor([[1, 101], [2, 102], [3, 103], [4, 104]]))


def test_async_codec_drops_only_warmup_not_moved_into_prefill():
    manager = _manager()
    manager.config.hf_config = SimpleNamespace(
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    manager.connector.config["extra"]["codec_chunk_frames"] = 1
    request = _Request()
    request.additional_information = {"text_prefill_num": 4}

    # The prefill callback carries no generated acoustic frame and must not
    # consume the one remaining warm-up slot.
    assert (
        talker2code2wav_async_chunk(
            manager,
            {"audio_codes": torch.zeros(1, 2, dtype=torch.long)},
            request,
        )
        is None
    )
    assert manager._emp_seen_frames[request.external_req_id] == 0

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    first_audio = talker2code2wav_async_chunk(manager, _output(2), request)
    torch.testing.assert_close(first_audio.codes.audio, torch.tensor([[2, 102]]))


def test_async_codec_forwards_terminal_audio_eos_row():
    manager = _manager()
    manager.config.hf_config = SimpleNamespace(
        streaming_speech_delay=0,
        forced_audio_eos_id=1025,
        codebook_size=1024,
    )
    manager.connector.config["extra"]["codec_chunk_frames"] = 4
    request = _Request()
    request.resumable = False

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    assert talker2code2wav_async_chunk(manager, _output(2), request) is None

    request.finished = True
    terminal = talker2code2wav_async_chunk(
        manager,
        {"audio_codes": torch.tensor([[1025, 777]], dtype=torch.long)},
        request,
        is_finished=True,
    )

    assert bool(terminal.meta.finished)
    torch.testing.assert_close(
        terminal.codes.audio,
        torch.tensor([[1, 101], [2, 102], [1025, 777], [-1, -1]]),
    )


def test_async_codec_pads_short_terminal_chunk_to_steady_size():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"]["codec_chunk_frames"] = 4
    request = _Request()
    request.resumable = False

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.finished = True
    terminal = talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True)

    torch.testing.assert_close(
        terminal.codes.audio,
        torch.tensor([[1, 101], [2, 102], [-1, -1], [-1, -1]]),
    )


def test_async_codec_buffer_drops_emitted_rows():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    request = _Request()

    last = None
    for value in range(1, 11):
        last = talker2code2wav_async_chunk(manager, _output(value), request)

    assert last is not None
    torch.testing.assert_close(last.codes.audio, torch.tensor([[9, 109], [10, 110]]))
    assert last.meta.left_context_size == 0
    buffer = manager._emp_frame_buffer[request.external_req_id]
    assert buffer == []
    assert manager._emp_frame_buffer_base[request.external_req_id] == 10
    assert manager._emp_emitted_frames[request.external_req_id] == 10


def test_async_codec_uses_configured_startup_chunk_ramp():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [1, 2]})
    request = _Request()

    first = talker2code2wav_async_chunk(manager, _output(1), request)
    assert first is not None
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101]]))

    assert talker2code2wav_async_chunk(manager, _output(2), request) is None
    second = talker2code2wav_async_chunk(manager, _output(3), request)
    assert second is not None
    torch.testing.assert_close(second.codes.audio, torch.tensor([[2, 102], [3, 103]]))
    assert second.meta.left_context_size == 0

    for value in range(4, 7):
        assert talker2code2wav_async_chunk(manager, _output(value), request) is None
    steady = talker2code2wav_async_chunk(manager, _output(7), request)
    assert steady is not None
    torch.testing.assert_close(
        steady.codes.audio,
        torch.tensor([[4, 104], [5, 105], [6, 106], [7, 107]]),
    )
    assert steady.meta.left_context_size == 0
    assert manager._emp_emitted_chunks[request.external_req_id] == 3


def test_async_codec_allows_larger_first_chunk_than_steady_state():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [5]})
    request = _Request()

    for value in range(1, 5):
        assert talker2code2wav_async_chunk(manager, _output(value), request) is None
    first = talker2code2wav_async_chunk(manager, _output(5), request)
    assert first is not None
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101], [2, 102], [3, 103], [4, 104], [5, 105]]))


def test_async_codec_rejects_invalid_startup_chunk_ramp():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 4, "codec_startup_chunk_frames": [1, 0]})
    with pytest.raises(ValueError, match="codec_startup_chunk_frames"):
        talker2code2wav_async_chunk(manager, _output(1), _Request())


def test_stateful_codec_emits_only_new_time_major_rows():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update({"codec_chunk_frames": 2})
    request = _Request()

    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    first = talker2code2wav_async_chunk(manager, _output(2), request)
    assert first.codes.audio.shape == (2, 2)
    torch.testing.assert_close(first.codes.audio, torch.tensor([[1, 101], [2, 102]]))
    assert first.meta.left_context_size == 0
    assert manager._emp_frame_buffer[request.external_req_id] == []

    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    second = talker2code2wav_async_chunk(manager, _output(4), request)
    assert second.codes.audio.shape == (2, 2)
    torch.testing.assert_close(second.codes.audio, torch.tensor([[3, 103], [4, 104]]))
    assert manager._emp_frame_buffer_base[request.external_req_id] == 4


@pytest.mark.parametrize("busy_chunks", [0, "8", [0], [-1]])
def test_async_codec_rejects_invalid_busy_startup_ramp(busy_chunks):
    manager = _manager()
    manager.connector.config["extra"]["codec_busy_startup_chunk_frames"] = busy_chunks
    with pytest.raises(ValueError, match="codec_busy_startup_chunk_frames"):
        talker2code2wav_async_chunk(manager, _output(1), _Request())


def test_async_codec_busy_startup_ramp_is_frozen_across_segments_and_cleaned_up():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update(
        codec_chunk_frames=4, codec_startup_chunk_frames=[1, 2], codec_busy_startup_chunk_frames=[3, 1]
    )
    first, busy = _Request(), _Request()
    busy.external_req_id = "busy"
    assert talker2code2wav_async_chunk(manager, _output(1), first) is not None
    assert talker2code2wav_async_chunk(manager, _output(10), busy) is None

    first.finished, first.resumable = True, False
    assert talker2code2wav_async_chunk(manager, None, first, is_finished=True) is None
    assert first.external_req_id not in manager._emp_request_startup_chunks

    # Finishing the other request and a resumable segment must not change the profile.
    busy.finished = True
    assert talker2code2wav_async_chunk(manager, _output(11), busy, is_finished=True) is None
    assert manager._emp_request_startup_chunks[busy.external_req_id] == [3, 1]
    assert manager._emp_requests[busy.external_req_id] is busy
    busy.finished = False
    chunk = talker2code2wav_async_chunk(manager, _output(12), busy)
    torch.testing.assert_close(chunk.codes.audio, torch.tensor([[10, 110], [11, 111], [12, 112]]))
    chunk = talker2code2wav_async_chunk(manager, _output(13), busy)
    torch.testing.assert_close(chunk.codes.audio, torch.tensor([[13, 113]]))

    busy.finished, busy.resumable = True, False
    assert talker2code2wav_async_chunk(manager, None, busy, is_finished=True) is None
    assert manager._emp_request_startup_chunks == {}
    assert manager._emp_emitted_chunks == {}
    assert manager._emp_requests == {}
    chunk = talker2code2wav_async_chunk(manager, _output(20), _Request())
    torch.testing.assert_close(chunk.codes.audio, torch.tensor([[20, 120]]))


@pytest.mark.parametrize("other_emitted", [0, 1])
@pytest.mark.parametrize("configure_busy", [False, True])
def test_async_codec_busy_selection_requires_another_emitting_request(other_emitted, configure_busy):
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update(codec_chunk_frames=4, codec_startup_chunk_frames=[1, 2])
    if configure_busy:
        manager.connector.config["extra"]["codec_busy_startup_chunk_frames"] = [3]
    manager._emp_emitted_chunks = defaultdict(int, other=other_emitted)
    request = _Request()

    chunk = talker2code2wav_async_chunk(manager, _output(1), request)
    if configure_busy and other_emitted:
        assert chunk is None
        assert manager._emp_request_startup_chunks[request.external_req_id] == [3]
    else:
        torch.testing.assert_close(chunk.codes.audio, torch.tensor([[1, 101]]))
        # Its own first emission must not switch the request to the busy profile.
        assert talker2code2wav_async_chunk(manager, _output(2), request) is None
        chunk = talker2code2wav_async_chunk(manager, _output(3), request)
        torch.testing.assert_close(chunk.codes.audio, torch.tensor([[2, 102], [3, 103]]))


@pytest.mark.parametrize("speech_delay", [0, 2])
def test_async_codec_terminal_cleans_busy_profile_with_partial_or_no_audio(speech_delay):
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = speech_delay
    manager.connector.config["extra"].update(codec_chunk_frames=4, codec_busy_startup_chunk_frames=[3])
    request = _Request()
    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    assert request.external_req_id in manager._emp_request_startup_chunks

    request.finished, request.resumable = True, False
    terminal = talker2code2wav_async_chunk(manager, None, request, is_finished=True)
    assert terminal is not None
    assert manager._emp_request_startup_chunks == {}


@pytest.mark.parametrize("late_callback", [False, True])
@pytest.mark.parametrize("configure_busy", [False, True])
@pytest.mark.parametrize("status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR])
def test_async_codec_abort_cleans_sender_state_before_next_request(monkeypatch, late_callback, configure_busy, status):
    manager = _adapter_manager(monkeypatch)
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update(codec_chunk_frames=4, codec_startup_chunk_frames=[1, 2])
    if configure_busy:
        manager.connector.config["extra"]["codec_busy_startup_chunk_frames"] = [3]
    aborted = _Request()
    aborted.request_id = "internal-aborted"
    aborted.status = RequestStatus.RUNNING
    assert talker2code2wav_async_chunk(manager, _output(1), aborted) is not None
    assert manager._emp_emitted_chunks[aborted.external_req_id] == 1
    if late_callback:
        aborted.num_computed_tokens = 2
        manager.save_async(_output(2), aborted)
    manager.finish_requests(aborted.request_id, status, {aborted.request_id: aborted})
    aborted.status, aborted.finished = status, True

    # A save already queued before the abort must not recreate request state.
    if late_callback:
        manager.connector.put = lambda **kwargs: pytest.fail("aborted audio must not be sent")
        manager._send_single_request(manager._pending_save_reqs.popleft())
    current = _Request()
    current.request_id, current.external_req_id = "internal-current", "current"
    first = talker2code2wav_async_chunk(manager, _output(10), current)
    assert first is not None
    torch.testing.assert_close(first.codes.audio, torch.tensor([[10, 110]]))
    assert manager._emp_request_startup_chunks[current.external_req_id] == [1, 2]
    for name, state in vars(manager).items():
        if name.startswith("_emp_"):
            assert aborted.external_req_id not in state, name


@pytest.mark.parametrize("status", [RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED])
def test_async_codec_normal_receiver_cleanup_before_save_preserves_terminal_tail(monkeypatch, status):
    manager = _adapter_manager(monkeypatch)
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"]["codec_chunk_frames"] = 4
    request = _Request()
    request.request_id, request.num_computed_tokens = "internal-finished", 2
    request.status = RequestStatus.RUNNING
    assert talker2code2wav_async_chunk(manager, _output(1), request) is None

    # The real AR scheduler frees receiver state before enqueueing terminal audio.
    request.status, request.finished, request.resumable = status, True, False
    manager.cleanup_receiver(request.request_id)
    assert request.request_id in manager._cancelled_load_reqs
    manager.save_async(_output(2), request)
    sent = []

    def put(**kwargs):
        sent.append(kwargs["data"])
        return True, 1, {}

    manager.connector.put = put
    manager._send_single_request(manager._pending_save_reqs.popleft())
    assert len(sent) == 1
    torch.testing.assert_close(sent[0].codes.audio, torch.tensor([[1, 101], [2, 102], [-1, -1], [-1, -1]]))
    assert bool(sent[0].meta.finished)
    for name, state in vars(manager).items():
        if name.startswith("_emp_"):
            assert request.external_req_id not in state, name


def test_async_codec_defers_other_failed_request_cleanup_until_admission():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"].update(codec_startup_chunk_frames=[1], codec_busy_startup_chunk_frames=[3])
    aborted, active, new = _Request(), _Request(), _Request()
    active.external_req_id, new.external_req_id = "active", "new"
    assert talker2code2wav_async_chunk(manager, _output(1), aborted) is not None
    assert talker2code2wav_async_chunk(manager, _output(2), active) is None
    aborted.status, aborted.finished = RequestStatus.FINISHED_ABORTED, True

    assert talker2code2wav_async_chunk(manager, _output(3), active) is None
    assert manager._emp_requests[aborted.external_req_id] is aborted
    assert talker2code2wav_async_chunk(manager, _output(4), new) is not None
    assert aborted.external_req_id not in manager._emp_requests
    assert manager._emp_request_startup_chunks[active.external_req_id] == [3]
    assert manager._emp_request_startup_chunks[new.external_req_id] == [1]

    # A late callback for an old object must not erase a replacement with its ID.
    replacement = _Request()
    assert talker2code2wav_async_chunk(manager, _output(5), replacement) is None
    assert talker2code2wav_async_chunk(manager, _output(6), aborted) is None
    assert manager._emp_requests[replacement.external_req_id] is replacement
    assert manager._emp_frame_buffer[replacement.external_req_id][0].tolist() == [5, 105]


@pytest.mark.parametrize("terminal_has_frame", [False, True])
@pytest.mark.parametrize(
    ("mode", "delay"),
    [
        ("http_ordered", 0),
        ("http_all_backlog", 0),
        ("http_tail_backlog", 0),
        ("http_ordered", 2),
        ("http_all_backlog", 2),
        ("http_tail_backlog", 2),
        ("stream_ordered", 2),
        ("stream_segment_backlog", 2),
        ("stream_terminal_backlog", 2),
    ],
)
def test_async_codec_queued_finish_uses_captured_task_state(monkeypatch, mode, delay, terminal_has_frame):
    reference_mode = "stream_ordered" if mode.startswith("stream") else "http_ordered"
    expected = _drain_codec_frames(monkeypatch, reference_mode, delay, terminal_has_frame)
    actual = _drain_codec_frames(monkeypatch, mode, delay, terminal_has_frame)
    assert actual == expected
    assert sum(payload["finished"] for payload in actual) == 1
    assert [row[0] for payload in actual for row in payload["codes"] if row[0] >= 0] == list(range(delay + 1, 9))


def test_aborted_callback_cannot_erase_reused_request_id():
    manager = _manager()
    manager.config.hf_config.streaming_speech_delay = 0
    aborted = _Request()
    assert talker2code2wav_async_chunk(manager, _output(1), aborted) is None
    aborted.status = RequestStatus.FINISHED_ABORTED

    replacement = _Request()
    assert talker2code2wav_async_chunk(manager, _output(10), replacement) is None
    assert talker2code2wav_async_chunk(manager, _output(2), aborted) is None
    assert manager._emp_requests[replacement.external_req_id] is replacement
    output = talker2code2wav_async_chunk(manager, _output(11), replacement)
    torch.testing.assert_close(output.codes.audio, torch.tensor([[10, 110], [11, 111]]))


@pytest.mark.parametrize("previous", [None, (True, False)])
@pytest.mark.parametrize("fail", [False, True])
def test_talker_sender_restores_nested_finish_context(monkeypatch, previous, fail):
    adapter = _adapter_manager(monkeypatch)
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


@pytest.mark.parametrize("stage_id", [0, 1])
@pytest.mark.parametrize("has_adapter", [False, True])
def test_captured_finish_hook_only_installs_on_stage_zero(monkeypatch, stage_id, has_adapter):
    initialized = []
    monkeypatch.setattr(OmniARAsyncScheduler, "__init__", lambda self: initialized.append(self))
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage_id))
    scheduler.max_num_running_reqs = 64
    adapter = SimpleNamespace() if has_adapter else None
    scheduler.chunk_transfer_adapter = adapter

    scheduler.__init__()

    assert initialized == [scheduler]
    assert hasattr(adapter, "_send_single_request") is (stage_id == 0 and has_adapter)


@pytest.mark.parametrize("frames", [0, 2, 6])
@pytest.mark.parametrize("backlog", [False, True])
def test_late_stream_final_frees_request_and_delivers_codec_tail(monkeypatch, frames, backlog):
    manager, request, scheduler, sent = _waiting_stream(monkeypatch, frames, backlog)
    before_tokens = list(request.output_token_ids)
    before_computed = request.num_computed_tokens
    final = Request(request.request_id, [0], SamplingParams(max_tokens=1), None, resumable=False)

    scheduler.add_request(final)

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert not request.resumable
    _assert_stream_freed(scheduler, request)
    assert request.num_computed_tokens == before_computed
    assert list(request.output_token_ids) == before_tokens
    assert manager._pending_save_reqs[-1]["is_finished"] is True
    assert manager._pending_save_reqs[-1]["is_segment_finished"] is False
    while manager._pending_save_reqs:
        manager._send_single_request(manager._pending_save_reqs.popleft())

    assert sum(bool(payload.meta.finished) for payload in sent) == 1
    rows = [row for payload in sent if payload.codes is not None for row in payload.codes.audio.tolist()]
    expected = [[frame, frame + 100] for frame in range(1, frames + 1)]
    assert rows == expected + [[-1, -1]] * ((-frames) % 4)
    for name, state in vars(manager).items():
        if name.startswith("_emp_"):
            assert request.external_req_id not in state, name
    assert request.external_req_id not in manager.requests_num_chunks_sent
    assert request.external_req_id not in manager.put_req_chunk


@pytest.mark.parametrize("backlog", [False, True])
def test_waiting_stream_abort_still_discards_pending_audio(monkeypatch, backlog):
    manager, request, scheduler, sent = _waiting_stream(monkeypatch, 2, backlog)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    _assert_stream_freed(scheduler, request)
    assert request.status == RequestStatus.FINISHED_ABORTED
    while manager._pending_save_reqs:
        manager._send_single_request(manager._pending_save_reqs.popleft())
    assert not sent
    # Without a pending callback, upstream abort leaves producer-owned state
    # until the next callback or new request. Keep that existing lifetime explicit.
    if not backlog:
        assert manager._emp_requests[request.external_req_id] is request
    next_request = Request("next", [1], SamplingParams(max_tokens=4), None, resumable=True)
    next_request.external_req_id = "next-external"
    assert talker2code2wav_async_chunk(manager, _output(10), next_request) is None
    for name, state in vars(manager).items():
        if name.startswith("_emp_"):
            assert request.external_req_id not in state, name


def test_late_final_hook_does_not_change_other_stages(monkeypatch):
    manager, request, scheduler, _ = _waiting_stream(monkeypatch, 0, False)
    scheduler.vllm_config.model_config.stage_id = 1
    final = Request(request.request_id, [0], SamplingParams(max_tokens=1), None, resumable=False)

    scheduler.add_request(final)

    assert request.status == RequestStatus.FINISHED_ABORTED
    _assert_stream_freed(scheduler, request)
    assert not manager._pending_save_reqs


@pytest.mark.parametrize("case", ["active", "new", "update", "nonresumable"])
def test_late_final_hook_delegates_other_inputs(monkeypatch, case):
    _, request, scheduler, _ = _waiting_stream(monkeypatch, 0, False)
    if case == "active":
        request.status = RequestStatus.RUNNING
    elif case == "nonresumable":
        request.resumable = False
    incoming = Request(
        "new" if case == "new" else request.request_id,
        [0],
        SamplingParams(max_tokens=1),
        None,
        resumable=case == "update",
    )
    original = Mock()
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", original)

    scheduler.add_request(incoming)

    original.assert_called_once_with(incoming)


def _waiting_stream(monkeypatch, frames, backlog):
    manager = _adapter_manager(monkeypatch)
    manager.config.hf_config.streaming_speech_delay = 0
    manager.connector.config["extra"]["codec_chunk_frames"] = 4
    request = Request("internal-final", [1], SamplingParams(max_tokens=64), None, resumable=True)
    request.external_req_id = "external-final"
    request.streaming_queue = deque()
    sent = []

    def put(**kwargs):
        sent.append(kwargs["data"])
        return True, 1, {}

    manager.connector.put = put
    for frame in range(1, frames + 1):
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids(0)
        request.num_computed_tokens = frame
        manager.save_async(_output(frame), request)
        if not backlog:
            manager._send_single_request(manager._pending_save_reqs.popleft())
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    scheduler.chunk_transfer_adapter = manager
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.skipped_waiting.add_request(request)
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.connector = None
    scheduler.defer_block_free = False
    scheduler.kv_cache_manager = SimpleNamespace(free=Mock())
    scheduler.encoder_cache_manager = SimpleNamespace(free=Mock())
    scheduler.input_coordinator = SimpleNamespace(free_finished_request=Mock())
    scheduler._omits_kv_transfer_cache = {}
    scheduler._new_prompt_len_snapshot = {request.request_id: 1}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    return manager, request, scheduler, sent


def _assert_stream_freed(scheduler, request):
    assert request.request_id not in scheduler.requests
    assert request.request_id in scheduler.finished_req_ids
    assert scheduler.num_waiting_for_streaming_input == 0
    assert not scheduler.running and not scheduler.waiting and not scheduler.skipped_waiting
    scheduler.kv_cache_manager.free.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    assert scheduler.input_coordinator.free_finished_request.called


def _drain_codec_frames(monkeypatch, mode, delay, terminal_has_frame):
    manager = _adapter_manager(monkeypatch)
    manager.config.hf_config.streaming_speech_delay = delay
    manager.connector.config["extra"].update(codec_chunk_frames=4, codec_startup_chunk_frames=[2, 2])
    streaming = mode.startswith("stream")
    request = Request("internal", [1], SamplingParams(max_tokens=64), None, resumable=streaming)
    request.external_req_id = "external"
    sent = []

    def put(**kwargs):
        payload = kwargs["data"]
        audio = payload.codes.audio if payload.codes is not None else None
        sent.append(
            {
                "codes": audio.tolist() if isinstance(audio, torch.Tensor) else [],
                "finished": bool(payload.meta.finished),
                "segment_finished": bool(payload.meta.is_segment_finished),
            }
        )
        return True, 1, {}

    manager.connector.put = put

    def drain():
        while manager._pending_save_reqs:
            manager._send_single_request(manager._pending_save_reqs.popleft())

    for frame in range(1, 9):
        request.status, request.num_computed_tokens = RequestStatus.RUNNING, frame
        segment, terminal = streaming and frame == 4, terminal_has_frame and frame == 8
        if segment or terminal:
            request.status = RequestStatus.FINISHED_STOPPED
        if terminal:
            request.resumable = False
            manager.cleanup_receiver(request.request_id)
        manager.save_async(_output(frame), request, is_segment_finished=segment)
        if mode.endswith("ordered") or (mode == "http_tail_backlog" and frame <= 3):
            drain()
        elif mode == "stream_segment_backlog" and (segment or frame > 4):
            drain()
    if not terminal_has_frame:
        request.status, request.resumable = RequestStatus.FINISHED_STOPPED, False
        manager.cleanup_receiver(request.request_id)
        manager.save_async(None, request)
    drain()
    assert manager._emp_requests == {}
    return sent


def _adapter_manager(monkeypatch=None):
    manager = object.__new__(OmniChunkTransferAdapter)
    vars(manager).update(vars(_manager()))
    manager.connector.stage_id = 0
    manager.custom_process_next_stage_input_func = talker2code2wav_async_chunk
    manager._save_cond = threading.Condition()
    for name in (
        "requests_origin_status",
        "_active_streams",
        "get_req_chunk",
        "request_ids_mapping",
        "put_req_chunk",
        "request_payload",
        "requests_num_chunks_sent",
        "ramp_chunk_count",
        "_pending_streaming_prefills",
    ):
        setattr(manager, name, defaultdict(int))
    for name in (
        "waiting_for_chunk_waiting_requests",
        "waiting_for_chunk_running_requests",
        "_held_non_active",
        "_pending_save_reqs",
    ):
        setattr(manager, name, deque())
    for name in (
        "requests_with_ready_chunks",
        "upstream_exhausted_requests",
        "segment_finished_requests",
        "_finished_load_reqs",
        "_cancelled_load_reqs",
    ):
        setattr(manager, name, set())
    if monkeypatch is not None:
        monkeypatch.setattr(OmniARAsyncScheduler, "__init__", lambda *args, **kwargs: None)
        scheduler = object.__new__(EasyMagpieARAsyncScheduler)
        scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
        scheduler.max_num_running_reqs = 64
        scheduler.chunk_transfer_adapter = manager
        scheduler.__init__()
    return manager
