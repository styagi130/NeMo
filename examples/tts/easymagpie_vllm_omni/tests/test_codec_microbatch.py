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
"""Regression coverage for codec-window microbatching."""
from __future__ import annotations

from collections import deque
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from easymagpie_vllm_omni.codec_microbatch import (
    CodecMicroBatcher,
    _drain_dynamic_codec_buffers,
    _prepare_codec_cohort_for_schedule,
    _wait_for_codec_cohort,
    group_pending_tasks,
    install_scheduler_microbatch_patch,
    install_transfer_microbatch_patch,
    merge_codec_windows,
)


def test_codec_microbatcher_coalesces_simultaneous_windows():
    """Concurrent codec workers must make one explicit batched decode call."""
    calls: list[np.ndarray] = []

    def decode_batch(codes: np.ndarray) -> np.ndarray:
        calls.append(codes.copy())
        # One output row per input row; retain each caller's distinguishing value.
        return codes[:, :, 0].astype(np.float32)

    batcher = CodecMicroBatcher(decode_batch, max_batch_size=4, max_wait_s=0.05)
    barrier = threading.Barrier(4)
    results: list[np.ndarray | None] = [None] * 4

    def submit(index: int) -> None:
        codes = np.full((3, 2), index, dtype=np.int64)
        barrier.wait()
        results[index] = batcher.decode(codes)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    batcher.close()
    assert len(calls) == 1
    assert calls[0].shape == (4, 3, 2)
    assert sorted(int(row[0, 0]) for row in calls[0]) == [0, 1, 2, 3]
    assert [int(result[0]) for result in results if result is not None] == [0, 1, 2, 3]


def test_codec_microbatcher_defers_different_window_shapes():
    """A mismatched window must not block later compatible windows."""
    calls: list[tuple[int, ...]] = []

    def decode_batch(codes: np.ndarray) -> np.ndarray:
        calls.append(codes.shape)
        return codes[:, :, 0].astype(np.float32)

    batcher = CodecMicroBatcher(decode_batch, max_batch_size=2, max_wait_s=0.01)
    first = batcher.decode(np.full((2, 2), 1, dtype=np.int64))
    second = batcher.decode(np.full((3, 2), 2, dtype=np.int64))
    batcher.close()

    assert first.tolist() == [1.0, 1.0]
    assert second.tolist() == [2.0, 2.0, 2.0]
    assert calls == [(1, 2, 2), (1, 3, 2)]


def test_group_pending_tasks_preserves_each_streams_chunk_order():
    """Parallel transfer dispatch may not reorder codec chunks from one stream."""
    def task(stream: str, chunk: int):
        return {"request": SimpleNamespace(external_req_id=stream), "chunk": chunk}

    groups = group_pending_tasks([task("a", 0), task("b", 0), task("a", 1), task("b", 1), task("a", 2)])

    assert list(groups) == ["a", "b"]
    assert [item["chunk"] for item in groups["a"]] == [0, 1, 2]
    assert [item["chunk"] for item in groups["b"]] == [0, 1]


def test_merge_codec_windows_keeps_codebook_major_order_and_one_overlap_trim():
    """A merged window must append only new frames for every codebook."""
    # current: F=4, Q=2; following repeats one frame then adds three.
    current = [0, 1, 2, 3, 10, 11, 12, 13]
    following = [3, 4, 5, 6, 13, 14, 15, 16]

    merged = merge_codec_windows(
        current,
        following,
        num_quantizers=2,
        next_left_context_frames=1,
        max_frames=8,
    )

    assert merged == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]


def test_merge_codec_windows_does_not_exceed_exported_codec_capacity():
    """The scheduler must leave a successor for the next fixed-shape decode."""
    current = [0, 1, 2, 3, 10, 11, 12, 13]
    following = [3, 4, 5, 6, 13, 14, 15, 16]

    assert (
        merge_codec_windows(
            current,
            following,
            num_quantizers=2,
            next_left_context_frames=1,
            max_frames=6,
        )
        is None
    )


def test_nonblocking_codec_cohort_holds_only_until_deadline_then_releases():
    """Underfilled first and post-first cohorts never call scheduler sleep."""
    adapter = SimpleNamespace(_finished_load_reqs={"a"})
    scheduler = SimpleNamespace(
        chunk_transfer_adapter=adapter,
        requests={"a": object(), "b": object()},
        max_num_running_reqs=2,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            model_config=SimpleNamespace(
                stage_id=1,
                model_arch="EasyMagpieCode2Wav",
                stage_connector_config={
                    "extra": {
                        "codec_microbatch_wait_us": 1_000,
                        "codec_dynamic_chunking": True,
                        "codec_dynamic_chunk_wait_us": 100_000,
                        "codec_dynamic_wait_only_underfilled": True,
                    }
                },
            ),
        ),
    )

    with patch("easymagpie_vllm_omni.codec_microbatch.time.monotonic", return_value=10.0):
        assert _prepare_codec_cohort_for_schedule(scheduler) == "held"
    assert adapter._finished_load_reqs == set()

    with patch("easymagpie_vllm_omni.codec_microbatch.time.monotonic", return_value=10.002):
        assert _prepare_codec_cohort_for_schedule(scheduler) == "initial"
    assert adapter._finished_load_reqs == {"a"}

    scheduler._easymagpie_codec_initial_ids = {"a"}
    adapter._finished_load_reqs = {"a"}
    with patch("easymagpie_vllm_omni.codec_microbatch.time.monotonic", return_value=20.0):
        assert _prepare_codec_cohort_for_schedule(scheduler) == "held"
    assert adapter._finished_load_reqs == set()

    with patch("easymagpie_vllm_omni.codec_microbatch.time.monotonic", return_value=20.101):
        assert _prepare_codec_cohort_for_schedule(scheduler) == "dynamic"
    assert adapter._finished_load_reqs == {"a"}


def test_dynamic_codec_buffer_drain_skips_first_packet_then_merges_ready_successors():
    """The first frame is immediate; later packets drain all available windows."""
    request = SimpleNamespace(
        request_id="request-0",
        # F=1, Q=2, codebook-major.
        prompt_token_ids=[0, 10],
        additional_information={"meta": {"left_context_size": 0}},
    )
    payloads = deque(
        [
            # F=4, left=1 -> add three real frames.
            ([0, 1, 2, 3, 10, 11, 12, 13], 1),
            # F=7, left=4 -> add three more real frames.
            ([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16], 4),
        ]
    )

    def poll(received):
        if not payloads:
            return False
        ids, left = payloads.popleft()
        received.prompt_token_ids = ids
        received.additional_information = {"meta": {"left_context_size": left}}
        return True

    adapter = SimpleNamespace(
        _finished_load_reqs={"request-0"},
        _poll_single_request=poll,
        is_done_receiving_chunks=lambda _request_id: False,
    )
    scheduler = SimpleNamespace(
        chunk_transfer_adapter=adapter,
        requests={"request-0": request},
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=1,
                model_arch="EasyMagpieCode2Wav",
                hf_config=SimpleNamespace(num_audio_codebooks=1, frame_stacking_factor=2),
                stage_connector_config={
                    "extra": {
                        "codec_dynamic_chunking": True,
                        "codec_dynamic_chunk_wait_us": 0,
                        "codec_fixed_chunk_frames": 8,
                        "codec_chunk_frames": 3,
                    }
                },
            )
        ),
    )

    # Initial packet is intentionally untouched to preserve TTFA.
    assert _drain_dynamic_codec_buffers(scheduler) == 0
    assert request.prompt_token_ids == [0, 10]

    # The next scheduling pass drains both source windows into one F=7 packet.
    assert _drain_dynamic_codec_buffers(scheduler) == 2
    assert request.prompt_token_ids == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
    assert request.additional_information["meta"]["left_context_size"] == 0


def test_dynamic_codec_buffer_drain_preserves_tensor_payload_placeholder():
    """Metadata-carried codec ids must not be replaced by the scheduler's [0]."""
    request = SimpleNamespace(
        request_id="request-0",
        prompt_token_ids=[0],
        additional_information={"codes": {"audio": [[0, 10]]}, "meta": {"left_context_size": 0}},
    )
    payloads = deque(
        [
            ([0, 1, 2, 3, 10, 11, 12, 13], 1),
            ([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16], 4),
        ]
    )

    def poll(received):
        if not payloads:
            return False
        ids, left = payloads.popleft()
        received.prompt_token_ids = [0]
        received.additional_information = {"codes": {"audio": [ids]}, "meta": {"left_context_size": left}}
        return True

    adapter = SimpleNamespace(
        _finished_load_reqs={"request-0"},
        _poll_single_request=poll,
        is_done_receiving_chunks=lambda _request_id: False,
    )
    scheduler = SimpleNamespace(
        chunk_transfer_adapter=adapter,
        requests={"request-0": request},
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=1,
                model_arch="EasyMagpieCode2Wav",
                hf_config=SimpleNamespace(num_audio_codebooks=1, frame_stacking_factor=2),
                stage_connector_config={
                    "extra": {
                        "codec_dynamic_chunking": True,
                        "codec_dynamic_chunk_wait_us": 0,
                        "codec_fixed_chunk_frames": 8,
                        "codec_chunk_frames": 3,
                    }
                },
            )
        ),
    )

    assert _drain_dynamic_codec_buffers(scheduler) == 0
    assert _drain_dynamic_codec_buffers(scheduler) == 2
    assert request.prompt_token_ids == [0]
    assert request.additional_information["codes"]["audio"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]


def test_dynamic_codec_buffer_drain_skips_contained_stale_window():
    """A delayed duplicate must not be mistaken for a non-contiguous successor."""
    request = SimpleNamespace(
        request_id="request-0",
        # F=7, Q=2; absolute source interval [0, 7).
        prompt_token_ids=[0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16],
        additional_information={"generated_len": 7, "meta": {"left_context_size": 0}},
    )
    payloads = deque(
        [
            # A stale slice [3, 7) is already represented by ``request``.
            ([3, 4, 5, 6, 13, 14, 15, 16], 7, 1),
            # The real successor covers [3, 10), so it overlaps four frames.
            ([3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19], 10, 4),
        ]
    )

    def poll(received):
        if not payloads:
            return False
        ids, end, left = payloads.popleft()
        received.prompt_token_ids = ids
        received.additional_information = {"generated_len": end, "meta": {"left_context_size": left}}
        return True

    adapter = SimpleNamespace(
        _finished_load_reqs={"request-0"},
        _poll_single_request=poll,
        is_done_receiving_chunks=lambda _request_id: False,
    )
    scheduler = SimpleNamespace(
        chunk_transfer_adapter=adapter,
        requests={"request-0": request},
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=1,
                model_arch="EasyMagpieCode2Wav",
                hf_config=SimpleNamespace(num_audio_codebooks=1, frame_stacking_factor=2),
                stage_connector_config={
                    "extra": {
                        "codec_dynamic_chunking": True,
                        "codec_dynamic_chunk_wait_us": 0,
                        "codec_fixed_chunk_frames": 10,
                        "codec_chunk_frames": 3,
                    }
                },
            )
        ),
    )

    assert _drain_dynamic_codec_buffers(scheduler) == 0
    assert _drain_dynamic_codec_buffers(scheduler) == 1
    assert request.prompt_token_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert request.additional_information["generated_len"] == 10


def test_transfer_microbatch_dispatches_different_streams_in_parallel():
    """Stage-0 transfer must retain intra-stream order without serializing peers."""
    pytest.importorskip("vllm_omni")
    from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase

    assert install_transfer_microbatch_patch()
    sent: list[tuple[str, int]] = []
    sent_lock = threading.Lock()
    completed = threading.Event()
    active = 0
    max_active = 0

    def send(task):
        nonlocal active, max_active
        request = task["request"]
        with sent_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with sent_lock:
            sent.append((request.external_req_id, task["chunk"]))
            active -= 1
            if len(sent) == 3:
                completed.set()

    connector = SimpleNamespace(stage_id=0, close=lambda: None)
    adapter = SimpleNamespace(
        connector=connector,
        config=SimpleNamespace(
            stage_connector_config={"extra": {"codec_microbatch_wait_us": 1_000, "codec_microbatch_parallelism": 2}}
        ),
        scheduler_max_num_seqs=2,
        stop_event=threading.Event(),
        _save_cond=threading.Condition(),
        _recv_cond=threading.Condition(),
        _pending_save_reqs=deque(
            [
                {"request": SimpleNamespace(external_req_id="a"), "chunk": 0},
                {"request": SimpleNamespace(external_req_id="b"), "chunk": 0},
                {"request": SimpleNamespace(external_req_id="a"), "chunk": 1},
            ]
        ),
        _send_single_request=send,
    )
    worker = threading.Thread(target=OmniTransferAdapterBase.save_loop, args=(adapter,))
    worker.start()
    assert completed.wait(timeout=2)

    OmniTransferAdapterBase.shutdown(adapter)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert [chunk for stream, chunk in sent if stream == "a"] == [0, 1]
    assert max_active == 2


def test_codec_scheduler_waits_once_for_a_partial_ready_cohort():
    """Stage 1 must delay only when a real codec cohort is still incomplete."""
    scheduler = SimpleNamespace(
        chunk_transfer_adapter=SimpleNamespace(_finished_load_reqs={"request-0"}),
        max_num_running_reqs=8,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=1,
                model_arch="EasyMagpieCode2Wav",
                stage_connector_config={"extra": {"codec_microbatch_wait_us": 1_500}},
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
        ),
    )

    with patch("easymagpie_vllm_omni.codec_microbatch.time.sleep") as sleep:
        assert _wait_for_codec_cohort(scheduler)
    sleep.assert_called_once()
    # A tiny floating-point rounding increment is possible when the deadline
    # is calculated from two consecutive monotonic-clock readings.
    assert 0 < sleep.call_args.args[0] <= 0.0016
    assert scheduler._easymagpie_codec_cohort_deadline == 0.0


def test_scheduler_microbatch_patch_forwards_vllm_schedule_arguments():
    """The scheduler wrapper must support vLLM's optional throttle argument."""
    pytest.importorskip("vllm_omni")
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

    original_schedule = OmniGenerationScheduler.schedule
    had_patch_flag = hasattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched")
    previous_patch_flag = getattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched", None)
    calls = []

    def fake_schedule(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return "scheduled"

    try:
        OmniGenerationScheduler.schedule = fake_schedule
        if had_patch_flag:
            delattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched")
        assert install_scheduler_microbatch_patch()
        scheduler = SimpleNamespace()
        assert OmniGenerationScheduler.schedule(scheduler, False, throttle=True) == "scheduled"
        assert calls == [(scheduler, (False,), {"throttle": True})]
    finally:
        OmniGenerationScheduler.schedule = original_schedule
        if had_patch_flag:
            OmniGenerationScheduler._easymagpie_codec_microbatch_patched = previous_patch_flag
        elif hasattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched"):
            delattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched")
