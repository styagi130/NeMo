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
"""Exercise the pinned CPU scheduler and KV allocator with phase eligibility."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from vllm import SamplingParams
from vllm.config import DeviceConfig, VllmConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler

TRACE = {}


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def scheduler(monkeypatch, request, tmp_path):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = SimpleNamespace(
        is_encoder_decoder=False,
        max_model_len=256,
        is_diffusion=False,
        enable_return_routed_experts=False,
        stage_id=0,
        runner_type="pooling",
    )
    config.cache_config.num_gpu_blocks = 512
    config.cache_config.enable_prefix_caching = False
    config.scheduler_config.max_num_seqs = 8
    config.scheduler_config.max_num_batched_tokens = 128
    config.scheduler_config.max_num_scheduled_tokens = 128
    cache = KVCacheConfig(
        512,
        [],
        [
            KVCacheGroupSpec(
                ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
            )
        ],
    )
    instance = object.__new__(EasyMagpieARAsyncScheduler)
    Scheduler.__init__(
        instance, config, cache, None, 16, mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False)
    )
    instance.phase_admission_enabled = True
    instance.phase_admission_phase = "mixed"
    instance.phase_admission_force_decode = False
    instance.phase_admission_throttled = False
    instance.phase_admission_counts = Counter()
    instance.pp_size = 1
    instance._spec_token_placeholders = []
    instance.chunk_transfer_adapter = None
    instance.input_coordinator = None
    instance._omits_kv_transfer_cache = {}
    instance._new_prompt_len_snapshot = {}
    instance.kv_transfer_criteria = None
    trace_path = tmp_path / "trace" / "phase-trace.json"
    trace_path.parent.mkdir()

    # Isolate transport setup while exercising the real full Scheduler.schedule and async accounting.
    def transport_free_schedule(self, throttle_prefills=False):
        self._should_defer_waiting_admission()
        before = {
            key: {
                "computed": value.num_computed_tokens,
                "prompt": value.num_prompt_tokens,
                "placeholders": value.num_output_placeholders,
                "in_flight": value.num_in_flight_tokens,
            }
            for key, value in self.requests.items()
        }
        output = Scheduler.schedule(self, throttle_prefills)
        TRACE.setdefault(request.node.name, []).append(
            {
                "phase": self.phase_admission_phase,
                "scheduled_tokens": dict(output.num_scheduled_tokens),
                "before": before,
                "after": {
                    key: {
                        "computed": value.num_computed_tokens,
                        "placeholders": value.num_output_placeholders,
                        "in_flight": value.num_in_flight_tokens,
                    }
                    for key, value in self.requests.items()
                },
            }
        )
        trace_path.write_text(json.dumps(TRACE, indent=2) + "\n")
        return output

    monkeypatch.setattr(OmniARAsyncScheduler, "schedule", transport_free_schedule)
    return instance


def add_request(scheduler, request_id, prompt=8, skipped=False):
    request = Request(request_id, list(range(prompt)), SamplingParams(max_tokens=32), None)
    request.external_req_id = request_id
    scheduler.requests[request_id] = request
    queue = scheduler.skipped_waiting if skipped else scheduler.waiting
    queue.add_request(request)
    return request


def complete(scheduler, output):
    for request_id, tokens in output.num_scheduled_tokens.items():
        request = scheduler.requests[request_id]
        request.num_in_flight_tokens -= tokens
        if not request.is_prefill_chunk:
            request.append_output_token_ids(1)
            request.num_output_placeholders -= 1
        assert request.num_in_flight_tokens >= 0 and request.num_output_placeholders >= 0


def test_prefill_then_decode_excludes_both_waiting_queues(scheduler):
    first = add_request(scheduler, "first")
    initial = scheduler.schedule()
    assert initial.num_scheduled_tokens == {"first": 8}
    assert first.num_computed_tokens == 8 and first.num_output_placeholders == 1
    complete(scheduler, initial)
    add_request(scheduler, "second")
    add_request(scheduler, "skipped", skipped=True)
    decoded = scheduler.schedule()
    assert decoded.num_scheduled_tokens == {"first": 1}
    assert len(scheduler.waiting) == 1 and len(scheduler.skipped_waiting) == 1
    complete(scheduler, decoded)
    admitted = scheduler.schedule()
    assert admitted.num_scheduled_tokens == {"skipped": 8, "second": 8}
    assert first.num_computed_tokens == 9
    complete(scheduler, admitted)
    decoded = scheduler.schedule()
    assert decoded.num_scheduled_tokens == {"first": 1, "skipped": 1, "second": 1}
    assert scheduler.phase_admission_status()["counts"]["prefill_steps"] == 2


def test_disabled_preserves_mixed_batch(scheduler):
    scheduler.phase_admission_enabled = False
    first = scheduler.schedule()  # Empty behavior remains valid.
    add_request(scheduler, "first")
    first = scheduler.schedule()
    complete(scheduler, first)
    add_request(scheduler, "new")
    mixed = scheduler.schedule()
    assert mixed.num_scheduled_tokens == {"first": 1, "new": 8}
    assert scheduler.phase_admission_counts == {}


def test_async_inflight_counts_stay_on_requests(scheduler):
    request = add_request(scheduler, "first")
    output = scheduler.schedule()
    add_request(scheduler, "new")
    decoded = scheduler.schedule()
    assert decoded.num_scheduled_tokens == {"first": 1}
    assert request.num_computed_tokens == 9
    assert request.num_in_flight_tokens == 9 and request.num_output_placeholders == 2
    admitted = scheduler.schedule()
    assert admitted.num_scheduled_tokens == {"new": 8}
    assert request.num_computed_tokens == 9 and request.num_in_flight_tokens == 9
    complete(scheduler, output)
    complete(scheduler, decoded)
    assert request.num_in_flight_tokens == 0 and request.num_output_placeholders == 0


def test_partial_prefill_is_separate_and_decode_not_starved(scheduler):
    add_request(scheduler, "first", 2)
    initial = scheduler.schedule()
    complete(scheduler, initial)
    scheduler.max_num_scheduled_tokens = 4
    add_request(scheduler, "long", 10)
    trace = []
    for _ in range(9):
        output = scheduler.schedule()
        trace.append((scheduler.phase_admission_phase, dict(output.num_scheduled_tokens)))
        complete(scheduler, output)
    prefill = [counts for phase, counts in trace if phase == "prefill"]
    assert prefill == [{"long": 4}, {"long": 4}, {"long": 2}]
    assert all("first" not in counts for counts in prefill)
    assert all(trace[i + 1][0] == "decode" for i, (phase, _) in enumerate(trace[:-1]) if phase == "prefill")
    assert scheduler.requests["long"].num_computed_tokens >= 10


def test_capacity_and_paused_stream_slots_preserved(scheduler):
    scheduler.max_num_running_reqs = 2
    add_request(scheduler, "first")
    complete(scheduler, scheduler.schedule())
    scheduler.num_waiting_for_streaming_input = 1
    add_request(scheduler, "waiting")
    for _ in range(3):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"first": 1}
        complete(scheduler, output)
    assert len(scheduler.waiting) == 1
    scheduler.num_waiting_for_streaming_input = 0
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"waiting": 8}


def test_allocation_failure_cannot_starve_decode(scheduler, monkeypatch):
    add_request(scheduler, "first")
    complete(scheduler, scheduler.schedule())
    add_request(scheduler, "no-space")
    original = scheduler.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        lambda request, *args, **kwargs: (
            None if request.request_id == "no-space" else original(request, *args, **kwargs)
        ),
    )
    trace = []
    for _ in range(6):
        output = scheduler.schedule()
        trace.append((scheduler.phase_admission_phase, dict(output.num_scheduled_tokens)))
        complete(scheduler, output)
    assert trace == [("decode", {"first": 1}), ("prefill", {})] * 3
    assert scheduler.waiting.peek_request().request_id == "no-space"


def test_unsupported_throttle_delegates_mixed_behavior(scheduler):
    add_request(scheduler, "first")
    complete(scheduler, scheduler.schedule())
    add_request(scheduler, "next")
    output = scheduler.schedule(throttle_prefills=True)
    assert scheduler.phase_admission_phase == "mixed"
    assert output.num_scheduled_tokens == {"first": 1}
    assert scheduler.phase_admission_counts["unsupported_fallback_steps"] == 1


def test_decode_cadence_does_not_admit_on_wrong_step(scheduler):
    request = add_request(scheduler, "first")
    complete(scheduler, scheduler.schedule())
    request.next_decode_eligible_step = scheduler.current_step + 3
    add_request(scheduler, "next")
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"next": 8}
    assert request.num_computed_tokens == 8


def test_cancel_frees_actual_allocator_and_keeps_waiting_order(scheduler):
    first = add_request(scheduler, "first")
    complete(scheduler, scheduler.schedule())
    add_request(scheduler, "second")
    add_request(scheduler, "third")
    Scheduler.finish_requests(scheduler, "first", RequestStatus.FINISHED_ABORTED)
    assert first.status == RequestStatus.FINISHED_ABORTED
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == ["second", "third"]
    assert "first" in output.finished_req_ids


def test_live_capacity128_has_no_hidden_clamp(scheduler):
    scheduler.max_num_running_reqs = 128
    scheduler.max_num_scheduled_tokens = 256
    for index in range(129):
        add_request(scheduler, f"request-{index}", 1)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 128
    assert len(scheduler.running) == 128 and len(scheduler.waiting) == 1
    complete(scheduler, output)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 128
    assert all(count == 1 for count in output.num_scheduled_tokens.values())
    assert len(scheduler.running) == 128 and len(scheduler.waiting) == 1


def test_resumed_stream_is_admitted_without_resetting_old_computed_prefix(scheduler):
    request = add_request(scheduler, "resumed")
    output = scheduler.schedule()
    complete(scheduler, output)
    scheduler.running.remove(request)
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.num_waiting_for_streaming_input = 1
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[4, 5, 6],
        max_tokens=7,
        arrival_time=2.0,
        sampling_params=SamplingParams(max_tokens=7),
    )
    scheduler._update_request_as_session(request, update)
    scheduler.waiting.add_request(request)
    assert scheduler.num_waiting_for_streaming_input == 0
    assert request.num_computed_tokens == 8 and request.max_tokens == 7
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"resumed": 3}
    assert request.num_computed_tokens == 11


def test_optin_rejects_missing_runtime_hooks(monkeypatch):
    monkeypatch.setenv("EASYMAGPIE_PHASE_ADMISSION", "1")
    monkeypatch.setattr(OmniARAsyncScheduler, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.delattr(Scheduler, "_easymagpie_phase_hooks_version", raising=False)
    with pytest.raises(RuntimeError, match="version-1 scheduler eligibility hooks"):
        EasyMagpieARAsyncScheduler()
