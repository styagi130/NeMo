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
import ast
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.admission_fence import AdmissionDecodeFence
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from test_phase_admission import add_request, complete, no_cuda, scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


@pytest.fixture
def fenced(scheduler):
    scheduler.admission_decode_fence = AdmissionDecodeFence(scheduler)
    return scheduler


def arm(scheduler, name="cold"):
    owner = add_request(scheduler, name, 71)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {name: 71}
    assert scheduler.admission_decode_fence.remaining == 4
    complete(scheduler, output)
    return owner


def test_four_decodes_then_waiter_admitted(fenced):
    arm(fenced)
    add_request(fenced, "waiter", 71)
    for remaining in (3, 2, 1, 0):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {"cold": 1}
        assert fenced.admission_decode_fence.remaining == remaining
        complete(fenced, out)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"waiter": 71}
    counts = fenced.admission_decode_fence.counts
    assert counts["decode_steps"] == 4 and counts["completed_epochs"] == 1
    assert counts["max_epoch_decode_steps"] == 4
    assert counts["waiting_admission_deferrals"] == 4
    assert counts["extra_waiting_admission_deferrals"] == 3


def test_arrivals_cannot_extend_fence_or_reorder_waiters(fenced):
    arm(fenced)
    for i in range(4):
        add_request(fenced, f"waiter-{i}", 71)
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {"cold": 1}
        complete(fenced, out)
    out = fenced.schedule()
    assert list(out.num_scheduled_tokens)[0] == "waiter-0"
    assert fenced.admission_decode_fence.counts["completed_epochs"] == 1


def test_steady_running_rows_keep_decode_opportunities(fenced):
    add_request(fenced, "steady", 8)
    complete(fenced, fenced.schedule())
    complete(fenced, fenced.schedule())
    arm(fenced)
    add_request(fenced, "waiter", 8)
    for _ in range(4):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {"steady": 1, "cold": 1}
        complete(fenced, out)


def test_partial_cold71_does_not_arm(fenced):
    fenced.max_num_scheduled_tokens = 32
    add_request(fenced, "partial", 71)
    for expected in (32, 32, 7):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {"partial": expected}
        complete(fenced, out)
        assert fenced.admission_decode_fence.remaining == 0
    assert not fenced.admission_decode_fence.counts


@pytest.mark.parametrize("length", [1, 70, 72])
def test_other_prompt_lengths_keep_original_one_decode(fenced, length):
    add_request(fenced, "old", length)
    complete(fenced, fenced.schedule())
    add_request(fenced, "new", 8)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"old": 1}
    complete(fenced, out)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"new": 8}
    assert fenced.admission_decode_fence.remaining == 0


def test_cancel_releases_actual_allocator_and_fence(fenced):
    owner = arm(fenced)
    add_request(fenced, "waiter", 71)
    Scheduler.finish_requests(fenced, owner.request_id, RequestStatus.FINISHED_ABORTED)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"waiter": 71}
    assert fenced.admission_decode_fence.counts["released_no_eligible_owner"] == 1


def test_same_id_new_object_does_not_inherit_old_fence(fenced):
    owner = arm(fenced)
    Scheduler.finish_requests(fenced, "cold", RequestStatus.FINISHED_ABORTED)
    replacement = add_request(fenced, "cold", 8)
    assert replacement is not owner
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"cold": 8}
    assert fenced.admission_decode_fence.remaining == 0


def test_preemption_and_recompute_never_rearm(fenced):
    owner = arm(fenced)
    fenced.running.remove(owner)
    fenced._preempt_request(owner, 1.0)
    assert owner.num_preemptions == 1
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"cold": 72}
    assert fenced.admission_decode_fence.remaining == 0
    assert fenced.admission_decode_fence.counts["armed_epochs"] == 1


def test_no_eligible_owner_does_not_block_waiter(fenced):
    owner = arm(fenced)
    owner.next_decode_eligible_step = fenced.current_step + 20
    add_request(fenced, "waiter", 8)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"waiter": 8}
    assert fenced.admission_decode_fence.remaining == 0


def test_real_allocator_failure_preempts_and_releases(fenced, monkeypatch):
    owner = arm(fenced)
    add_request(fenced, "waiter", 8)
    allocate = fenced.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        fenced.kv_cache_manager,
        'allocate_slots',
        lambda req, *a, **kw: None if req is owner else allocate(req, *a, **kw),
    )
    out = fenced.schedule()
    assert not out.num_scheduled_tokens and owner.status == RequestStatus.PREEMPTED
    assert fenced.admission_decode_fence.remaining == 0
    monkeypatch.setattr(fenced.kv_cache_manager, 'allocate_slots', allocate)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {'cold': 72, 'waiter': 8}
    assert fenced.admission_decode_fence.counts['armed_epochs'] == 1


def test_cancel_one_owner_preserves_other_four_opportunities(fenced):
    fenced.max_num_scheduled_tokens = 256
    for name in ['a', 'b']:
        add_request(fenced, name, 71)
    complete(fenced, fenced.schedule())
    Scheduler.finish_requests(fenced, 'a', RequestStatus.FINISHED_ABORTED)
    add_request(fenced, 'waiter', 8)
    for _ in range(4):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {'b': 1}
        complete(fenced, out)
    assert fenced.schedule().num_scheduled_tokens == {'waiter': 8}


def test_multiple_new_owners_share_one_bounded_budget(fenced):
    fenced.max_num_scheduled_tokens = 256
    for n in ("a", "b", "c"):
        add_request(fenced, n, 71)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == dict(a=71, b=71, c=71)
    complete(fenced, out)
    add_request(fenced, "waiter", 8)
    for _ in range(4):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == dict(a=1, b=1, c=1)
        complete(fenced, out)
    assert fenced.admission_decode_fence.counts["protected_decode_rows"] == 12
    assert fenced.schedule().num_scheduled_tokens == {"waiter": 8}


def test_async_placeholders_do_not_need_output_resolution(fenced):
    owner = add_request(fenced, "cold", 71)
    initial = fenced.schedule()
    add_request(fenced, "waiter", 8)
    outs = [fenced.schedule() for _ in range(4)]
    assert all(out.num_scheduled_tokens == {"cold": 1} for out in outs)
    assert owner.num_computed_tokens == owner.num_in_flight_tokens == 75
    assert owner.num_output_placeholders == 5
    assert fenced.admission_decode_fence.remaining == 0
    for out in [initial] + outs:
        complete(fenced, out)
    assert owner.num_in_flight_tokens == owner.num_output_placeholders == 0


def test_no_prefill_steady_path_and_finished_idle(fenced):
    owner = add_request(fenced, "steady", 8)
    complete(fenced, fenced.schedule())
    for _ in range(8):
        out = fenced.schedule()
        assert out.num_scheduled_tokens == {"steady": 1}
        complete(fenced, out)
    assert fenced.admission_decode_fence.counts == Counter()
    Scheduler.finish_requests(fenced, owner.request_id, RequestStatus.FINISHED_ABORTED)
    assert not fenced.schedule().num_scheduled_tokens


def test_unsupported_throttle_clears_fence(fenced):
    arm(fenced)
    add_request(fenced, "waiter", 8)
    out = fenced.schedule(throttle_prefills=True)
    assert fenced.phase_admission_phase == "mixed" and "waiter" not in out.num_scheduled_tokens
    assert fenced.admission_decode_fence.remaining == 0


def test_schedule_exception_clears_ownership_and_preserves_error(fenced, monkeypatch):
    arm(fenced)
    error = ValueError("original schedule failure")

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(OmniARAsyncScheduler, "schedule", fail)
    with pytest.raises(ValueError) as exc:
        fenced.schedule()
    assert exc.value is error
    assert fenced.admission_decode_fence.remaining == 0 and not fenced.admission_decode_fence.owners


def test_empty_decode_releases_bound_without_starvation(fenced, monkeypatch):
    arm(fenced)

    def empty(self, *a, **k):
        self.phase_admission_phase = "decode"
        return SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0, scheduled_new_reqs=[])

    monkeypatch.setattr(OmniARAsyncScheduler, "schedule", empty)
    fenced.schedule()
    assert fenced.admission_decode_fence.remaining == 0
    assert fenced.admission_decode_fence.counts["released_non_decode_or_empty"] == 1


def test_resumed_session_not_new_cold71(fenced):
    from vllm import SamplingParams
    from vllm.v1.request import StreamingUpdate

    owner = add_request(fenced, "resumed", 8)
    complete(fenced, fenced.schedule())
    fenced.running.remove(owner)
    owner.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    fenced.num_waiting_for_streaming_input = 1
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=list(range(63)),
        max_tokens=20,
        arrival_time=2.0,
        sampling_params=SamplingParams(max_tokens=20),
    )
    fenced._update_request_as_session(owner, update)
    fenced.waiting.add_request(owner)
    out = fenced.schedule()
    assert out.num_scheduled_tokens == {"resumed": 63}
    assert owner.num_computed_tokens == 71 and fenced.admission_decode_fence.remaining == 0


def test_startup_and_shutdown_receipts_no_hot_writes(scheduler, monkeypatch, tmp_path):
    monkeypatch.setenv("EASYMAGPIE_ADMISSION_FENCE_STATUS_DIR", str(tmp_path))
    scheduler.admission_decode_fence = AdmissionDecodeFence(scheduler)
    arm(scheduler)
    assert len(list(tmp_path.glob('*.json'))) == 1
    monkeypatch.setattr(OmniARAsyncScheduler, "shutdown", lambda self: "original-result")
    assert scheduler.shutdown() == "original-result"
    receipts = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert {x['lifecycle'] for x in receipts} == {'startup', 'shutdown'}
    end = next(x for x in receipts if x['lifecycle'] == 'shutdown')
    assert end['fence']['counts']['armed_epochs'] == 1 and len(end['source_sha256']) == 3


def test_shutdown_original_exception_survives_receipt_failure(fenced, monkeypatch):
    arm(fenced)
    error = RuntimeError('original shutdown')

    def fail(self):
        raise error

    def badwrite(*args):
        raise OSError('receipt')

    monkeypatch.setattr(OmniARAsyncScheduler, 'shutdown', fail)
    monkeypatch.setattr(fenced.admission_decode_fence, 'write', badwrite)
    with pytest.raises(RuntimeError) as exc:
        fenced.shutdown()
    assert exc.value is error
    assert fenced.admission_decode_fence.remaining == 0 and not fenced.admission_decode_fence.owners


def test_cancel_after_final_arm_then_shutdown_releases_strong_owners(fenced, monkeypatch):
    owner = arm(fenced)
    Scheduler.finish_requests(fenced, owner.request_id, RequestStatus.FINISHED_ABORTED)
    assert fenced.admission_decode_fence.owners['cold'] is owner
    monkeypatch.setattr(OmniARAsyncScheduler, 'shutdown', lambda self: None)
    fenced.shutdown()
    assert fenced.admission_decode_fence.remaining == 0 and not fenced.admission_decode_fence.owners
    assert fenced.admission_decode_fence.counts['shutdown_releases'] == 1


def test_actual_lifetime_receipts_keep_source_identity_and_drained_state(scheduler, monkeypatch, tmp_path):
    monkeypatch.delenv('EASYMAGPIE_ADMISSION_FENCE_STATUS_DIR', raising=False)
    monkeypatch.setenv('VLLM_STARTUP_EVIDENCE', str(tmp_path))
    scheduler.admission_decode_fence = AdmissionDecodeFence(scheduler)
    arm(scheduler)
    add_request(scheduler, 'waiting', 8)
    for _ in range(4):
        complete(scheduler, scheduler.schedule())
    monkeypatch.setattr(OmniARAsyncScheduler, 'shutdown', lambda self: None)
    scheduler.shutdown()
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
    assert {item['lifecycle'] for item in receipts} == {'startup', 'shutdown'}
    startup = next(item for item in receipts if item['lifecycle'] == 'startup')
    shutdown = next(item for item in receipts if item['lifecycle'] == 'shutdown')
    assert startup['pid'] == shutdown['pid']
    assert startup['source_sha256'] == shutdown['source_sha256']
    assert shutdown['fence']['decode_limit'] == 4
    assert shutdown['fence']['counts']['decode_steps'] == 4
    assert shutdown['fence']['remaining'] == 0


@pytest.mark.parametrize('phase,stage,valid', [(True, 0, True), (False, 0, False), (True, 1, False)])
def test_constructor_guards(scheduler, phase, stage, valid):
    scheduler.phase_admission_enabled = phase
    scheduler.vllm_config.model_config.stage_id = stage
    if valid:
        assert AdmissionDecodeFence(scheduler).remaining == 0
    else:
        with pytest.raises(RuntimeError):
            AdmissionDecodeFence(scheduler)


def test_default_off_real_constructor_and_codec_scope(monkeypatch):
    def init(self, *a, **k):
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
        self.chunk_transfer_adapter = None

    monkeypatch.setattr(OmniARAsyncScheduler, '__init__', init)
    monkeypatch.delenv('EASYMAGPIE_ADMISSION_DECODE_FENCE', raising=False)
    monkeypatch.setenv('EASYMAGPIE_PHASE_ADMISSION', '1')
    assert EasyMagpieARAsyncScheduler().admission_decode_fence is None
    monkeypatch.setenv('EASYMAGPIE_ADMISSION_DECODE_FENCE', '1')
    assert EasyMagpieARAsyncScheduler().admission_decode_fence.remaining == 0
    # Enabling the talker fence does not add a fence to the codec scheduler.
    from easymagpie_vllm_omni.scheduler import EasyMagpieCodecScheduler
    from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

    def codec_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(stage_id=1, hf_config=SimpleNamespace(num_stacked_codebooks=16))
        )
        self.chunk_transfer_adapter = SimpleNamespace()

    monkeypatch.setattr(OmniGenerationScheduler, '__init__', codec_init)
    assert not hasattr(EasyMagpieCodecScheduler(), 'admission_decode_fence')
