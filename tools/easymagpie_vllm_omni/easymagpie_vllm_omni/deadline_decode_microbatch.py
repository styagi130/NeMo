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
"""Bounded Q1 selection using committed publication deadlines.

All cold-fence and endangered owners remain mandatory. Unknown authority or
more than 128 mandatory rows restores full service. Two limited schedules or
40 ms require a full eligible schedule before another limited window.
"""
import time
from collections import Counter


def _running(request):
    return (
        getattr(getattr(request, 'status', None), 'name', None) == 'RUNNING'
        and request.num_preemptions == 0
        and request.resumable is False
    )


def _has_Q1(scheduler, request):
    # Same eligibility arithmetic as the pinned phase selector, before the
    # runtime increments current_step. The original hook remains authoritative.
    return (
        _running(request)
        and scheduler.current_step + 1 >= request.next_decode_eligible_step
        and request.num_tokens_with_spec + request.num_output_placeholders > request.num_computed_tokens
        and not scheduler._phase_request_is_prefill(request)
        and not (
            request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        )
    )


class DeadlineDecodePolicy:
    """One ephemeral owner snapshot; four-step cold fence semantics preserved.

    Mandatory union: every currently eligible canonical cold-fence owner, every
    committed F2 owner whose next F6 is not produced, and every stream whose
    committed-publication budget ends within160ms. Unknown non-fence authority,
    >128 mandatory owners, unsupported state, or fairness limits restore the
    original full eligible set. No wait or padding guarantee is introduced.
    """

    first_samples = 3528
    next_cumulative_samples = 14112
    reserve_ns = 160_000_000
    max_limited_steps = 2
    max_window_ns = 40_000_000
    caps = (64, 128)

    def __init__(self, queue, *, clock=time.monotonic_ns):
        self.queue = queue
        self.clock = clock
        self.plan = None
        self.eligible = {}
        self.fence_required = set()
        self.plan_step = None
        self.plan_started_ns = None
        self.limited_steps = 0
        self.window_start = None
        self.force_full = False
        self.closed = False
        self.counts = Counter()

    def clear_plan(self):
        self.plan = None
        self.eligible.clear()
        self.fence_required.clear()
        self.plan_step = None
        self.plan_started_ns = None

    def fallback(self, reason):
        self.counts['fallback:' + reason] += 1
        self.plan = None
        return False

    def choose(self, scheduler, original_eligible):
        self.clear_plan()
        self.counts['choices'] += 1
        if (
            self.closed
            or scheduler.phase_admission_phase != 'decode'
            or not scheduler.phase_admission_enabled
            or scheduler.phase_admission_throttled
            or scheduler.num_spec_tokens != 0
            or scheduler.num_sampled_tokens_per_step != 1
            or scheduler.parallel_config.pipeline_parallel_size != 1
            or scheduler.connector is not None
            or scheduler.ec_connector is not None
            or scheduler.num_waiting_for_streaming_input != 0
            or getattr(scheduler._pause_state, 'name', None) != 'UNPAUSED'
        ):
            return self.fallback('unsupported_or_original_prefill')
        raw_rows = [
            r
            for r in scheduler.running
            if original_eligible(r)
            and scheduler.current_step + 1 >= r.next_decode_eligible_step
            and r.num_tokens_with_spec + r.num_output_placeholders > r.num_computed_tokens
            and not scheduler._phase_request_is_prefill(r)
            and not (
                r.num_output_placeholders > 0
                and r.num_computed_tokens + 2 - r.num_output_placeholders >= r.num_prompt_tokens + r.max_tokens
            )
        ]
        if any(not _running(r) for r in raw_rows):
            return self.fallback('unsupported_recompute_resume_or_owner')
        rows = raw_rows
        self.eligible = {r.request_id: id(r) for r in rows}
        self.plan_step = scheduler.current_step + 1
        if len(self.eligible) != len(rows):
            return self.fallback('duplicate_internal_id')
        if len(rows) <= self.caps[0]:
            return self.fallback('already_small')
        now = self.clock()
        if (
            self.force_full
            or self.limited_steps >= self.max_limited_steps
            or self.window_start is not None
            and now - self.window_start >= self.max_window_ns
        ):
            return self.fallback('full_service_bound')
        fence = scheduler.admission_decode_fence
        if fence is None or fence.limit != 4 or getattr(fence, 'max_wait_ms', 0) != 0:
            return self.fallback('canonical_fence_missing')
        mandatory = {}
        if fence.remaining:
            for rid, request in fence.owners.items():
                if rid in self.eligible:
                    if scheduler.requests.get(rid) is not request or id(request) != self.eligible[rid]:
                        return self.fallback('fence_owner_reused')
                    self.fence_required.add(rid)
                    mandatory[rid] = (id(request), None)
        with self.queue._lock:
            if self.queue._closed:
                return self.fallback('publication_queue_closed')
            for order, request in enumerate(rows):
                rid = request.request_id
                if scheduler.requests.get(rid) is not request:
                    return self.fallback('scheduler_owner_changed')
                owners = self.queue._rid_owners.get(request.external_req_id)
                state = self.queue._states.get(owners[0]) if owners else None
                # New cold-fence owners can precede sender registration/F2;
                # they stay mandatory, never receive a fabricated audio budget.
                if rid in self.fence_required:
                    if state is not None and state.request is request:
                        mandatory[rid] = (id(request), state.owner)
                    continue
                if (
                    state is None
                    or state.request is not request
                    or state.closed
                    or state.cancelled
                    or state.terminal_acknowledged
                    or state.first_publication_ns is None
                    or state.estimated_end_ns is None
                    or state.estimated_samples < self.first_samples
                ):
                    return self.fallback('nonfence_publication_authority_missing')
                needs_F6 = (
                    state.estimated_samples == self.first_samples
                    and state.last_dispatched == 0
                    and state.next_sequence == 1
                    and not state.ready
                    and state.inflight is None
                )
                endangered = state.estimated_end_ns <= now + self.reserve_ns
                if needs_F6 or endangered:
                    mandatory[rid] = (id(request), state.owner)
        count = len(mandatory)
        if not count:
            return self.fallback('no_urgent_producer')
        if count > self.caps[-1]:
            return self.fallback('too_many_mandatory_owners')
        if count == len(rows):
            return self.fallback('all_eligible_urgent')
        cap = self.caps[0] if count <= self.caps[0] else self.caps[1]
        self.plan = mandatory
        self.plan_started_ns = now
        self.counts['limited_choices'] += 1
        self.counts['cap:' + str(cap)] += 1
        self.counts['mandatory_selected_rows'] += count
        self.counts['mandatory_fence_rows'] += len(self.fence_required)
        self.counts['deferred_row_opportunities'] += len(rows) - len(self.plan)
        return True

    def allow(self, scheduler, request, original_allowed):
        if not original_allowed or self.plan is None:
            return original_allowed
        if scheduler.current_step != self.plan_step:
            raise RuntimeError('deadline decode eligibility outside its original schedule')
        expected = self.eligible.get(request.request_id)
        if expected is None:
            # Preserve original non-Q1/empty/cadence handling; no selected work
            # can be introduced in this phase by the runtime waiting queue.
            return original_allowed
        if scheduler.requests.get(request.request_id) is not request or id(request) != expected:
            raise RuntimeError('deadline decode current Request identity changed')
        return request.request_id in self.plan

    def observe(self, scheduler, output):
        scheduled = output.num_scheduled_tokens
        if self.plan is not None:
            if (
                scheduler.phase_admission_phase != 'decode'
                or any(n != 1 for n in scheduled.values())
                or not self.fence_required.issubset(scheduled)
                or not set(scheduled).issubset(self.plan)
                or set(self.plan) != set(scheduled)
                or getattr(output, 'preempted_req_ids', None)
            ):
                self.force_full = True
                self.counts['limited_allocation_or_progress_anomalies'] += 1
            self.limited_steps += 1
            if self.window_start is None:
                self.window_start = self.plan_started_ns
            self.counts['limited_schedules'] += 1
            self.counts['limited_Q1_rows'] += sum(n == 1 for n in scheduled.values())
        elif (
            scheduler.phase_admission_phase == 'decode'
            and self.eligible
            and all(scheduled.get(rid) == 1 for rid in self.eligible)
        ):
            self.force_full = False
            self.limited_steps = 0
            self.window_start = None
            self.counts['full_service_schedules'] += 1

    def close(self):
        self.clear_plan()
        self.closed = True

    def status(self):
        return dict(
            enabled=True,
            cap_choices=list(self.caps),
            reserve_ms=self.reserve_ns / 1e6,
            maximum_consecutive_limited_schedules=self.max_limited_steps,
            maximum_limited_window_ms=self.max_window_ns / 1e6,
            clock_scope='committed connector publication; not client playback',
            retained_Request_references=0,
            active_plan_rows=len(self.plan or {}),
            eligible_metadata_rows=len(self.eligible),
            closed=self.closed,
            counts=dict(self.counts),
        )


def create_policy(scheduler):
    """Construct the retained C256 policy before any owner is admitted."""
    from .admission_fence import AdmissionDecodeFence
    from .bulk_native import BulkChunkAccumulator
    from .bulk_sender import BulkSender
    from .ready_deadlines import ReadyDeadlineQueue

    if (
        scheduler.vllm_config.model_config.stage_id != 0
        or not scheduler.phase_admission_enabled
        or scheduler._easymagpie_phase_hooks_version != 1
        or scheduler.max_num_running_reqs != 256
        or scheduler.max_num_scheduled_tokens != 8165
        or type(scheduler.admission_decode_fence) is not AdmissionDecodeFence
        or scheduler.admission_decode_fence.limit != 4
        or scheduler.admission_decode_fence.remaining
        or scheduler.requests
        or scheduler.running
    ):
        raise ValueError('deadline decode requires the idle C256 phase-admission talker')
    controller = scheduler.bulk_sender
    queue = controller.deadline_policy
    if (
        type(controller) is not BulkSender
        or controller.adapter is not scheduler.chunk_transfer_adapter
        or type(controller.backend) is not BulkChunkAccumulator
        or type(queue) is not ReadyDeadlineQueue
        or controller.backend_name != 'native'
        or tuple(controller.startup_frames) != (2, 6)
    ):
        raise ValueError('deadline decode requires the native committed-publication queue')
    with queue._lock:
        if queue._closed or queue._states or queue._rid_owners:
            raise ValueError('deadline decode must initialize before owner registration')
    return DeadlineDecodePolicy(queue)
