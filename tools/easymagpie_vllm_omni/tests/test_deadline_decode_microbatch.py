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
"""Stdlib source-derived publication/fence/selection component tests; no Torch."""
import ast
import enum
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from easymagpie_vllm_omni import deadline_decode_microbatch as candidate
from easymagpie_vllm_omni import ready_deadlines as deadline
from easymagpie_vllm_omni.admission_fence import AdmissionDecodeFence as Fence
from vllm.v1.request import RequestStatus as Status


class Clock:
    now = 1_000_000_000

    def __call__(self):
        return self.now


class Component(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.queue = deadline.ReadyDeadlineQueue(clock=self.clock)
        self.policy = candidate.DeadlineDecodePolicy(self.queue, clock=self.clock)
        self.rows = [
            NS(
                request_id='r' + str(i),
                external_req_id='e' + str(i),
                status=Status.RUNNING,
                num_preemptions=0,
                resumable=False,
                num_computed_tokens=80,
                num_tokens_with_spec=81,
                num_output_placeholders=0,
                next_decode_eligible_step=0,
                num_prompt_tokens=71,
                max_tokens=1000,
            )
            for i in range(144)
        ]
        self.s = NS(
            running=self.rows[:],
            requests={r.request_id: r for r in self.rows},
            current_step=20,
            phase_admission_phase='decode',
            phase_admission_enabled=True,
            phase_admission_throttled=False,
            num_spec_tokens=0,
            num_sampled_tokens_per_step=1,
            parallel_config=NS(pipeline_parallel_size=1),
            connector=None,
            ec_connector=None,
            num_waiting_for_streaming_input=0,
            _pause_state=NS(name='UNPAUSED'),
            vllm_config=NS(model_config=NS(stage_id=0)),
            phase_admission_force_decode=False,
        )
        self.s._phase_request_is_prefill = (
            lambda r: r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens > 1
        )
        self.fence = Fence(self.s)
        self.s.admission_decode_fence = self.fence
        for i, r in enumerate(self.rows):
            owner = (r.external_req_id, i + 1)
            self.queue.register(owner, r, arrival_ns=self.clock())
            self.commit(r, 2)
            if i >= 8:
                self.commit(r, 6)
                self.commit(r, 8)

    def commit(self, request, frames, finish=False):
        owner = self.queue._rid_owners[request.external_req_id][0]
        record = dict(owner=owner, request=request, produced_frames=frames, ready_ns=self.clock(), is_finished=finish)
        assert self.queue.offer(record)
        assert self.queue.take_next() is record
        assert self.queue.complete(record, True)
        return record

    def choose(self):
        return self.policy.choose(self.s, lambda r: True)

    def actual_selection(self):
        self.s.current_step += 1
        return [r.request_id for r in self.s.running if self.policy.allow(self.s, r, True)]

    def test_first_F2_and_buffered_steady_selection_preserves_order(self):
        order = list(self.s.running)
        self.assertTrue(self.choose())
        self.assertEqual(self.actual_selection(), ['r' + str(i) for i in range(8)])
        self.assertEqual(self.s.running, order)
        self.assertEqual(self.policy.status()['retained_Request_references'], 0)

    def test_endangered_steady_is_mandatory(self):
        self.clock.now += 1_150_000_000
        self.assertFalse(self.choose())  # All144 budgets now unsafe; full fallback.
        self.assertEqual(self.policy.counts['fallback:too_many_mandatory_owners'], 1)

    def test_canonical_fence_cohort_each_gets_all_four_opportunities(self):
        self.fence.remaining = 4
        self.fence.owners = {r.request_id: r for r in self.rows[8:13]}
        protected = 0
        for step in range(4):
            self.choose()
            chosen = self.actual_selection()
            self.assertTrue(set(self.fence.owners).issubset(chosen))
            output = NS(num_scheduled_tokens={rid: 1 for rid in chosen})
            self.fence.after_schedule(self.s, output)
            self.policy.observe(self.s, output)
            self.policy.clear_plan()
            protected += 5
        self.assertEqual(self.fence.remaining, 0)
        self.assertEqual(self.fence.counts['protected_decode_rows'], protected)
        self.assertEqual(self.policy.counts['full_service_schedules'], 1)

    def test_unpublished_cold_fence_owner_still_mandatory(self):
        r = self.rows[130]
        self.queue.cancel(self.queue._rid_owners[r.external_req_id][0])
        self.fence.remaining = 4
        self.fence.owners = {r.request_id: r}
        self.assertTrue(self.choose())
        self.assertIn(r.request_id, self.actual_selection())

    def test_unpublished_other_owner_falls_back_all(self):
        r = self.rows[-1]
        self.queue.cancel(self.queue._rid_owners[r.external_req_id][0])
        self.assertFalse(self.choose())
        self.assertEqual(len(self.actual_selection()), 144)

    def test_external_id_reuse_cannot_borrow_previous_generation_budget(self):
        old = self.rows[-1]
        replacement = NS(**vars(old))
        self.s.running[-1] = replacement
        self.s.requests[replacement.request_id] = replacement
        self.assertFalse(self.choose())
        self.assertEqual(self.policy.counts['fallback:nonfence_publication_authority_missing'], 1)

    def test_inflight_nextchunk_releases_startup_priority_without_crediting_samples(self):
        r = self.rows[0]
        owner = self.queue._rid_owners[r.external_req_id][0]
        record = dict(owner=owner, request=r, produced_frames=6, ready_ns=self.clock())
        self.queue.offer(record)
        self.queue.take_next()
        self.assertTrue(self.choose())
        # F2 end is exactly160ms away, so conservative endangered criterion
        # still includes it; its queued F6 is not counted as committed audio.
        self.assertIn(r.request_id, self.policy.plan)
        self.assertEqual(self.queue._states[owner].estimated_samples, 3528)

    def test_prefill_unsupported_resume_and_pause_keep_original(self):
        self.s.phase_admission_phase = 'prefill'
        self.assertFalse(self.choose())
        self.s.phase_admission_phase = 'decode'
        self.rows[-1].num_preemptions = 1
        self.assertFalse(self.choose())
        self.rows[-1].num_preemptions = 0
        self.s._pause_state = NS(name='PAUSED_ALL')
        self.assertFalse(self.choose())

    def test_two_limited_steps_force_full_service_and_resume_deferred(self):
        selected = []
        for i in range(3):
            self.choose()
            chosen = self.actual_selection()
            selected.append(chosen)
            self.policy.observe(self.s, NS(num_scheduled_tokens={r: 1 for r in chosen}))
            self.policy.clear_plan()
        self.assertEqual(list(map(len, selected)), [8, 8, 144])
        self.assertEqual(self.policy.limited_steps, 0)

    def test_wall_window_measured_from_selection_not_completion(self):
        self.choose()
        chosen = self.actual_selection()
        self.clock.now += 41_000_000
        self.policy.observe(self.s, NS(num_scheduled_tokens={r: 1 for r in chosen}))
        self.policy.clear_plan()
        self.assertFalse(self.choose())

    def test_allocation_or_missing_fence_progress_forces_next_original(self):
        self.fence.remaining = 4
        self.fence.owners = {'r9': self.rows[9]}
        self.assertTrue(self.choose())
        self.s.current_step += 1
        self.policy.observe(self.s, NS(num_scheduled_tokens={'r0': 1}))
        self.policy.clear_plan()
        self.assertFalse(self.choose())
        self.assertTrue(self.policy.force_full)

    def test_cancel_close_and_exception_cleanup_hold_no_Request_references(self):
        self.assertTrue(self.choose())
        self.policy.close()
        self.assertIsNone(self.policy.plan)
        self.assertFalse(self.policy.eligible)
        self.assertFalse(self.policy.fence_required)
        self.assertFalse(self.choose())


if __name__ == '__main__':
    unittest.main(verbosity=2)
