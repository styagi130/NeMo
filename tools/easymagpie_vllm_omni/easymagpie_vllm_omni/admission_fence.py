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
"""Bounded CPU scheduling experiment; no acoustic/model/cache operation changes."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from vllm.v1.request import RequestStatus


class AdmissionDecodeFence:
    """Protect at most four successful decode schedules after a full cold71.

    The budget is global and cannot be extended by waiting arrivals. Every
    running decode retains the parent's eligibility/allocation order. Requests
    that cancel, preempt, resume or become ineligible cannot hold the fence.
    """

    limit = 4

    def __init__(self, scheduler):
        if not scheduler.phase_admission_enabled:
            raise RuntimeError("admission decode fence requires phase admission")
        if scheduler.vllm_config.model_config.stage_id != 0:
            raise RuntimeError("admission decode fence is talker-only")
        self.remaining = 0
        self.owners = {}
        self.counts = Counter()
        self.epoch_steps = 0
        self.source_sha256 = {}
        # Startup/lifetime receipts only; no request clocks or hot-path writes.
        directory = os.environ.get("EASYMAGPIE_ADMISSION_FENCE_STATUS_DIR") or os.environ.get("VLLM_STARTUP_EVIDENCE")
        self.receipt = Path(directory) / f"admission-fence-{os.getpid()}.json" if directory else None
        if self.receipt is not None:
            from vllm.v1.core.sched import scheduler as runtime

            for name, filename in (
                ("helper", __file__),
                ("plugin_scheduler", Path(__file__).with_name("scheduler.py")),
                ("runtime_scheduler", runtime.__file__),
            ):
                self.source_sha256[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            self.write("startup")

    def clear(self, reason):
        if self.remaining:
            self.counts[reason] += 1
        self.remaining = 0
        self.owners.clear()
        self.epoch_steps = 0

    def select(self, scheduler, has_work, waiting_prefill):
        if not self.remaining:
            return False
        live = {}
        for rid, owner in self.owners.items():
            if (
                scheduler.requests.get(rid) is owner
                and owner.status == RequestStatus.RUNNING
                and owner.num_preemptions == 0
                and owner.num_computed_tokens >= 71
                and not scheduler._phase_request_is_prefill(owner)
            ):
                live[rid] = owner
            else:
                self.counts["retired_owners"] += 1
        self.owners = live
        if not any(has_work(owner) for owner in live.values()):
            self.clear("released_no_eligible_owner")
            return False
        self.counts["selected_steps"] += 1
        if waiting_prefill:
            self.counts["waiting_admission_deferrals"] += 1
            if not scheduler.phase_admission_force_decode:
                self.counts["extra_waiting_admission_deferrals"] += 1
        return True

    def after_schedule(self, scheduler, output):
        phase = scheduler.phase_admission_phase
        scheduled = output.num_scheduled_tokens
        if self.remaining:
            if phase != "decode" or not scheduled or any(n != 1 for n in scheduled.values()):
                self.clear("released_non_decode_or_empty")
            else:
                self.remaining -= 1
                self.epoch_steps += 1
                self.counts["decode_steps"] += 1
                self.counts["protected_decode_rows"] += sum(
                    rid in scheduled and scheduler.requests.get(rid) is owner for rid, owner in self.owners.items()
                )
                self.counts["max_epoch_decode_steps"] = max(self.counts["max_epoch_decode_steps"], self.epoch_steps)
                if not self.remaining:
                    self.counts["completed_epochs"] += 1
                    self.owners.clear()
                    self.epoch_steps = 0
        if phase != "prefill":
            return
        # NewRequestData carries the *pre-schedule* computed count. A partial
        # prefill, cached/resumed request or recomputation never arms a fence.
        owners = {}
        for data in output.scheduled_new_reqs:
            rid = data.req_id
            owner = scheduler.requests.get(rid)
            if (
                data.num_computed_tokens == 0
                and data.prompt_token_ids is not None
                and len(data.prompt_token_ids) == 71
                and scheduled.get(rid) == 71
                and owner is not None
                and owner.status == RequestStatus.RUNNING
                and owner.num_preemptions == 0
                and owner.num_computed_tokens == 71
                and not owner.is_prefill_chunk
            ):
                owners[rid] = owner
        if owners:
            self.remaining = self.limit
            self.owners = owners
            self.epoch_steps = 0
            self.counts["armed_epochs"] += 1
            self.counts["qualified_cold_requests"] += len(owners)

    def status(self):
        return {
            "enabled": True,
            "version": 1,
            "cold_prompt_tokens": 71,
            "decode_limit": self.limit,
            "remaining": self.remaining,
            "protected_owner_count": len(self.owners),
            "epoch_decode_steps": self.epoch_steps,
            "counts": dict(self.counts),
            "request_clocks": False,
            "request_receipt_writes": False,
        }

    def write(self, lifecycle):
        if self.receipt is None:
            return
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "stage_id": 0,
            "pid": os.getpid(),
            "lifecycle": lifecycle,
            "source_sha256": self.source_sha256,
            "fence": self.status(),
        }
        target = (
            self.receipt if lifecycle == "startup" else self.receipt.with_name(self.receipt.stem + "-shutdown.json")
        )
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(target)
