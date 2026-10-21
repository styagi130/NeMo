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
"""Scheduler teardown restores owned delegates and preserves native failures."""
from types import SimpleNamespace

import pytest

from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler, OmniARAsyncScheduler


def scheduler():
    value = object.__new__(EasyMagpieARAsyncScheduler)
    order = []
    value.deadline_decode_policy = SimpleNamespace(close=lambda: order.append("close"))
    value.q1_no_growth_allocator = value.empty_prefix_planner = None
    value._shutdown_admission = lambda: order.append("native") or "native-result"
    return value, order


def test_native_shutdown_precedes_delegate_close():
    value, order = scheduler()
    assert value.shutdown() == "native-result"
    assert order == ["native", "close"]


def test_original_error_survives_cleanup_failure_and_other_helpers_close():
    value, order = scheduler()
    original = RuntimeError("native failure")

    def failed():
        raise original

    def failed_close():
        raise ValueError("cleanup failure")

    value._shutdown_admission = failed
    value.deadline_decode_policy.close = failed_close
    value.q1_no_growth_allocator = SimpleNamespace(close=lambda: order.append("close"))
    with pytest.raises(RuntimeError) as actual:
        value.shutdown()
    assert actual.value is original
    assert any("cleanup failure" in note for note in original.__notes__)
    assert order == ["close"]


def test_publication_queue_closes_after_native_shutdown(monkeypatch):
    value, order = scheduler()
    del value._shutdown_admission
    value.bulk_sender = SimpleNamespace(deadline_policy=SimpleNamespace(close=lambda: order.append("queue")))
    monkeypatch.setattr(OmniARAsyncScheduler, "shutdown", lambda self: order.append("native"))
    value.shutdown()
    assert order == ["native", "queue", "close"]
