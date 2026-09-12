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
"""Actual upstream shutdown methods with CPU-only fake engines and coordinator."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import vllm_plugin_easymagpie_omni as plugin
from vllm_omni.engine import membership_controller
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.messages import AbortRequestMessage, RegisterRemoteReplicaMessage, ShutdownRequestMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool


@pytest.fixture
def runtime(monkeypatch):
    # run() owns and cleans its entire loop; do not share pytest's asyncio loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    monkeypatch.setattr(Orchestrator, "_request_handler", Orchestrator._request_handler)
    plugin.register()
    events = []
    hub = SimpleNamespace(
        get_replica_list=lambda: SimpleNamespace(replicas=[]),
        close=lambda: events.append("hub_closed"),
    )
    monkeypatch.setattr(membership_controller, "OmniCoordClientForHub", lambda address: hub)

    def make(model_type="easymagpie_codec", architectures=None, membership=True):
        engine = SimpleNamespace(
            stage_type="diffusion",
            final_output=True,
            _shutting_down=False,
            get_diffusion_output_nowait=Mock(return_value=None),
            abort_requests_async=AsyncMock(),
            shutdown=Mock(side_effect=lambda: events.append("engine_closed")),
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type, architectures=architectures))
        )
        pool = StagePool(0, [engine], stage_vllm_config=config)
        member = MembershipController([pool], "unused", lambda: None, lambda *args: None) if membership else None
        if member is not None:
            member.WATCH_INTERVAL_S = 0.01
        owner = Orchestrator(asyncio.Queue(), asyncio.Queue(), asyncio.Queue(), [pool], membership_controller=member)
        return owner, member, pool, engine

    yield loop, events, make
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()
    asyncio.set_event_loop(None)


def _run(loop, owner, cancel=False):
    task = loop.create_task(owner.run())
    timed_out = []

    def timeout():
        timed_out.append(True)
        task.cancel()

    timer = loop.call_later(0.3, timeout)
    if cancel:
        loop.call_later(0.01, task.cancel)
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        if not cancel and not timed_out:
            raise
    finally:
        timer.cancel()
    assert not asyncio.all_tasks(loop)
    return bool(timed_out)


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("easymagpie_codec", None),
        ("nemotron_h", ["EasyMagpieTTS"]),
        ("nemotron_h", ["EasyMagpieTTSForConditionalGeneration"]),
    ],
)
def test_membership_stops_before_orchestrator_waits_forever(runtime, model_type, architectures):
    loop, events, make = runtime
    owner, member, _, engine = make(model_type, architectures)
    owner.request_async_queue.put_nowait(ShutdownRequestMessage())
    assert not _run(loop, owner)
    assert member._shutdown_event.is_set()
    assert member._watcher_task.done() and not member._watcher_task.cancelled()
    assert engine._shutting_down
    assert events == ["hub_closed", "engine_closed"]
    member.shutdown()
    owner._shutdown_stages()
    assert events == ["hub_closed", "engine_closed"]


def test_inflight_membership_work_drains_before_hub_and_engines_close(runtime):
    loop, events, make = runtime
    owner, member, _, _ = make()

    async def pending():
        await member._shutdown_event.wait()
        await asyncio.sleep(0.03)
        assert member._hub is not None
        events.append("membership_finished")

    async def start():
        member._spawn_task(pending(), label="pending")
        owner.request_async_queue.put_nowait(ShutdownRequestMessage())

    loop.run_until_complete(start())
    assert not _run(loop, owner)
    assert events == ["membership_finished", "hub_closed", "engine_closed"]
    assert not member._membership_tasks


@pytest.mark.parametrize("registration_error", [False, True])
def test_actual_registration_task_completes_or_reports_error_before_cleanup(runtime, registration_error, caplog):
    loop, events, make = runtime
    owner, member, pool, _ = make()
    remote = SimpleNamespace(request_address="remote", shutdown=lambda: events.append("remote_closed"))

    def factory(*args):
        assert member._hub is not None
        events.append("registration_attempted")
        if registration_error:
            raise ValueError("registration fixture failed")
        return remote

    member._remote_replica_factory = factory
    owner.request_async_queue.put_nowait(RegisterRemoteReplicaMessage(stage_id=0, replica_id=1))
    owner.request_async_queue.put_nowait(ShutdownRequestMessage())
    assert not _run(loop, owner)
    assert events.index("registration_attempted") < events.index("hub_closed")
    assert not member._membership_tasks
    if registration_error:
        assert "registration fixture failed" in caplog.text
        assert pool.num_replicas == 1
    else:
        assert pool.clients[1] is remote
        assert events[-1] == "remote_closed"


def test_pending_abort_keeps_upstream_cleanup_and_then_stops(runtime):
    loop, events, make = runtime
    owner, member, pool, engine = make()
    owner.request_states["active"] = SimpleNamespace(running_counter_registered=False)
    owner._pd_kv_params["active"] = {"fixture": True}
    pool._request_bindings["active"] = 0
    owner.request_async_queue.put_nowait(AbortRequestMessage(request_ids=["active"]))
    owner.request_async_queue.put_nowait(ShutdownRequestMessage())
    assert not _run(loop, owner)
    engine.abort_requests_async.assert_awaited_once_with(["active"])
    assert not owner.request_states and not owner._pd_kv_params and not pool._request_bindings
    assert events == ["hub_closed", "engine_closed"]
    assert member._hub is None


@pytest.mark.parametrize("membership", [False, True])
def test_output_failure_keeps_exception_and_upstream_cleanup(runtime, membership):
    loop, events, make = runtime
    owner, member, _, engine = make(membership=membership)
    error = RuntimeError("engine fixture failed")
    engine.get_diffusion_output_nowait.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        _run(loop, owner)
    assert raised.value is error
    assert owner._stages_shutdown
    assert events == (["hub_closed"] if membership else []) + ["engine_closed"]
    if member is not None:
        assert member._hub is None


def test_cancellation_keeps_upstream_cleanup(runtime):
    loop, events, make = runtime
    owner, member, _, _ = make()
    assert not _run(loop, owner, cancel=True)
    assert events == ["hub_closed", "engine_closed"]
    assert member._hub is None


@pytest.mark.parametrize("model_type,architectures", [("nemotron_h", ["NemotronHForCausalLM"]), (None, None)])
def test_non_easymagpie_watcher_behavior_is_unchanged(runtime, model_type, architectures):
    loop, events, make = runtime
    owner, _, _, _ = make(model_type, architectures)
    owner.request_async_queue.put_nowait(ShutdownRequestMessage())
    # The existing upstream cycle remains untouched for another model.
    assert _run(loop, owner)
    assert events == ["hub_closed", "engine_closed"]


def test_no_membership_uses_the_ordinary_shutdown_path(runtime):
    loop, events, make = runtime
    owner, _, _, _ = make(membership=False)
    owner.request_async_queue.put_nowait(ShutdownRequestMessage())
    assert not _run(loop, owner)
    assert events == ["engine_closed"]


def test_lifecycle_registration_is_idempotent(runtime):
    handler = Orchestrator._request_handler
    plugin.register()
    assert Orchestrator._request_handler is handler
    assert getattr(handler, "_easymagpie_shutdown", False) is True
