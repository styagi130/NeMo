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
"""Terminal streaming codec control comes from the connector, not another prewarm."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import vllm_plugin_easymagpie_omni as plugin
from test_codec_completion import _processor, _raw, _request
from vllm_omni.engine import orchestrator
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool


@pytest.mark.asyncio
@pytest.mark.parametrize("already_finished", [False, True])
async def test_terminal_stream_never_submits_another_codec_request(monkeypatch, already_finished):
    owner, request, state, codec, client = _runtime(monkeypatch)
    if already_finished:
        raw = _raw("request")
        raw.multimodal_output = None
        result = codec.output_processor.process_outputs([raw], None, None)
        assert result.request_outputs[0].finished
        state.finished_final_output_stage_ids.add(1)
        owner.request_states.clear()
    before = dict(codec.output_processor.request_states)
    add = Mock(wraps=codec.output_processor.add_request)
    monkeypatch.setattr(codec.output_processor, "add_request", add)

    await owner._prewarm_async_chunk_stages("request", request, state)

    add.assert_not_called()
    client.add_request_async.assert_not_awaited()
    assert codec.output_processor.request_states == before


@pytest.mark.asyncio
async def test_final_codec_completion_can_outrun_the_stage0_send(monkeypatch):
    owner, request, state, codec, client = _runtime(monkeypatch)
    sent, resume = asyncio.Event(), asyncio.Event()

    async def delayed_send(request):
        sent.set()
        await resume.wait()

    owner.stage_pools[0].clients[0].add_request_async = delayed_send
    message = SimpleNamespace(
        request_id="request", prompt=request, final_stage_id=1, sampling_params_list=[], output_prompt_text=None
    )
    task = asyncio.create_task(owner._handle_streaming_update(message))
    try:
        await asyncio.wait_for(sent.wait(), 1)
        raw = _raw("request")
        raw.multimodal_output = None
        result = codec.output_processor.process_outputs([raw], None, None)
        assert result.request_outputs[0].finished and not codec.output_processor.request_states
        state.finished_final_output_stage_ids.add(1)
        owner.request_states.clear()
        resume.set()
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not codec.output_processor.request_states
    assert not codec.output_processor.external_req_ids
    client.add_request_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_stream_does_not_enter_a_delayed_codec_pick(monkeypatch):
    owner, request, state, codec, client = _runtime(monkeypatch)
    entered = asyncio.Event()

    async def delayed_pick(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    pick = AsyncMock(side_effect=delayed_pick)
    monkeypatch.setattr(codec, "_pick_or_select", pick)
    task = asyncio.create_task(owner._prewarm_async_chunk_stages("request", request, state))
    try:
        await asyncio.sleep(0)
        assert task.done(), "A terminal dummy must not wait for codec replica selection"
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not entered.is_set()
    pick.assert_not_awaited()
    client.add_request_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming,resumable", [(False, False), (True, True)])
async def test_http_initial_and_resumable_updates_keep_actual_prewarm(monkeypatch, streaming, resumable):
    owner, request, state, codec, client = _runtime(monkeypatch)
    state.streaming.enabled = streaming
    request.resumable = resumable
    add = Mock(wraps=codec.output_processor.add_request)
    monkeypatch.setattr(codec.output_processor, "add_request", add)

    await owner._prewarm_async_chunk_stages("request", request, state)

    assert add.call_count == 1
    client.add_request_async.assert_awaited_once()
    assert client.add_request_async.await_args.args[0].resumable is resumable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "http",
        "nonfinal",
        "foreign",
        "missing_model",
        "missing_resumable",
        "non_async",
        "multistage",
        "different_final_stage",
    ],
)
async def test_unrelated_prewarm_calls_delegate_once_with_same_arguments(monkeypatch, case):
    owner, request, state, _, _ = _runtime(monkeypatch)
    if case == "http":
        state.streaming.enabled = False
    elif case == "nonfinal":
        request.resumable = True
    elif case == "foreign":
        owner.stage_pools[1]._stage_vllm_config.model_config.hf_config.model_type = "other_codec"
    elif case == "missing_model":
        owner.stage_pools[1]._stage_vllm_config = None
    elif case == "missing_resumable":
        request = SimpleNamespace(prompt_token_ids=[0])
    elif case == "non_async":
        owner.async_chunk = False
    elif case == "multistage":
        owner.stage_pools.append(owner.stage_pools[1])
        owner.num_stages = 3
        state.final_stage_id = 2
    else:
        state.final_stage_id = 0
    callback = AsyncMock(return_value=object())
    monkeypatch.setattr(Orchestrator, "_prewarm_async_chunk_stages", callback)
    plugin.register()

    result = await Orchestrator._prewarm_async_chunk_stages(owner, "request", request, state)

    assert result is callback.return_value
    callback.assert_awaited_once_with(owner, "request", request, state)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("prewarm failed"), asyncio.CancelledError()])
async def test_delegated_exception_and_cancellation_are_preserved(monkeypatch, error):
    owner, request, state, _, _ = _runtime(monkeypatch)
    request.resumable = True
    callback = AsyncMock(side_effect=error)
    monkeypatch.setattr(Orchestrator, "_prewarm_async_chunk_stages", callback)
    plugin.register()

    with pytest.raises(type(error)) as raised:
        await Orchestrator._prewarm_async_chunk_stages(owner, "request", request, state)

    assert raised.value is error
    callback.assert_awaited_once_with(owner, "request", request, state)


def test_prewarm_registration_is_idempotent(monkeypatch):
    _runtime(monkeypatch)
    callback = Orchestrator._prewarm_async_chunk_stages
    plugin.register()
    assert Orchestrator._prewarm_async_chunk_stages is callback


def _runtime(monkeypatch):
    original = inspect.unwrap(Orchestrator._prewarm_async_chunk_stages)
    monkeypatch.setattr(Orchestrator, "_prewarm_async_chunk_stages", original)
    plugin.register()
    pools = []
    for index, model_type in enumerate(("easymagpie", "easymagpie_codec")):
        client = SimpleNamespace(stage_type="llm", final_output=True, add_request_async=AsyncMock())
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))
        processor = _processor(model_type=model_type)
        processor.add_request(_request("request"), prompt=None, queue=None)
        pool = StagePool(index, [client], output_processor=processor, stage_vllm_config=config)
        pool._request_bindings["request"] = 0
        pools.append(pool)
    owner = Orchestrator.__new__(Orchestrator)
    owner.stage_pools, owner.num_stages, owner.async_chunk = pools, 2, True
    owner._stage_receives_async_chunks = lambda _: True
    owner._record_duplex_stage_submission = Mock()
    owner._emit_tx_edge = Mock()
    state = SimpleNamespace(
        request_id="request",
        final_stage_id=1,
        prompt={},
        sampling_params_list=[None, None],
        stage_submit_ts={0: 0.0, 1: 0.0},
        streaming=SimpleNamespace(enabled=True),
        finished_final_output_stage_ids=set(),
    )
    owner.request_states = {"request": state}
    monkeypatch.setattr(
        orchestrator,
        "build_engine_core_request_from_tokens",
        lambda **kwargs: _request(kwargs["request_id"], resumable=kwargs["resumable"]),
    )
    return owner, _request("request", resumable=False), state, pools[1], pools[1].clients[0]
