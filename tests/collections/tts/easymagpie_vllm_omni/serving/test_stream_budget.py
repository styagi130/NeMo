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
"""Global WebSocket token budgets with the real upstream input feeder."""

import asyncio
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.serving_stream import EasyMagpieInputStream
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm_omni.entrypoints.async_omni import AsyncOmni


class PacedEngine:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []
        self.tasks = []
        self.first_output = asyncio.Event()
        self.output_gate = asyncio.Event()
        self.output_gate.set()

    async def add_request_async(self, **kwargs):
        await self._add(kwargs)

    async def add_streaming_update_async(self, **kwargs):
        await self._add(kwargs)

    async def _add(self, kwargs):
        self.calls.append(kwargs)
        if kwargs['resumable']:
            self.tasks.append(asyncio.create_task(self._output(kwargs)))

    async def _output(self, kwargs):
        await self.output_gate.wait()
        await asyncio.sleep(0)
        count = kwargs['sampling_params_list'][0].max_tokens
        # Separate data from the terminal callback: reaching the count is not
        # itself permission to resume or finalize the active segment.
        self.stream.observe_output(
            SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[0] * count, finish_reason=None)])
        )
        await asyncio.sleep(0)
        self.stream.observe_output(
            SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[], finish_reason='length')])
        )
        self.first_output.set()

    async def drain(self):
        await asyncio.gather(*self.tasks)


def make_stream(cap, *, coalesce=False, queue_depth=2):
    params = SamplingParams(max_tokens=2048, output_kind=RequestOutputKind.DELTA)
    stream = EasyMagpieInputStream(
        prefill_prompt={'prompt_token_ids': [0] * 8},
        sampling_params=params,
        text_eos_id=99,
        max_new_tokens=cap,
        text_prefill_num=4,
        coalesce_queued_tokens=coalesce,
        queue_depth=queue_depth,
        pace_timeout_s=1,
    )
    return stream, params


async def start_feeder(stream, params):
    engine = PacedEngine(stream)
    state = SimpleNamespace(queue=asyncio.Queue())
    omni = SimpleNamespace(
        engine=engine,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        request_states={'cap-test': state},
        _validate_streaming_input_sampling_params=AsyncOmni._validate_streaming_input_sampling_params,
    )
    task = await AsyncOmni._add_streaming_input_request(
        omni,
        request_id='cap-test',
        input_stream=stream.inputs(),
        sampling_params_list=[params],
        final_stage_id=1,
        final_output_stage_ids=[1],
        arrival_time=0,
    )
    return engine, state, task


async def run_session(cap, chunks, *, coalesce=False):
    stream, params = make_stream(cap, coalesce=coalesce)
    engine, state, feeder = await start_feeder(stream, params)

    async def send():
        for tokens in chunks:
            await stream.put_tokens(tokens)
        await stream.finish()

    sender = asyncio.create_task(send())
    try:
        await asyncio.wait_for(asyncio.gather(sender, feeder), 3)
        await engine.drain()
    finally:
        for task in [sender, feeder, *engine.tasks]:
            if not task.done():
                task.cancel()
        await asyncio.gather(sender, feeder, *engine.tasks, return_exceptions=True)
    errors = []
    while not state.queue.empty():
        errors.append(str(state.queue.get_nowait()))
    assert not errors, errors
    assert stream.finished
    assert len([call for call in engine.calls if not call['resumable']]) == 1
    assert engine.calls[-1]['resumable'] is False
    calls = [
        {
            'frames': call['sampling_params_list'][0].max_tokens,
            'tokens': call['prompt'].get('additional_information', {}).get('text_token'),
            'start': call['prompt'].get('additional_information', {}).get('text_token_start'),
            'resumable': call['resumable'],
        }
        for call in engine.calls
    ]
    return stream, calls


@pytest.mark.asyncio
@pytest.mark.parametrize('cap', [1, 4, 16])
async def test_cap_applies_to_initial_segment(cap):
    stream, _ = await run_session(cap, [list(range(24))])
    assert stream.observed_output_frames <= cap


@pytest.mark.asyncio
@pytest.mark.parametrize('cap', [1, 4, 16])
@pytest.mark.parametrize('coalesce', [False, True])
async def test_cap_applies_to_buffered_updates(cap, coalesce):
    stream, _ = await run_session(cap, [[1, 2, 3, 4]] + [[5]] * 24, coalesce=coalesce)
    assert stream.observed_output_frames <= cap


@pytest.mark.asyncio
@pytest.mark.parametrize('cap', [1, 4, 16])
async def test_no_extra_eos_or_tail_when_initial_segment_exhausts_cap(cap):
    stream, calls = await run_session(cap, [list(range(cap + 3))])
    assert stream.observed_output_frames == cap
    assert len([call for call in calls if call['resumable']]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('cap', [4, 16])
async def test_no_tail_when_eos_exhausts_remaining_budget(cap):
    stream, calls = await run_session(cap, [list(range(cap + 2))])
    assert stream.observed_output_frames == cap
    assert [call['tokens'] for call in calls if call['resumable']][-1] == [99]


@pytest.mark.asyncio
@pytest.mark.parametrize('chunk_size', [1, 5])
@pytest.mark.parametrize('coalesce', [False, True])
async def test_normal_one_and_five_token_sessions_complete_unchanged(chunk_size, coalesce):
    tokens = list(range(24))
    chunks = [tokens[index : index + chunk_size] for index in range(0, len(tokens), chunk_size)]
    stream, calls = await run_session(64, chunks, coalesce=coalesce)
    yielded_tokens = [token for call in calls if call['resumable'] for token in call['tokens']]
    assert yielded_tokens == tokens + [99]
    assert stream.observed_output_frames == 64
    assert calls[-2]['tokens'] == []
    assert all(call['frames'] > 0 for call in calls)


@pytest.mark.asyncio
async def test_upstream_cancellation_suppresses_terminal_update():
    stream, params = make_stream(4)
    engine, state, feeder = await start_feeder(stream, params)
    engine.output_gate.clear()
    await stream.put_tokens([1, 2, 3, 4])
    await stream.put_tokens([5])
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    feeder.cancel()
    await asyncio.gather(feeder, return_exceptions=True)
    for task in engine.tasks:
        task.cancel()
    await asyncio.gather(*engine.tasks, return_exceptions=True)
    assert engine.calls and all(call['resumable'] for call in engine.calls)
    assert state.queue.empty()
    assert stream.observed_output_frames == 0


@pytest.mark.asyncio
async def test_cancellation_after_exhaustion_does_not_send_final_input():
    stream, params = make_stream(1)
    engine, state, feeder = await start_feeder(stream, params)
    await stream.put_tokens([1, 2, 3, 4])
    await engine.first_output.wait()
    await stream.put_tokens([5])
    await asyncio.sleep(0)
    feeder.cancel()
    await asyncio.gather(feeder, return_exceptions=True)
    await engine.drain()
    assert len(engine.calls) == 1 and engine.calls[0]['resumable']
    assert stream.observed_output_frames == 1
    assert state.queue.empty()


@pytest.mark.asyncio
async def test_waits_for_finish_callback_even_when_count_reaches_cap():
    stream, _ = make_stream(1)
    await stream.put_tokens([1, 2, 3, 4])
    await stream.finish()
    inputs = stream.inputs()
    await anext(inputs)
    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[0], finish_reason=None)]))
    pending = asyncio.create_task(anext(inputs))
    await asyncio.sleep(0)
    assert not pending.done()
    stream.observe_output(SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[], finish_reason='length')]))
    try:
        await asyncio.wait_for(pending, 1)
    except StopAsyncIteration:
        pass
    finally:
        await inputs.aclose()
