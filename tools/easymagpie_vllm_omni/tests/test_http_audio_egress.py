"""HTTP egress retains terminal, mixed-payload and foreign-request routing."""

import asyncio
import os
from types import SimpleNamespace as NS

import pytest
import torch
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

vllm.platforms._current_platform = CpuPlatform()
from easymagpie_vllm_omni import http_audio_egress as gate
from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls
from starlette.requests import Request
from starlette.responses import StreamingResponse
from vllm.outputs import CompletionOutput, RequestOutput
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech


@pytest.fixture(autouse=True)
def installed(monkeypatch, request):
    originals = (Orchestrator._route_output, OmniOpenAIServingSpeech.create_speech, AsyncOmniEngine.shutdown)
    monkeypatch.setenv('EASYMAGPIE_HTTP_AUDIO_EGRESS', '1')
    assert not gate._INSTALLED
    if request.node.name != 'test_opt_in_off_and_invalid_do_not_modify_classes':
        gate.install_http_audio_egress()
    yield
    (Orchestrator._route_output, OmniOpenAIServingSpeech.create_speech, AsyncOmniEngine.shutdown) = originals
    gate._INSTALLED = False
    assert gate._HTTP.get() is None and gate._ROUTE.get() is None
    assert not torch.cuda.is_initialized()


def output(rid='r', finished=False, tokens=(11,), **attributes):
    completion = CompletionOutput(index=0, text='', token_ids=list(tokens), cumulative_logprob=None, logprobs=None)
    result = RequestOutput(
        request_id=rid,
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
    )
    for name, value in attributes.items():
        setattr(result, name, value)
    return result


def owner_state(rid='r', marked=True, queue=None):
    owner = object.__new__(Orchestrator)
    state = OrchestratorRequestState(
        request_id=rid, prompt={}, final_stage_id=1, final_output_stage_ids={1}, sampling_params_list=[None, None]
    )
    if marked:
        state.prompt[gate._PROMPT_KEY] = gate._CAPABILITY
    owner.request_states = {rid: state}
    owner.output_async_queue = queue if queue is not None else asyncio.Queue()
    owner.stage_pools = [
        NS(final_output=True, stage_client=NS(final_output_type='latent')),
        NS(final_output=True, stage_client=NS(final_output_type='audio')),
    ]
    owner._cfg_tracker = NS(
        is_companion=lambda rid: False, has_companions=lambda rid: False, cleanup_parent=lambda rid: []
    )
    owner._pd_pair = None
    owner._is_duplex_session_request = lambda state: False
    owner._duplex_output_decision = lambda *args: None
    owner.async_chunk = True
    owner._stage_receives_async_chunks = lambda stage: True
    owner.cleaned, owner.forwarded = [], []

    async def cleanup(ids, **kwargs):
        owner.cleaned.append((list(ids), kwargs))
        for item in ids:
            owner.request_states.pop(item, None)

    async def forward(*args, **kwargs):
        owner.forwarded.append((args, kwargs))

    owner._cleanup_request_ids, owner._forward_to_next_stage = cleanup, forward
    return owner, state


async def route(owner, state, value, stage=0, metrics=None):
    return await owner._route_output(stage, 0, value, state, metrics)


def test_http_delta_removed_codec_terminal_and_metrics_kept():
    async def check():
        owner, state = owner_state()
        delta = output()
        delta.outputs[0].multimodal_output = {'codes': {'audio': torch.ones(1, 16)}}
        await route(owner, state, delta)
        assert owner.output_async_queue.empty() and not owner.cleaned
        metrics = NS(num_tokens_out=12)
        talker_end = output(finished=True)
        await route(owner, state, talker_end, metrics=metrics)
        first = owner.output_async_queue.get_nowait()
        assert first.engine_outputs is talker_end and first.metrics is metrics and not first.finished
        assert not owner.cleaned and state.finished_final_output_stage_ids == {0}
        pcm = output(tokens=())
        pcm.outputs[0].multimodal_output = {'audio': torch.tensor([0.1, 0.2]), 'sr': 22050}
        await route(owner, state, pcm, stage=1)
        assert owner.output_async_queue.get_nowait().engine_outputs is pcm
        end = output(finished=True, tokens=())
        await route(owner, state, end, stage=1, metrics=metrics)
        assert owner.output_async_queue.get_nowait().finished
        assert owner.cleaned == [(['r'], {})] and not owner.request_states

    asyncio.run(check())


@pytest.mark.parametrize(
    'case',
    [
        'websocket',
        'resumable',
        'mixed_modalities',
        'duplex',
        'forged_boolean',
        'wrong_stage_type',
        'id_reused',
        'finished_state',
    ],
)
def test_scope_guards_keep_output(case):
    async def check():
        owner, state = owner_state(marked=case != 'websocket')
        if case == 'resumable':
            state.streaming.enabled = True
        if case == 'mixed_modalities':
            state.final_output_stage_ids = {0, 1}
        if case == 'duplex':
            owner._is_duplex_session_request = lambda state: True
        if case == 'forged_boolean':
            state.prompt[gate._PROMPT_KEY] = True
        if case == 'wrong_stage_type':
            owner.stage_pools[0].stage_client.final_output_type = 'audio'
        if case == 'id_reused':
            owner.request_states['r'] = OrchestratorRequestState(request_id='r')
        if case == 'finished_state':
            owner.request_states.clear()
        value = output()
        await route(owner, state, value)
        assert owner.output_async_queue.get_nowait().engine_outputs is value

    asyncio.run(check())


@pytest.mark.parametrize(
    'case',
    [
        'empty',
        'metrics',
        'finished',
        'stopped',
        'aborted',
        'cancelled',
        'segment',
        'events',
        'kv_transfer',
        'finish_reason',
        'stop_reason',
        'completion_error',
        'mixed_audio',
        'unknown_metadata',
        'nested_control',
        'nested_model_audio',
        'nested_codes_control',
    ],
)
def test_terminal_metadata_and_mixed_payloads_preserved(case):
    async def check():
        owner, state = owner_state()
        value = output(tokens=() if case == 'empty' else (11,), finished=case == 'finished')
        metrics = NS(num_tokens_out=1) if case == 'metrics' else None
        if case in ('stopped', 'aborted', 'cancelled'):
            setattr(value, case, True)
        if case == 'segment':
            value.is_segment_finished = True
        if case == 'events':
            value.events = ['stage_event']
        if case == 'kv_transfer':
            value.kv_transfer_params = {'key': 'value'}
        if case == 'finish_reason':
            value.outputs[0].finish_reason = 'stop'
        if case == 'stop_reason':
            value.outputs[0].stop_reason = 0
        if case == 'completion_error':
            value.outputs[0].error = 'error'
        if case == 'mixed_audio':
            value.outputs[0].multimodal_output = {'codes.audio': torch.ones(1), 'audio': torch.ones(1)}
        if case == 'unknown_metadata':
            value.outputs[0].multimodal_output = {'codes.audio': torch.ones(1), 'event': 'value'}
        if case == 'nested_control':
            value.outputs[0].multimodal_output = {'codes': {'audio': torch.ones(1), 'event': 'value'}}
        if case == 'nested_model_audio':
            value.outputs[0].multimodal_output = {'model_outputs': {'audio': torch.ones(1), 'events': ['required']}}
        if case == 'nested_codes_control':
            value.outputs[0].multimodal_output = {'codes': {'audio': {'events': ['required']}}}
        await route(owner, state, value, metrics=metrics)
        assert owner.output_async_queue.get_nowait().engine_outputs is value

    asyncio.run(check())


def test_terminal_downstream_transfer_preserved():
    async def check():
        owner, state = owner_state()
        owner.async_chunk = False
        value = output(finished=True)
        await route(owner, state, value)
        assert len(owner.forwarded) == 1
        assert owner.forwarded[0][0] == ('r', 0, value, state)
        assert owner.output_async_queue.get_nowait().engine_outputs is value
        assert not owner.cleaned

    asyncio.run(check())


def test_actual_processed_output_errors_cleanup_and_late_output():
    async def check():
        owner, state = owner_state()
        await route(owner, state, output())  # install queue proxy
        value = output(error='controlled error', error_status_code=503)
        await owner._handle_processed_outputs(0, 0, [value])
        message = owner.output_async_queue.get_nowait()
        assert isinstance(message, ErrorMessage) and message.error == 'controlled error'
        assert message.status_code == 503
        assert owner.cleaned == [(['r'], {'abort': True, 'close_duplex_sessions': True})]
        await owner._handle_processed_outputs(0, 0, [output()])
        assert owner.output_async_queue.empty()

    asyncio.run(check())


def test_out_of_order_codec_terminal_then_talker_delta_is_dropped_by_original_router():
    async def check():
        owner, state = owner_state()
        await route(owner, state, output(finished=True), stage=1)
        assert owner.output_async_queue.get_nowait().finished
        await owner._handle_processed_outputs(0, 0, [output()])
        assert owner.output_async_queue.empty() and len(owner.cleaned) == 1

    asyncio.run(check())


def test_concurrent_blocked_put_cancel_scope_and_queue_methods():
    class BlockingQueue:
        def __init__(self):
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.items = []

        async def put(self, value):
            self.entered.set()
            await self.release.wait()
            self.items.append(value)
            return 'delegated'

        def put_nowait(self, value):
            self.items.append(value)
            return 'nowait'

    async def check():
        queue = BlockingQueue()
        owner, state = owner_state(queue=queue)
        blocked = asyncio.create_task(route(owner, state, output(finished=True)))
        await queue.entered.wait()
        second = OrchestratorRequestState(
            request_id='second',
            prompt={gate._PROMPT_KEY: gate._CAPABILITY},
            final_stage_id=1,
            final_output_stage_ids={1},
        )
        owner.request_states['second'] = second
        await route(owner, second, output('second'))
        assert not queue.items
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        assert gate._ROUTE.get() is None
        queue.release.set()
        late = OutputMessage(request_id='second', stage_id=0, engine_outputs=output('second'), finished=False)
        assert await owner.output_async_queue.put(late) == 'delegated'  # no inherited context
        assert owner.output_async_queue.put_nowait(late) == 'nowait'
        assert queue.items == [late, late] and owner.output_async_queue.items is queue.items
        assert not owner.cleaned

    asyncio.run(check())


def make_serving():
    server = object.__new__(OmniOpenAIServingSpeech)
    server.engine_client = object()
    server._tts_model_type, server._diffusion_mode = 'easymagpie', False

    async def valid(request):
        return None

    server._check_model = valid
    server._validate_speech_streaming_request = lambda *args, **kwargs: ('pcm', None)
    adapter = object.__new__(_build_adapter_cls())
    adapter.ctx = NS(engine_client=server.engine_client)
    adapter._text_stream_metadata = lambda: (99, 4)
    adapter._model_tokenizer = lambda: NS(encode=lambda *args, **kwargs: [1, 2, 3])
    adapter._prompt_len = lambda *args: 67
    server.prepared = []

    async def prepare(request, **kwargs):
        prepared = await adapter.build(request, [], False)
        server.prepared.append(prepared.prompt)

        async def empty():
            if False:
                yield None

        return None, empty(), {}

    server._prepare_speech_generation = prepare
    return server, adapter


@pytest.mark.parametrize(
    'transport,path,fmt,model,marked',
    [
        ('http', '/v1/audio/speech', 'pcm', 'easymagpie', True),
        ('http', '/v1/audio/speech', 'wav', 'easymagpie', False),
        ('http', '/other', 'pcm', 'easymagpie', False),
        ('http', '/v1/audio/speech', 'pcm', 'other', False),
    ],
)
def test_real_create_speech_and_adapter_build_provenance(transport, path, fmt, model, marked):
    async def check():
        server, adapter = make_serving()
        server._tts_model_type = model
        req = OpenAICreateSpeechRequest(
            input='test',
            voice='eng',
            response_format=fmt,
            stream_format='audio',
            extra_params={gate._PROMPT_KEY: True},
        )
        raw = Request({'type': transport, 'method': 'POST', 'path': path, 'headers': []})
        result = await server.create_speech(req, raw)
        assert isinstance(result, StreamingResponse)
        prompt = server.prepared[0]
        assert (prompt.get(gate._PROMPT_KEY) is gate._CAPABILITY) == marked
        assert gate._HTTP.get() is None
        standalone = await adapter.build(req, [], False)
        assert gate._PROMPT_KEY not in standalone.prompt

    asyncio.run(check())


def test_http_cancel_resets_context_and_cannot_mark_other_identity():
    async def check():
        server, _ = make_serving()
        req = OpenAICreateSpeechRequest(input='test', voice='eng', response_format='pcm', stream_format='audio')
        raw = Request({'type': 'http', 'method': 'POST', 'path': '/v1/audio/speech', 'headers': []})
        entered = asyncio.Event()

        async def prepare(request, **kwargs):
            assert gate.mark_legacy_http_audio_prompt({}, object(), server.engine_client) == {}
            assert gate.mark_legacy_http_audio_prompt({}, request, object()) == {}
            entered.set()
            await asyncio.Event().wait()

        server._prepare_speech_generation = prepare
        task = asyncio.create_task(server.create_speech(req, raw))
        await entered.wait()
        task.cancel()
        result = await task
        assert result.error.message == 'Client disconnected'  # original create_speech behavior
        assert gate._HTTP.get() is None

    asyncio.run(check())


def test_opt_in_off_and_invalid_do_not_modify_classes(monkeypatch):
    original = Orchestrator._route_output
    monkeypatch.setenv('EASYMAGPIE_HTTP_AUDIO_EGRESS', '0')
    gate.install_http_audio_egress()
    assert Orchestrator._route_output is original
    assert not gate._INSTALLED
    monkeypatch.setenv('EASYMAGPIE_HTTP_AUDIO_EGRESS', 'invalid')
    with pytest.raises(ValueError):
        gate.install_http_audio_egress()
    assert Orchestrator._route_output is original


def test_marker_survives_actual_janus_request_handoff_and_is_not_preprocessed_wire_input():
    import threading

    import janus
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm_omni.engine.messages import StageSubmissionMessage

    async def check():
        prompt = {
            'prompt_token_ids': [1, 2, 3],
            gate._PROMPT_KEY: gate._CAPABILITY,
            'additional_information': {'speaker_id': 'eng'},
        }
        processor = object.__new__(InputPreprocessor)
        processor._truncate_inputs = lambda ids, kwargs: ids
        processed = processor._prompt_to_llm_inputs(prompt)
        assert gate._PROMPT_KEY not in processed
        assert processed['prompt_token_ids'] == [1, 2, 3]
        msg = StageSubmissionMessage(
            type='add_request',
            request_id='r',
            prompt=processed,
            original_prompt=prompt,
            output_prompt_text=None,
            sampling_params_list=[None, None],
            final_stage_id=1,
            final_output_stage_ids=[1],
            preprocess_ms=0,
            request_timestamp=0,
            enqueue_ts=0,
        )
        queue = janus.Queue(maxsize=256)
        thread = threading.Thread(target=lambda: queue.sync_q.put(msg))
        thread.start()
        received = await queue.async_q.get()
        thread.join(timeout=2)
        assert not thread.is_alive() and received is msg and received.original_prompt is prompt
        owner, _ = owner_state(marked=False)
        owner.request_states.clear()
        owner._register_running_request = lambda state: None
        submissions, prewarms = [], []

        async def submit(*args, **kwargs):
            submissions.append((args, kwargs))

        async def prewarm(*args, **kwargs):
            prewarms.append((args, kwargs))

        owner.stage_pools[0].submit_initial = submit
        owner._prewarm_async_chunk_stages = prewarm
        await owner._handle_add_request(received)
        state = owner.request_states['r']
        assert state.prompt is prompt and gate._eligible(owner, state, 0)
        assert len(submissions) == len(prewarms) == 1
        await route(owner, state, output())
        assert owner.output_async_queue.empty()
        queue.close()
        await queue.wait_closed()

    asyncio.run(check())


def test_websocket_scope_and_nested_http_request_identity_are_not_marked():
    request = NS(response_format='pcm', is_raw_audio_stream=lambda: True)
    serving = NS(_tts_model_type='easymagpie', engine_client=object())
    assert not gate._legacy_pcm_request(serving, request, NS(scope={'type': 'websocket', 'path': '/v1/audio/speech'}))
    token = gate._HTTP.set((request, serving.engine_client))
    try:
        assert gate.mark_legacy_http_audio_prompt({}, NS(), serving.engine_client) == {}
        assert gate.mark_legacy_http_audio_prompt({}, request, object()) == {}
    finally:
        gate._HTTP.reset(token)
