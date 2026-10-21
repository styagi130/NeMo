"""CPU lifecycle gate using pinned AsyncOmni.generate and StreamingResponse."""

import asyncio
from types import SimpleNamespace as NS

import torch
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

vllm.platforms._current_platform = CpuPlatform()
import numpy as np
from easymagpie_vllm_omni import eager_admission as eager
from vllm import SamplingParams
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.async_omni import AsyncOmni, OutputMessage
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai import serving_speech as speech
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest

RESULTS = []
SCOPE = {
    'type': 'http',
    'method': 'POST',
    'path': '/v1/audio/speech',
    'asgi': {'version': '3.0', 'spec_version': '2.4'},
    'headers': [],
}


class FakeEngine:
    add_request_async = AsyncOmniEngine.add_request_async
    abort_async = AsyncOmniEngine.abort_async
    num_stages = 2
    stage_clients = []
    _running_counter = None

    def __init__(self, mode='ready'):
        self.mode, self.added, self.aborted = mode, [], []

    def is_alive(self):
        return True

    def add_request(self, **kwargs):
        self.added.append(kwargs)
        if self.mode == 'error':
            raise ValueError('controlled admission error')
        if self.mode == 'ready':
            self.emit(kwargs['request_id'])

    def abort(self, request_ids):
        self.aborted.extend(request_ids)

    def emit(self, rid, values=(1, 2, 3)):
        for index, value in enumerate(values):
            self.owner.request_states[rid].queue.put_nowait(
                OutputMessage(
                    request_id=rid,
                    stage_id=1,
                    finished=index == len(values) - 1,
                    engine_outputs=NS(request_id=rid, payload=value),
                )
            )


def make_client(mode='ready'):
    client = object.__new__(AsyncOmni)
    client.engine = FakeEngine(mode)
    client.engine.owner = client
    client._pause_cond, client._paused = asyncio.Condition(), False
    client._sleeping_tags = set()
    client.log_stats, client.request_states = False, {}
    client.default_sampling_params_list = [SamplingParams(), SamplingParams()]
    client._final_output_handler = lambda: None
    client._maybe_expand_sampling_params = lambda values: values
    client.resolve_sampling_params_list = lambda values, **kwargs: values
    client._compute_final_stage_id = lambda modalities: 1
    client._compute_final_output_stage_ids = lambda modalities: [1]
    client._get_pd_separation_pair = lambda: None
    client._check_engine_output_error = lambda *args: None
    client.observed_outputs = []

    def output(result, *args):
        client.observed_outputs.append(result.engine_outputs.payload)
        return result.engine_outputs

    client._process_single_result = output
    client.prom_metrics = NS(
        request_failed=lambda: None, set_running=lambda value: None, set_waiting=lambda value: None
    )
    return client


class Adapter:
    def validate(self, request):
        return None

    async def build(self, request, sampling, inline):
        return PreparedRequest(prompt={'prompt_token_ids': [1, 2, 3]}, tts_params={}, model_type='easymagpie')


def make_serving(client):
    server = object.__new__(speech.OmniOpenAIServingSpeech)
    server.engine_client = client
    server.model_config = NS(async_chunk=True)
    server._tts_model_type = 'easymagpie'
    server._is_tts, server._diffusion_mode = True, False
    server._get_tts_adapter = lambda: Adapter()
    server._track_ref_audio_artifact_warmup = lambda *args, **kwargs: None
    server._mark_ref_audio_artifact_ready_for_request = lambda *args: None
    server._discard_ref_audio_artifact_warmup = lambda *args: None
    server._validate_speech_streaming_request = lambda *args, **kwargs: ('pcm', None)

    async def valid(request):
        return None

    server._check_model = valid
    server._extract_audio_output = lambda output: (
        {'audio': np.array([output.payload], dtype=np.float32), 'sr': 22050},
        'audio',
    )
    server.create_audio = lambda audio: NS(audio_data=bytes([int(audio.audio_tensor[0])]))
    return server


def request():
    return OpenAICreateSpeechRequest(
        model='easymagpie', input='lifecycle fixture', voice='eng', response_format='pcm', stream_format='audio'
    )


async def no_receive():
    await asyncio.Future()


def app_for(server, *, before_body=None, send_hook=None):
    sent = []

    async def send(message):
        if send_hook:
            await send_hook(message)
        if message.get('body'):
            sent.append(message['body'])

    async def app(scope, receive, outgoing):
        response = await server.create_speech(request())
        if before_body:
            await before_body(response)
        await response(scope, receive, outgoing)

    return app, sent, send


async def check_case(loop_name, name, function):
    start_active = eager.eager_admission_status()['counts'].get('active', 0)
    await asyncio.wait_for(function(), timeout=10)
    await asyncio.sleep(0)
    assert eager.eager_admission_status()['counts'].get('active', 0) == start_active == 0
    assert not [t for t in asyncio.all_tasks() if t.get_name() == 'easymagpie-eager-admission' and not t.done()]
    RESULTS.append({'loop': loop_name, 'case': name, 'status': 'PASS'})


async def suite(loop_name):
    async def control_lazy():
        client, observations = make_client(), []
        server = make_serving(client)

        async def before(response):
            assert not client.engine.added and not client.request_states
            observations.append('lazy')

        app, sent, send = app_for(server, before_body=before)
        await app(SCOPE, no_receive, send)  # no middleware owner: original behavior
        assert observations == ['lazy'] and sent == [b'\x01', b'\x02', b'\x03']
        assert len(client.engine.added) == 1 and not client.engine.aborted and not client.request_states

    await check_case(loop_name, 'original generator remains lazy without HTTP owner', control_lazy)

    async def actual_factory_request(app):
        global _factory_server
        client = make_client()
        _factory_server = make_serving(client)
        sent = []
        prepared = eager._COUNTS['prepared']

        async def send(message):
            if message.get('body'):
                sent.append(message['body'])

        scope = dict(
            SCOPE,
            http_version='1.1',
            scheme='http',
            root_path='',
            query_string=b'',
            raw_path=b'/v1/audio/speech',
            server=('localhost', 80),
            client=('127.0.0.1', 12345),
        )
        await app(scope, no_receive, send)
        assert sent == [b'\x01', b'\x02', b'\x03'], sent
        assert eager._COUNTS['prepared'] == prepared + 1
        assert not client.request_states and not client.engine.aborted

    for factory_name, factory_app in _apps:

        async def factory_case(app=factory_app):
            await actual_factory_request(app)

        await check_case(loop_name, 'actual FastAPI factory serves and drains ' + factory_name, factory_case)

    async def eager_all_terminal():
        client, observations = make_client(), []
        server = make_serving(client)

        async def before(response):
            assert len(client.engine.added) == 1 and len(client.request_states) == 1
            assert client.observed_outputs == [1], 'more than one engine output prefetched'
            observations.append('admitted-before-body')

        app, sent, send = app_for(server, before_body=before)
        await eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send)
        assert observations == ['admitted-before-body'] and sent == [b'\x01', b'\x02', b'\x03']
        assert client.observed_outputs == [1, 2, 3]
        assert not client.request_states and not client.engine.aborted
        assert client.engine.added[0]['prompt'] == {'prompt_token_ids': [1, 2, 3]}
        assert client.engine.added[0]['final_output_stage_ids'] == [1]

    await check_case(
        loop_name, 'eager enqueue before handoff, one-output bound, ordered complete drain', eager_all_terminal
    )

    async def cancel_before_body():
        client = make_client('held')
        prepared = asyncio.Event()

        async def before(response):
            assert len(client.engine.added) == 1
            prepared.set()
            await asyncio.Future()

        app, sent, send = app_for(make_serving(client), before_body=before)
        task = asyncio.create_task(eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send))
        await prepared.wait()
        rid = client.engine.added[0]['request_id']
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not sent and not client.request_states and client.engine.aborted == [rid]

    await check_case(loop_name, 'cancel after admission before response body ever starts', cancel_before_body)

    async def cancel_before_first_audio():
        client = make_client('held')
        headers = asyncio.Event()

        async def send_hook(message):
            if message['type'] == 'http.response.start':
                headers.set()

        app, sent, send = app_for(make_serving(client), send_hook=send_hook)
        task = asyncio.create_task(eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send))
        await headers.wait()
        rid = client.engine.added[0]['request_id']
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not sent and not client.request_states and client.engine.aborted == [rid]

    await check_case(loop_name, 'cancel pending first-audio consumer', cancel_before_first_audio)

    async def disconnect_on_first_audio():
        client = make_client()
        seen = []

        async def send_hook(message):
            if message.get('body'):
                seen.append(message['body'])
                raise OSError('controlled disconnected send')

        app, sent, send = app_for(make_serving(client), send_hook=send_hook)
        try:
            await eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send)
        except Exception as exc:
            assert type(exc).__name__ == 'ClientDisconnect'
        else:
            raise AssertionError('send failure swallowed')
        assert seen == [b'\x01'] and not client.request_states
        assert client.engine.aborted == [client.engine.added[0]['request_id']]

    await check_case(loop_name, 'disconnect while first audio is sent', disconnect_on_first_audio)

    async def abandoned_ready_response():
        client = make_client()

        async def before(response):
            assert client.observed_outputs == [1]
            raise RuntimeError('controlled failure before response invocation')

        app, sent, send = app_for(make_serving(client), before_body=before)
        try:
            await eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send)
        except RuntimeError:
            pass
        assert not sent and not client.request_states
        assert client.engine.aborted == [client.engine.added[0]['request_id']]

    await check_case(loop_name, 'abandoned response with already-prefetched output', abandoned_ready_response)

    async def admission_error():
        client = make_client('error')
        app, sent, send = app_for(make_serving(client))
        try:
            await eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send)
        except ValueError as exc:
            assert str(exc) == 'controlled admission error'
        else:
            raise AssertionError('original admission exception swallowed')
        assert not sent and not client.request_states
        assert client.engine.aborted == [client.engine.added[0]['request_id']]

    await check_case(loop_name, 'original admission exception propagated and state cleaned', admission_error)

    async def reused_external_id():
        client = make_client('held')
        server = make_serving(client)
        original_uuid = speech.random_uuid
        speech.random_uuid = lambda: 'same-external-id'
        entered = [asyncio.Event(), asyncio.Event()]
        tasks = []
        try:
            for index in range(2):

                async def before(response, index=index):
                    entered[index].set()

                app, sent, send = app_for(server, before_body=before)
                tasks.append((asyncio.create_task(eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send)), sent))
                await entered[index].wait()
            left, right = [r['request_id'] for r in client.engine.added]
            assert left != right and all(r.startswith('speech-same-external-id') for r in (left, right))
            tasks[0][0].cancel()
            try:
                await tasks[0][0]
            except asyncio.CancelledError:
                pass
            assert client.engine.aborted == [left] and set(client.request_states) == {right}
            client.engine.emit(right)
            await tasks[1][0]
            assert tasks[1][1] == [b'\x01', b'\x02', b'\x03'] and not client.request_states
        finally:
            speech.random_uuid = original_uuid
            for task, _ in tasks:
                if not task.done():
                    task.cancel()

    await check_case(
        loop_name, 'same external ID retains distinct internal cancellation ownership', reused_external_id
    )

    async def paused_cancel():
        client = make_client('held')
        client._paused = True
        prepared = asyncio.Event()

        async def before(response):
            assert not client.engine.added and not client.request_states
            prepared.set()
            await asyncio.Future()

        app, sent, send = app_for(make_serving(client), before_body=before)
        task = asyncio.create_task(eager.EagerAdmissionMiddleware(app)(SCOPE, no_receive, send))
        await prepared.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not client.engine.added and not client.engine.aborted and not client.request_states

    await check_case(loop_name, 'paused engine remains paused; cancellation creates no request', paused_cancel)

    async def asgi_disconnect_scope():
        client = make_client('held')
        scope = dict(SCOPE, asgi={'version': '3.0', 'spec_version': '2.0'})

        async def receive():
            return {'type': 'http.disconnect'}

        app, sent, send = app_for(make_serving(client))
        await eager.EagerAdmissionMiddleware(app)(scope, receive, send)
        assert not sent and not client.request_states
        assert client.engine.aborted == [client.engine.added[0]['request_id']]

    await check_case(loop_name, 'Starlette ASGI2 disconnect task-group cleanup', asgi_disconnect_scope)


assert not torch.cuda.is_initialized()


def test_actual_async_generator_and_asgi_lifetimes(monkeypatch):
    global _apps, _factory_server
    monkeypatch.setenv("EASYMAGPIE_EAGER_HTTP_ADMISSION", "1")
    for owner, name in (
        (speech.OmniOpenAIServingSpeech, "_prepare_speech_generation"),
        (api_server, "build_openai_app"),
        (AsyncOmniEngine, "shutdown"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    eager._COUNTS.clear()
    eager._INSTALLED = False
    try:
        assert eager.install_eager_admission() is True
        assert eager.install_eager_admission() is True
        # Exercise the pinned upstream app factory, not a fake app.add_middleware.
        from argparse import Namespace

        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

        _factory_server = None

        async def _factory_endpoint():
            return await _factory_server.create_speech(request())

        _apps = []
        for name, parser_type, argv in (
            ('plain', FlexibleArgumentParser, []),
            ('tracking-defaults', TrackingArgumentParser, []),
            (
                'tracking-explicit-middleware',
                TrackingArgumentParser,
                ['--middleware', 'starlette.middleware.gzip.GZipMiddleware'],
            ),
        ):
            args = make_arg_parser(parser_type()).parse_args(argv)
            assert isinstance(args, TrackingNamespace) == (parser_type is TrackingArgumentParser)
            before_values = dict(vars(args))
            before_middleware = args.middleware
            before_explicit = args.get_explicit_kwargs_dict() if isinstance(args, TrackingNamespace) else None
            app = api_server.build_openai_app(args, ('generate',))
            assert vars(args) == before_values and args.middleware is before_middleware
            assert app.state.args is not args and app.state.args.middleware is not args.middleware
            assert app.state.args.middleware == list(args.middleware) + [
                'easymagpie_vllm_omni.eager_admission.EagerAdmissionMiddleware'
            ]
            if isinstance(args, TrackingNamespace):
                assert isinstance(app.state.args, TrackingNamespace)
                assert app.state.args.unfiltered_ns is not args.unfiltered_ns
                assert app.state.args.explicit_keys is args.explicit_keys
                assert args.get_explicit_kwargs_dict() == before_explicit
            assert any(item.cls is eager.EagerAdmissionMiddleware for item in app.user_middleware)
            assert app.middleware_stack is not None
            app.add_api_route('/v1/audio/speech', _factory_endpoint, methods=['POST'])
            _apps.append((name, app))
        assert eager.eager_admission_status()['counts']['asgi_apps'] == 3
        # Even when the original factory rejects an invalid middleware, the caller's
        # tracked namespace and original middleware list remain untouched.
        bad_args = make_arg_parser(TrackingArgumentParser()).parse_args([])
        bad_args.middleware = ['module_that_does_not_exist.NoMiddleware']
        bad_list = bad_args.middleware
        try:
            api_server.build_openai_app(bad_args, ('generate',))
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError('invalid middleware unexpectedly succeeded')
        assert bad_args.middleware is bad_list and len(bad_args.middleware) == 1

        async def check_scope_passthrough():
            seen = []

            async def inner(scope, receive, send):
                seen.append(eager._OWNER.get())

            for scope in (
                {'type': 'websocket', 'path': '/v1/audio/speech'},
                {'type': 'http', 'method': 'GET', 'path': '/v1/audio/speech'},
                {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions'},
            ):
                await eager.EagerAdmissionMiddleware(inner)(scope, no_receive, None)
            assert seen == [None, None, None]

        asyncio.run(check_scope_passthrough())
        asyncio.run(suite('asyncio'))
        import uvloop

        uvloop.run(suite('uvloop'))
        assert not torch.cuda.is_initialized()
        assert eager.eager_admission_status()['counts']['active'] == 0
    finally:
        eager._INSTALLED = False
        eager._PATCHES.clear()
        eager._APP_PID = None
        eager._SHUTDOWN_PID = None
        eager._COUNTS.clear()
