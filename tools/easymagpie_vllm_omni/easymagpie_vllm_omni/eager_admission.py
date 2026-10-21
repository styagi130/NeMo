"""Opt-in HTTP PCM admission before the StreamingResponse task handoff.

The pinned AsyncOmni generator remains unchanged. One eager task per eligible
HTTP request advances it through its existing enqueue to its first suspension,
and holds at most one output. The ASGI request owns that task and closes it on
every exit, including cancellation before the response body is ever entered.
"""

import asyncio
import functools
import inspect
import os
from argparse import Namespace
from collections import Counter
from contextvars import ContextVar

_OWNER = ContextVar('easymagpie_eager_admission_owner', default=None)
_COUNTS = Counter(asgi_apps=0, started=0, active=0, peak_active=0, prepared=0, closed=0)
_INSTALLED = False
_APP_PID = None
_SHUTDOWN_PID = None
_PATCHES = {}


class EagerOutputLease:
    """Single-consumer iterator; one prefetched output, no independent pump."""

    def __init__(self, source):
        self.source = source
        self.first = None
        self.closed = False
        self.consuming = False
        self.loop = asyncio.get_running_loop()

    def start(self):
        if self.first is not None or self.closed:
            raise RuntimeError('eager output lease already started or closed')
        _COUNTS['started'] += 1
        _COUNTS['active'] += 1
        _COUNTS['peak_active'] = max(_COUNTS['peak_active'], _COUNTS['active'])
        # Python 3.12 Task eagerly runs the ORIGINAL generator only until its
        # first actual suspension. In the unpaused pinned path, enqueue has
        # completed before the subsequent per-request output-queue wait.
        self.first = asyncio.Task(
            self._first_output(), loop=self.loop, name='easymagpie-eager-admission', eager_start=True
        )

    async def _first_output(self):
        return await anext(self.source)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        if self.consuming:
            raise RuntimeError('eager output lease requires exactly one consumer')
        self.consuming = True
        try:
            if self.first is not None:
                task, self.first = self.first, None
                return await task
            return await anext(self.source)
        finally:
            self.consuming = False

    async def aclose(self):
        if self.closed:
            return
        if self.consuming:
            # ASGI middleware runs after its application's tasks have drained.
            # A violation must not silently close/reuse a live generator.
            raise RuntimeError('ASGI exited while its eager iterator is still consuming')
        self.closed = True
        try:
            task, self.first = self.first, None
            if task is not None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    # Retrieve a never-consumed output/error. The original
                    # generator owns its exception/abort handling.
                    pass
            await self.source.aclose()
        finally:
            self.source = None
            _COUNTS['active'] -= 1
            _COUNTS['closed'] += 1


class _RequestOwner:
    def __init__(self):
        self.lease = None

    def start(self, source):
        if self.lease is not None:
            raise RuntimeError('one eager speech generation is allowed per HTTP request')
        if not inspect.isasyncgen(source):
            raise TypeError('eager admission requires the original async generator')
        self.lease = EagerOutputLease(source)
        self.lease.start()
        return self.lease

    async def close(self):
        if self.lease is not None:
            await self.lease.aclose()


class EagerAdmissionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/v1/audio/speech':
            return await self.app(scope, receive, send)
        owner = _RequestOwner()
        token = _OWNER.set(owner)
        try:
            return await self.app(scope, receive, send)
        finally:
            try:
                await owner.close()
            finally:
                _OWNER.reset(token)


def eager_admission_status():
    wrappers = {
        name: getattr(owner, attribute) is installed for name, (owner, attribute, installed) in _PATCHES.items()
    }
    return {
        'schema_version': 3,
        'enabled': _INSTALLED,
        'pid': os.getpid(),
        'api_process': _APP_PID == os.getpid(),
        'counts': dict(_COUNTS),
        'drained': _COUNTS['started'] == _COUNTS['closed'] and _COUNTS['active'] == 0,
        'wrappers': wrappers,
        'scope': 'HTTP POST /v1/audio/speech, EasyMagpie raw PCM only',
        'prefetch_bound': 'one output and one eager task per eligible active HTTP request',
        'ownership': 'outer ASGI lifetime; original generator retains internal ID and abort semantics',
    }


def install_eager_admission():
    global _INSTALLED
    if os.environ.get('EASYMAGPIE_EAGER_HTTP_ADMISSION', '0') != '1':
        return False
    if _INSTALLED:
        return True
    if 'eager_start' not in inspect.signature(asyncio.Task).parameters:
        raise RuntimeError('eager admission requires the pinned Python 3.12 Task API')
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine as engine
    from vllm_omni.entrypoints.async_omni import AsyncOmni as async_omni
    from vllm_omni.entrypoints.openai import api_server as api
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech as speech
    from vllm_omni.utils.tracking_parser import TrackingNamespace

    original_prepare = speech._prepare_speech_generation
    original_build = api.build_openai_app
    original_shutdown = engine.shutdown

    @functools.wraps(original_prepare)
    async def prepared(self, request, *args, **kwargs):
        result = await original_prepare(self, request, *args, **kwargs)
        owner = _OWNER.get()
        eligible = (
            owner is not None
            and self._tts_model_type == 'easymagpie'
            and request.is_raw_audio_stream()
            and request.response_format == 'pcm'
        )
        if not eligible:
            return result
        if not isinstance(self.engine_client, async_omni):
            raise TypeError('eager admission requires the pinned AsyncOmni client')
        request_id, generator, params = result
        lease = owner.start(generator)
        _COUNTS['prepared'] += 1
        return request_id, lease, params

    @functools.wraps(original_build)
    def build(args, *positional, **kwargs):
        global _APP_PID
        # The pinned factory's SageMaker bootstrap constructs middleware_stack
        # before it returns. Register through its existing middleware argument
        # hook, rather than mutating an already-started FastAPI application.
        # TrackingNamespace forwards __dict__/__getattr__ to its backing
        # namespace, so copy.copy recurses before that backing field exists.
        # Reconstruct the two pinned namespace types explicitly and preserve
        # the tracking wrapper's immutable explicit-argument provenance.
        if isinstance(args, TrackingNamespace):
            configured = TrackingNamespace(Namespace(**vars(args.unfiltered_ns)), args.explicit_keys)
        elif type(args) is Namespace:
            configured = Namespace(**vars(args))
        else:
            raise TypeError('unsupported eager admission app-factory namespace')
        configured.middleware = list(args.middleware or ())
        configured.middleware.append(__name__ + '.EagerAdmissionMiddleware')
        app = original_build(configured, *positional, **kwargs)
        if not any(item.cls is EagerAdmissionMiddleware for item in app.user_middleware):
            raise RuntimeError('eager HTTP admission middleware was not installed')
        _COUNTS['asgi_apps'] += 1
        _APP_PID = os.getpid()
        return app

    @functools.wraps(original_shutdown)
    def shutdown_engine(self):
        global _SHUTDOWN_PID
        try:
            return original_shutdown(self)
        finally:
            # Check once in the API process, after the original engine cleanup.
            pid = os.getpid()
            if _APP_PID == pid and _SHUTDOWN_PID != pid:
                _SHUTDOWN_PID = pid
                if not eager_admission_status()['drained']:
                    raise RuntimeError('eager admission did not drain before engine shutdown')

    speech._prepare_speech_generation = prepared
    api.build_openai_app = build
    engine.shutdown = shutdown_engine
    _PATCHES.update(
        prepare=(speech, '_prepare_speech_generation', prepared), app_factory=(api, 'build_openai_app', build)
    )
    _INSTALLED = True
    return True
