"""Opt-in removal of unused HTTP talker deltas at orchestrator egress.

The original routing method still executes in full. Only one specifically
identified queue put may be omitted; terminal, error, metrics, transfer and
cleanup operations keep their original implementation and ordering.
"""

import functools
import os
import weakref
from collections import Counter
from collections.abc import Mapping
from contextvars import ContextVar

_ENV = 'EASYMAGPIE_HTTP_AUDIO_EGRESS'
_PROMPT_KEY = '_easymagpie_http_audio_egress_capability'
_CAPABILITY = object()
_HTTP = ContextVar('easymagpie_legacy_http_audio', default=None)
_ROUTE = ContextVar('easymagpie_http_audio_route', default=None)
_INSTALLED = False
_TENSOR_TYPES = ()
_COUNTS = Counter(
    marked_prompts=0, suppressed_nonterminal_stage0=0, eligible_stage0_puts=0, preserved_terminal_or_control=0
)


def status():
    return {
        'enabled': _INSTALLED,
        'pid': os.getpid(),
        'counters': dict(_COUNTS),
        'scope': 'Explicit raw-PCM HTTP requests only; no per-request registry',
        'terminal_metrics_errors_preserved': True,
    }


def mark_legacy_http_audio_prompt(prompt, request, engine_client):
    """Stamp a server-owned capability; client extras cannot authorize a drop."""
    scope = _HTTP.get()
    if _INSTALLED and scope is not None and scope[0] is request and scope[1] is engine_client:
        if not isinstance(prompt, dict) or _PROMPT_KEY in prompt:
            raise RuntimeError('HTTP egress marker requires a fresh prompt')
        prompt[_PROMPT_KEY] = _CAPABILITY
        _COUNTS['marked_prompts'] += 1
    return prompt


def _legacy_pcm_request(serving, request, raw_request):
    scope = getattr(raw_request, 'scope', None)
    return (
        getattr(serving, '_tts_model_type', None) == 'easymagpie'
        and isinstance(scope, dict)
        and scope.get('type') == 'http'
        and scope.get('path') == '/v1/audio/speech'
        and request.is_raw_audio_stream()
        and getattr(request, 'response_format', None) == 'pcm'
    )


def _eligible(owner, state, stage_id):
    if stage_id != 0 or state is None or state.streaming.enabled:
        return False
    if state.final_stage_id != 1 or state.final_output_stage_ids != {1}:
        return False
    if not isinstance(state.prompt, dict) or state.prompt.get(_PROMPT_KEY) is not _CAPABILITY:
        return False
    if owner.request_states.get(state.request_id) is not state:
        return False
    if owner._is_duplex_session_request(state):
        return False
    pool = owner.stage_pools[0]
    return pool.final_output and getattr(pool.stage_client, 'final_output_type', None) == 'latent'


def _has_control(record):
    # CompletionOutput.finished is a method, whereas RequestOutput.finished is
    # a boolean. Completion finish_reason is checked separately below.
    return (
        any(
            getattr(record, name, False) is True
            for name in ('finished', 'stopped', 'aborted', 'cancelled', 'canceled', 'is_segment_finished')
        )
        or getattr(record, 'error', None) is not None
        or bool(getattr(record, 'events', None))
        or bool(getattr(record, 'kv_transfer_params', None))
    )


def _latent_payload_only(record):
    payload = getattr(record, 'multimodal_output', None)
    if payload is None:
        return True
    if not isinstance(payload, Mapping):
        return False
    # Keep unfamiliar metadata/events and any mixed payload. This pinned
    # two-stage talker publishes codes.audio; other model outputs stay intact.
    if any(key not in ('codes', 'codes.audio') for key in payload):
        return False
    direct = payload.get('codes.audio')
    if direct is not None and not isinstance(direct, _TENSOR_TYPES):
        return False
    codes = payload.get('codes')
    return codes is None or (
        isinstance(codes, Mapping) and set(codes) == {'audio'} and isinstance(codes['audio'], _TENSOR_TYPES)
    )


def _is_nonterminal_delta(message):
    output = message.engine_outputs
    if message.finished or message.metrics is not None or _has_control(output) or not _latent_payload_only(output):
        return False
    completions = getattr(output, 'outputs', None)
    if not completions:
        return False
    for completion in completions:
        if (
            _has_control(completion)
            or not _latent_payload_only(completion)
            or getattr(completion, 'finish_reason', None) is not None
            or getattr(completion, 'stop_reason', None) is not None
            or getattr(completion, 'error', None) is not None
        ):
            return False
    # Empty metadata/control messages are retained even if marked unfinished.
    return any(isinstance(getattr(c, 'token_ids', None), (list, tuple)) and len(c.token_ids) > 0 for c in completions)


class _OutputQueue:
    def __init__(self, queue, owner, output_type):
        self._queue = queue
        self._owner = weakref.ref(owner)
        self._output_type = output_type

    def __getattr__(self, name):
        return getattr(self._queue, name)

    async def put(self, message):
        scope = _ROUTE.get()
        owner = self._owner()
        if (
            scope is not None
            and owner is scope[0]
            and isinstance(message, self._output_type)
            and message.engine_outputs is scope[2]
            and message.request_id == scope[1].request_id
            and message.stage_id == scope[3]
            and _eligible(owner, scope[1], scope[3])
        ):
            _COUNTS['eligible_stage0_puts'] += 1
            if _is_nonterminal_delta(message):
                _COUNTS['suppressed_nonterminal_stage0'] += 1
                return None
            _COUNTS['preserved_terminal_or_control'] += 1
        return await self._queue.put(message)


def install_http_audio_egress():
    global _INSTALLED, _TENSOR_TYPES
    option = os.environ.get(_ENV, '0')
    if option not in ('0', '1'):
        raise ValueError(f'{_ENV} must be 0 or 1')
    if option == '0' or _INSTALLED:
        return status()
    from torch import Tensor
    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    _TENSOR_TYPES = (Tensor,)

    create = OmniOpenAIServingSpeech.create_speech

    @functools.wraps(create)
    async def create_speech(self, request, raw_request=None):
        token = _HTTP.set((request, self.engine_client) if _legacy_pcm_request(self, request, raw_request) else None)
        try:
            return await create(self, request, raw_request)
        finally:
            _HTTP.reset(token)

    route = Orchestrator._route_output

    @functools.wraps(route)
    async def route_output(self, stage_id, replica_id, output, req_state, stage_metrics):
        if not isinstance(self.output_async_queue, _OutputQueue):
            self.output_async_queue = _OutputQueue(self.output_async_queue, self, OutputMessage)
        token = _ROUTE.set((self, req_state, output, stage_id))
        try:
            return await route(self, stage_id, replica_id, output, req_state, stage_metrics)
        finally:
            _ROUTE.reset(token)

    OmniOpenAIServingSpeech.create_speech = create_speech
    Orchestrator._route_output = route_output
    _INSTALLED = True
    return status()
