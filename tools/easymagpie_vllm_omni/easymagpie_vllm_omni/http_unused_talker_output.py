"""Default-off, admitted private HTTP-only stage0 output materialization skip.

All original processor/state/stop/logprob bodies execute. Only an exact fresh
frontend owner may omit a nonterminal MM-empty output after preserving the two
original readout/drain effects. The pool records the same timestamp/nonempty
row. Original terminal, control, mixed-MM and foreign frontend paths remain.
"""

import functools
import os
import threading
import weakref
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType

ENV = 'EASYMAGPIE_HTTP_UNUSED_TALKER_OUTPUTS'
_ADMISSION = ContextVar('easymagpie_unused_http_admission', default=None)
_POOL_CALL = ContextVar('easymagpie_unused_http_pool_call', default=None)
_KEY = '_easymagpie_unused_http_output_owner'
_INSTALLED = False
_CLOSED = False
_ISSUER = object()
_GENERATION = 0
_POOLS = weakref.WeakKeyDictionary()
_COUNTS = Counter()
_BINDINGS = None


def _flag():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(ENV + ' must be0/1')
    return value


@dataclass(frozen=True)
class _Lease:
    issuer: object
    generation: int
    owner_ref: object
    outer_ref: object
    state_ref: object
    pool_state: object
    internal_id: str
    external_id: str
    prompt: dict


class _PoolState:
    def __init__(self, pool, processor, owner):
        self.pool_ref = weakref.ref(pool)
        self.processor_ref = weakref.ref(processor)
        self.owner_ref = weakref.ref(owner)
        self.pid, self.tid = os.getpid(), threading.get_ident()
        self.scope = None
        self.pending = None
        self.closed = False
        self.leases = weakref.WeakValueDictionary()
        self.originals = (
            type(pool).process_llm_raw_outputs,
            type(pool).record_output_timestamps,
            type(processor).process_outputs,
        )
        self.installed = {}


def _release(state, reason):
    lease = getattr(state, _KEY, None)
    if type(lease) is not _Lease or lease.issuer is not _ISSUER:
        return
    state.__dict__.pop(_KEY, None)
    method = state.__dict__.get('make_request_output')
    if getattr(method, '__func__', None) is _make:
        state.__dict__.pop('make_request_output', None)
    ps = lease.pool_state
    if ps.leases.get(lease.internal_id) is lease:
        ps.leases.pop(lease.internal_id, None)
    _COUNTS['revoked_' + reason] += 1


def _eligible_pool(ps):
    pool = ps.pool_ref()
    processor = ps.processor_ref()
    owner = ps.owner_ref()
    return (
        not _CLOSED
        and not ps.closed
        and pool is not None
        and processor is not None
        and owner is not None
        and (os.getpid(), threading.get_ident()) == (ps.pid, ps.tid)
        and owner.stage_pools[0] is pool
        and pool.output_processor is processor
        and owner.duplex_control_plane is None
        and owner._pd_pair is None
        and owner.async_chunk is True
        and owner._stage_receives_async_chunks(1)
        and pool.stage_id == 0
        and pool.final_output
        and getattr(pool.stage_client, 'final_output_type', None) == 'latent'
        and len(owner.stage_pools) == 2
        and owner.stage_pools[1].final_output
        and getattr(owner.stage_pools[1].stage_client, 'final_output_type', None) == 'audio'
        and pool.__dict__.get('process_llm_raw_outputs') is ps.installed['pool']
        and pool.__dict__.get('record_output_timestamps') is ps.installed['record']
        and processor.__dict__.get('process_outputs') is ps.installed['process']
        and _bindings_current()
    )


def _record(pool, outputs, *, output_ts=None):
    ps = _POOLS.get(pool)
    if ps is None:
        raise RuntimeError('lost private pool binding')
    pending = ps.pending
    if _POOL_CALL.get() is not ps or pending is None:
        return type(pool).record_output_timestamps(pool, outputs, output_ts=output_ts)
    try:
        if output_ts is None:
            raise RuntimeError('original pool output timestamp required')
        result = ps.originals[1](pool, outputs, output_ts=output_ts)
        # Omitted rows were nonempty DELTA tokens, MM empty, with no audio.
        # Keep one timestamp entry per original would-be output, including
        # duplicate request IDs; never synthesize audio metrics.
        for rid in pending:
            pool._output_timestamps_by_request.setdefault(rid, []).append(output_ts)
            pool._non_empty_first_output_timestamps_by_request.setdefault(rid, output_ts)
        _COUNTS['timestamp_rows'] += len(pending)
        return result
    finally:
        ps.pending = None


def _process(processor, outputs, engine_core_timestamp=None, iteration_stats=None):
    ps = _POOL_CALL.get()
    if ps is None or ps.processor_ref() is not processor:
        return type(processor).process_outputs(processor, outputs, engine_core_timestamp, iteration_stats)
    _COUNTS['processor_batches'] += 1
    if ps.scope is not None or ps.pending is not None:
        raise RuntimeError('overlapping private pool batch')
    if not _eligible_pool(ps):
        _COUNTS['fallback_batches'] += 1
        return type(processor).process_outputs(processor, outputs, engine_core_timestamp, iteration_stats)
    omitted = []
    ps.scope = omitted
    try:
        result = ps.originals[2](processor, outputs, engine_core_timestamp, iteration_stats)
        ps.pending = omitted
        _COUNTS['selected_batches'] += 1
        return result
    except BaseException:
        _COUNTS['errors'] += 1
        raise
    finally:
        ps.scope = None


async def _pool_call(pool, replica_id, raw_outputs, iteration_stats=None):
    ps = _POOLS.get(pool)
    if ps is None:
        raise RuntimeError('lost private pool binding')
    token = _POOL_CALL.set(ps)
    try:
        return await type(pool).process_llm_raw_outputs(pool, replica_id, raw_outputs, iteration_stats=iteration_stats)
    finally:
        _POOL_CALL.reset(token)
        # Exception before timestamp recording has the same no-publication
        # outcome as the original; no partial selected IDs survive reuse.
        ps.pending = None
        if ps.closed and ps.scope is None:
            _detach_pool(ps)


def _make(
    state, new_token_ids, pooling_output, finish_reason, stop_reason, kv_transfer_params=None, routed_experts=None
):
    lease = getattr(state, _KEY, None)
    old = type(state).make_request_output
    if type(lease) is not _Lease or lease.issuer is not _ISSUER:
        return old(
            state, new_token_ids, pooling_output, finish_reason, stop_reason, kv_transfer_params, routed_experts
        )
    ps = lease.pool_state
    processor = ps.processor_ref()
    owner = lease.owner_ref()
    outer = lease.outer_ref()
    if (
        ps.closed
        or _CLOSED
        or processor is None
        or processor.request_states.get(lease.internal_id) is not state
        or owner is None
        or outer is None
        or owner.request_states.get(lease.external_id) is not outer
        or outer.prompt is not lease.prompt
        or lease.prompt.get(_BINDINGS['marker_key']) is not _BINDINGS['capability']
        or outer.streaming.enabled
        or outer.final_stage_id != 1
        or outer.final_output_stage_ids != {1}
    ):
        _release(state, 'association_changed')
        return old(
            state, new_token_ids, pooling_output, finish_reason, stop_reason, kv_transfer_params, routed_experts
        )
    if (
        not _CLOSED
        and not ps.closed
        and ps.scope is not None
        and _POOL_CALL.get() is ps
        and processor is not None
        and processor.request_states.get(lease.internal_id) is state
        and owner is not None
        and owner.request_states.get(lease.external_id) is outer
        and outer is not None
        and outer.prompt is lease.prompt
        and lease.prompt.get(_BINDINGS['marker_key']) is _BINDINGS['capability']
        and not outer.streaming.enabled
        and outer.final_stage_id == 1
        and outer.final_output_stage_ids == {1}
        and type(state) is _BINDINGS['state_type']
        and state.external_req_id == lease.external_id
        and state.request_id == lease.internal_id
        and type(new_token_ids) is list
        and bool(new_token_ids)
        and pooling_output is None
        and finish_reason is None
        and stop_reason is None
        and kv_transfer_params is None
        and routed_experts is None
        and not state.routed_experts_chunks
        and type(state.mm_accumulated) is _BINDINGS['payload_type']
        and state.mm_accumulated.is_empty
        and state.parent_req is None
        and state.queue is None
        and not state.streaming_input
        and state.output_kind is _BINDINGS['delta']
        and state.stream_interval == 1
        and type(state.detokenizer) is _BINDINGS['detokenizer_type']
        and type(state.logprobs_processor) is _BINDINGS['logprobs_type']
        and '_new_completion_output' not in state.__dict__
        and '_new_request_output' not in state.__dict__
        and 'get_next_output_text' not in state.detokenizer.__dict__
        and 'pop_prompt_logprobs' not in state.logprobs_processor.__dict__
    ):
        # These original calls advance text-read position and drain prompt
        # logprobs; neither may be omitted along with the pure constructors.
        state.detokenizer.get_next_output_text(False, True)
        state.logprobs_processor.pop_prompt_logprobs()
        ps.scope.append(lease.external_id)
        _COUNTS['omitted_nonterminal'] += 1
        return None
    _COUNTS['original_fallback_rows'] += 1
    return old(state, new_token_ids, pooling_output, finish_reason, stop_reason, kv_transfer_params, routed_experts)


def _bind(processor, request, prior):
    global _GENERATION
    admission = _ADMISSION.get()
    if prior is not None or admission is None or _CLOSED:
        return
    owner, msg = admission
    external = msg.request_id
    pool = owner.stage_pools[0]
    outer = owner.request_states.get(external)
    state = processor.request_states.get(request.request_id)
    egress = _BINDINGS['egress']
    if (
        msg.prompt is not request
        or msg.original_prompt is not getattr(outer, 'prompt', None)
        or pool.output_processor is not processor
        or outer is None
        or not egress._eligible(owner, outer, 0)
        or owner.duplex_control_plane is not None
        or owner._pd_pair is not None
        or not owner.async_chunk
        or not owner._stage_receives_async_chunks(1)
        or len(owner.stage_pools) != 2
        or not owner.stage_pools[1].final_output
        or getattr(owner.stage_pools[1].stage_client, 'final_output_type', None) != 'audio'
        or type(processor) is not _BINDINGS['processor_type']
        or type(state) is not _BINDINGS['state_type']
        or state.request_id != request.request_id
        or state.external_req_id != external
        or request.external_req_id != external
        or request.resumable
        or state.streaming_input
        or state.parent_req is not None
        or state.queue is not None
        or state.output_kind is not _BINDINGS['delta']
        or state.stream_interval != 1
        or type(state.detokenizer) is not _BINDINGS['detokenizer_type']
        or type(state.logprobs_processor) is not _BINDINGS['logprobs_type']
        or 'make_request_output' in state.__dict__
    ):
        _COUNTS['admission_fallbacks'] += 1
        return
    ps = _POOLS.get(pool)
    if ps is None:
        if (
            'process_llm_raw_outputs' in pool.__dict__
            or 'record_output_timestamps' in pool.__dict__
            or 'process_outputs' in processor.__dict__
        ):
            _COUNTS['admission_fallbacks'] += 1
            return
        ps = _PoolState(pool, processor, owner)
        _POOLS[pool] = ps
        pool.process_llm_raw_outputs = MethodType(_pool_call, pool)
        pool.record_output_timestamps = MethodType(_record, pool)
        processor.process_outputs = MethodType(_process, processor)
        ps.installed = {
            'pool': pool.process_llm_raw_outputs,
            'record': pool.record_output_timestamps,
            'process': processor.process_outputs,
        }
    if (
        ps.owner_ref() is not owner
        or ps.processor_ref() is not processor
        or ps.closed
        or (ps.pid, ps.tid) != (os.getpid(), threading.get_ident())
    ):
        _COUNTS['admission_fallbacks'] += 1
        return
    _GENERATION += 1
    lease = _Lease(
        _ISSUER,
        _GENERATION,
        weakref.ref(owner),
        weakref.ref(outer),
        weakref.ref(state),
        ps,
        state.request_id,
        external,
        outer.prompt,
    )
    setattr(state, _KEY, lease)
    state.make_request_output = MethodType(_make, state)
    ps.leases[state.request_id] = lease
    _COUNTS['bound_fresh_owners'] += 1


def _bindings_current():
    b = _BINDINGS
    return (
        b is not None
        and all(getattr(cls, name) is value for cls, name, value in b['class_bindings'])
        and all(namespace.get(name) is value for namespace, name, value in b['global_bindings'])
    )


def status():
    pools = list(_POOLS.values())
    return dict(
        schema=1,
        configured_flag=_flag(),
        enabled=_INSTALLED,
        closed=_CLOSED,
        bindings_matched=not _INSTALLED or _bindings_current(),
        counts=dict(_COUNTS),
        live_owners=sum(len(ps.leases) for ps in pools),
        active_scopes=sum(ps.scope is not None for ps in pools),
        pending_timestamp_rows=sum(len(ps.pending or ()) for ps in pools),
        source_method_bodies_unchanged=True,
        global_constructor_skips=False,
        model_math_changed=False,
        eligible_route='Initial private legacy rawPCM HTTP capability + exact current outer/frontend UUID owner; MM-empty nonterminal only',
        pool_bindings=[
            dict(pid=ps.pid, tid=ps.tid, pool=id(ps.pool_ref()), processor=id(ps.processor_ref()), closed=ps.closed)
            for ps in pools
        ],
    )


def _detach_pool(ps):
    pool = ps.pool_ref()
    processor = ps.processor_ref()
    for obj, name, key in (
        (pool, 'process_llm_raw_outputs', 'pool'),
        (pool, 'record_output_timestamps', 'record'),
        (processor, 'process_outputs', 'process'),
    ):
        if obj is not None and obj.__dict__.get(name) is ps.installed[key]:
            obj.__dict__.pop(name, None)


def close():
    global _CLOSED
    _CLOSED = True
    for ps in list(_POOLS.values()):
        ps.closed = True
        for lease in list(ps.leases.values()):
            state = lease.state_ref()
            if state is not None:
                _release(state, 'shutdown')
        # Keep an in-flight synchronous batch's recording path alive until its
        # original timestamp call/finally consumes the selected IDs. Closing
        # stops further omissions immediately and releases owner authority.
        if ps.scope is None and ps.pending is None:
            _detach_pool(ps)


def install_http_unused_talker_output():
    global _INSTALLED, _BINDINGS
    if _flag() == '0' or _INSTALLED:
        return status()
    from . import http_audio_egress as egress

    if not egress.status()['enabled']:
        raise RuntimeError('install original private HTTP egress first')
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.engine.stage_pool import StagePool
    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

    original_add = MultimodalOutputProcessor.add_request
    original_handle = Orchestrator._handle_add_request
    original_finish = MultimodalOutputProcessor._finish_request
    original_abort = MultimodalOutputProcessor.abort_requests
    original_remove = MultimodalOutputProcessor.remove_request
    original_cleanup = Orchestrator._cleanup_request_ids
    original_shutdown = AsyncOmniEngine.shutdown
    make = OmniRequestState.make_request_output

    @functools.wraps(original_handle)
    async def handle(self, msg):
        prior_outer = self.request_states.get(msg.request_id)
        if prior_outer is not None:
            for ps in list(_POOLS.values()):
                if ps.owner_ref() is self:
                    for lease in list(ps.leases.values()):
                        if lease.outer_ref() is prior_outer:
                            state = lease.state_ref()
                            if state is not None:
                                _release(state, 'outer_replaced')
        token = _ADMISSION.set((self, msg) if prior_outer is None else None)
        try:
            return await original_handle(self, msg)
        except BaseException:
            # Original StagePool rollback uses the outer ID in this pinned
            # runtime; a UUID frontend can remain registered on submission
            # failure. Revoke only our capability; do not change baseline
            # frontend cleanup or add a request/state mutation here.
            request = getattr(msg, 'prompt', None)
            pool = self.stage_pools[0]
            processor = getattr(pool, '_output_processor', None)
            state = (
                processor.request_states.get(getattr(request, 'request_id', None)) if processor is not None else None
            )
            if state is not None:
                _release(state, 'admission_error')
            raise
        finally:
            _ADMISSION.reset(token)

    @functools.wraps(original_add)
    def add(self, request, prompt, parent_req=None, request_index=0, queue=None):
        prior = self.request_states.get(request.request_id)
        if prior is not None:
            _release(prior, 'existing_owner_update')
        result = original_add(self, request, prompt, parent_req, request_index, queue)
        _bind(self, request, prior)
        return result

    @functools.wraps(original_finish)
    def finish(self, state):
        _release(state, 'finish')
        return original_finish(self, state)

    @functools.wraps(original_abort)
    def abort(self, request_ids, internal=False):
        for rid in request_ids:
            ids = (rid,) if internal else tuple(self.external_req_ids.get(rid, ()))
            for internal_id in ids:
                state = self.request_states.get(internal_id)
                if state is not None:
                    _release(state, 'abort')
        return original_abort(self, request_ids, internal)

    @functools.wraps(original_remove)
    def remove(self, request_id):
        state = self.request_states.get(request_id)
        if state is not None:
            _release(state, 'rollback_remove')
        return original_remove(self, request_id)

    @functools.wraps(original_cleanup)
    async def cleanup(self, request_ids, **kwargs):
        for ps in list(_POOLS.values()):
            if ps.owner_ref() is self:
                for lease in list(ps.leases.values()):
                    if lease.external_id in request_ids:
                        state = lease.state_ref()
                        if state is not None:
                            _release(state, 'outer_cleanup')
        return await original_cleanup(self, request_ids, **kwargs)

    @functools.wraps(original_shutdown)
    def shutdown(self):
        try:
            return original_shutdown(self)
        finally:
            close()

    hooks = [
        (Orchestrator, '_handle_add_request', handle),
        (MultimodalOutputProcessor, 'add_request', add),
        (MultimodalOutputProcessor, '_finish_request', finish),
        (MultimodalOutputProcessor, 'abort_requests', abort),
        (MultimodalOutputProcessor, 'remove_request', remove),
        (Orchestrator, '_cleanup_request_ids', cleanup),
        (AsyncOmniEngine, 'shutdown', shutdown),
    ]
    leaves = [
        (OmniRequestState, 'make_request_output'),
        (OmniRequestState, '_new_completion_output'),
        (OmniRequestState, '_new_request_output'),
        (RequestState, '_new_completion_output'),
        (RequestState, '_new_request_output'),
        (IncrementalDetokenizer, 'get_next_output_text'),
        (LogprobsProcessor, 'pop_prompt_logprobs'),
        (CompletionOutput, '__init__'),
        (RequestOutput, '__init__'),
        (Orchestrator, '_route_output'),
        (StagePool, 'process_llm_raw_outputs'),
        (StagePool, 'record_output_timestamps'),
        (MultimodalOutputProcessor, 'process_outputs'),
    ]
    _BINDINGS = dict(
        make=make,
        process=MultimodalOutputProcessor.process_outputs,
        state_type=OmniRequestState,
        processor_type=MultimodalOutputProcessor,
        payload_type=MultimodalPayload,
        delta=RequestOutputKind.DELTA,
        detokenizer_type=IncrementalDetokenizer,
        logprobs_type=LogprobsProcessor,
        marker_key=egress._PROMPT_KEY,
        capability=egress._CAPABILITY,
        egress=egress,
        class_bindings=[(cls, name, getattr(cls, name)) for cls, name in leaves] + hooks,
        global_bindings=[
            (fn.__globals__, name, fn.__globals__[name])
            for fn in (
                make,
                OmniRequestState._new_completion_output,
                OmniRequestState._new_request_output,
                RequestState._new_completion_output,
                RequestState._new_request_output,
            )
            for name in fn.__code__.co_names
            if name in fn.__globals__
        ],
        originals=dict(
            add=original_add,
            handle=original_handle,
            finish=original_finish,
            abort=original_abort,
            remove=original_remove,
            cleanup=original_cleanup,
            shutdown=original_shutdown,
        ),
    )
    for cls, name, method in hooks:
        setattr(cls, name, method)
    _INSTALLED = True
    _COUNTS['installs'] += 1
    return status()
