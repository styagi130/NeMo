import asyncio
import gc
import weakref
from collections import Counter
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreRequest

EngineCoreOutput = OmniEngineCoreOutput
from easymagpie_vllm_omni import http_audio_egress as egress
from easymagpie_vllm_omni import http_unused_talker_output as m
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.orchestrator import CfgCompanionTracker, Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState


@pytest.fixture(autouse=True)
def clean(monkeypatch, request):
    hooks = [
        (Orchestrator, '_handle_add_request'),
        (Orchestrator, '_cleanup_request_ids'),
        (Orchestrator, '_route_output'),
        (MultimodalOutputProcessor, 'add_request'),
        (MultimodalOutputProcessor, '_finish_request'),
        (MultimodalOutputProcessor, 'abort_requests'),
        (MultimodalOutputProcessor, 'remove_request'),
        (AsyncOmniEngine, 'shutdown'),
        (OmniOpenAIServingSpeech, 'create_speech'),
    ]
    saved = [(c, n, n in c.__dict__, getattr(c, n)) for c, n in hooks]
    monkeypatch.setenv(m.ENV, '1')
    monkeypatch.setenv('EASYMAGPIE_HTTP_AUDIO_EGRESS', '1')
    egress.install_http_audio_egress()
    if request.node.name != 'test_default_off_and_invalid_flag_leave_every_original_binding':
        m.install_http_unused_talker_output()
    import vllm_omni.engine.stage_pool as pool_module

    monkeypatch.setattr(pool_module._time, 'time', lambda: 100.0)
    yield
    m.close()
    for cls, name, had, value in saved:
        if had:
            setattr(cls, name, value)
        elif name in cls.__dict__:
            delattr(cls, name)
    m._POOLS.clear()
    m._INSTALLED = False
    m._CLOSED = False
    m._BINDINGS = None
    m._COUNTS.clear()
    m._GENERATION = 0
    egress._INSTALLED = False
    egress._COUNTS.clear()
    assert m._ADMISSION.get() is None and m._POOL_CALL.get() is None and not torch.cuda.is_initialized()


class World:
    def __init__(self, *, marked=True, distinct=False, fail_send=False, stats=False):
        self.marked = marked
        self.distinct = distinct
        self.fail_send = fail_send
        self.sends = []
        self.aborts = []
        self.tx = []
        self.owner = object.__new__(Orchestrator)
        o = self.owner
        o.request_states = {}
        o.async_chunk = True
        o._pd_pair = None
        o._pd_kv_params = {}
        o.duplex_control_plane = None
        o._cfg_tracker = CfgCompanionTracker()
        o._running_counter = None
        o.output_async_queue = asyncio.Queue()
        o._stage_receives_async_chunks = lambda stage: True
        o._emit_tx_edge = lambda **kw: self.tx.append(kw)
        o.stage_pools = [self.pool(0, 'latent', stats), self.pool(1, 'audio', stats)]
        self.pool0 = o.stage_pools[0]
        self.processor = self.pool0.output_processor
        self.engine = object.__new__(AsyncOmniEngine)
        e = self.engine
        e.stage_metadata = [NS(stage_type='llm'), NS(stage_type='llm')]
        e.stage_pools = o.stage_pools
        e.default_sampling_params_list = self.params()
        e.supported_tasks = ('generate',)
        e._scope_stage0_multimodal_cache_to_replica = lambda *a: 0
        p = object.__new__(InputProcessor)
        p._validate_params = lambda *a: None
        p._validate_lora = lambda *a: None
        p._validate_model_inputs = lambda *a: None
        p.vllm_config = NS(
            parallel_config=NS(data_parallel_size=1, data_parallel_size_local=1, local_engines_only=True)
        )
        p.model_config = NS(max_model_len=4096)
        p.generation_config_fields = {}
        p.renderer = NS(tokenizer=None, get_eos_token_id=lambda: 1)
        p.input_preprocessor = NS(
            preprocess=lambda prompt, **kw: {'type': 'token', 'prompt_token_ids': list(prompt['prompt_token_ids'])}
        )
        self.input_processor = p

        def process_inputs(**kwargs):
            value = InputProcessor.process_inputs(p, **kwargs)
            if self.distinct:
                InputProcessor.assign_request_id(value)
            return value

        e.input_processor = NS(process_inputs=process_inputs)

    def params(self):
        return [
            SamplingParams(detokenize=False, output_kind=RequestOutputKind.DELTA, max_tokens=256, stop_token_ids=[1]),
            SamplingParams(detokenize=True, output_kind=RequestOutputKind.DELTA, max_tokens=256),
        ]

    def pool(self, stage, kind, stats):
        from vllm_omni.engine.stage_pool import _ReplicaMetrics

        pool = object.__new__(StagePool)
        pool.stage_id = stage
        pool._replica_metrics = [_ReplicaMetrics()]

        async def send(request, **kw):
            if self.fail_send and stage == 0:
                raise RuntimeError('transport unavailable')
            # Actual core-wire serialization; private authority is API-local.
            restored = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
            assert not any(k.startswith('_easymagpie_unused') for k in getattr(request, '__dict__', {}))
            self.sends.append((stage, restored))

        async def abort(ids):
            self.aborts.append((stage, list(ids)))

        pool.clients = [
            NS(
                stage_type='llm',
                final_output=True,
                final_output_type=kind,
                add_request_async=send,
                abort_requests_async=abort,
            )
        ]
        pool._output_processor = MultimodalOutputProcessor(None, log_stats=stats, engine_core_output_type=kind)
        pool._stage_vllm_config = NS(model_config=NS(max_model_len=4096))
        pool._request_bindings = {}
        pool._affinity = {}
        pool._addr_to_replica_id = {}
        for name in (
            '_output_timestamps_by_request',
            '_non_empty_first_output_timestamps_by_request',
            '_audio_frames_by_request',
            '_audio_sample_rate_by_request',
        ):
            setattr(pool, name, {})

        async def pick(rid, **kw):
            pool._request_bindings[rid] = 0
            return 0

        pool._pick_or_select = pick
        return pool

    async def admit(self, *, external='http', original=None, request_id=None, resumable=False):
        # Actual entrypoint UUID function then actual engine preprocessing and
        # external-ID restoration; optional explicit actual assign_request_id
        # separately stresses unequal outer/core IDs, not a production claim.
        rid = request_id or AsyncOmni._get_unique_request_id(external)
        prompt = original if original is not None else {'prompt_token_ids': [0] * 71, 'prompt': 'text'}
        if self.marked and original is None:
            http = object()
            token = egress._HTTP.set((http, self.engine))
            try:
                egress.mark_legacy_http_audio_prompt(prompt, http, self.engine)
            finally:
                egress._HTTP.reset(token)
        msg = self.engine._build_add_request_message(
            rid,
            prompt,
            sampling_params_list=self.params(),
            final_stage_id=1,
            final_output_stage_ids=[1],
            arrival_time=1.0,
            resumable=resumable,
        )
        await self.owner._handle_add_request(msg)
        return msg, self.processor.request_states[msg.prompt.request_id]

    async def process(self, state, *, finish=None, ids=(0,), route=True, iteration_stats=None, **kwargs):
        row = EngineCoreOutput(
            request_id=state.request_id,
            new_token_ids=list(ids),
            finish_reason=finish,
            multimodal_output=None,
            **kwargs,
        )
        raw = EngineCoreOutputs(outputs=[row], timestamp=2.0)
        result = await self.pool0.process_llm_raw_outputs(0, raw, iteration_stats=iteration_stats)
        if route:
            await self.owner._handle_processed_outputs(0, 0, result)
        return result


def normalized(v):
    if isinstance(v, torch.Tensor):
        return (str(v.dtype), tuple(v.shape), v.tolist())
    if isinstance(v, dict):
        return {k: normalized(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalized(x) for x in v]
    if isinstance(v, (RequestOutput, CompletionOutput, MultimodalPayload)):
        return (type(v).__name__, normalized(v.__dict__))
    return v


@pytest.mark.parametrize('distinct', [False, True])
def test_actual_entrypoint_engine_input_stagepool_prewarm_UUID_and_output(distinct):
    async def run():
        w = World(distinct=distinct)
        msg, state = await w.admit()
        assert msg.request_id.startswith('http-') and len(msg.request_id) == 13
        assert (state.request_id != msg.request_id) is distinct
        assert state.external_req_id == msg.request_id
        lease = getattr(state, m._KEY)
        assert lease.outer_ref() is w.owner.request_states[msg.request_id] and lease.state_ref() is state
        assert len(w.sends) == 2 and [x[0] for x in w.sends] == [0, 1]
        assert w.sends[0][1].request_id == state.request_id and w.sends[0][1].external_req_id == msg.request_id
        assert w.sends[1][1].request_id == msg.request_id
        assert (
            not hasattr(w.owner.stage_pools[1].output_processor, 'process_outputs')
            or 'process_outputs' not in w.owner.stage_pools[1].output_processor.__dict__
        )
        result = await w.process(state)
        assert result == [] and w.owner.output_async_queue.qsize() == 0
        assert w.pool0._output_timestamps_by_request[msg.request_id] == [100.0]
        assert w.pool0._non_empty_first_output_timestamps_by_request[msg.request_id] == 100.0
        assert m.status()['counts']['omitted_nonterminal'] == m.status()['counts']['timestamp_rows'] == 1

    asyncio.run(run())


@pytest.mark.parametrize('finish', [FinishReason.STOP, FinishReason.LENGTH, FinishReason.ABORT])
def test_real_original_nonterminal_and_terminal_statistics_route_fields(finish):
    async def run():
        old, new = World(marked=False, stats=True), World(stats=True)
        # Identical chosen logicalID, different server-owned capability.
        msg, a = await old.admit(request_id='paired')
        _, b = await new.admit(request_id='paired')
        old.owner.request_states['paired'].pipeline_timings.clear()
        new.owner.request_states['paired'].pipeline_timings.clear()
        Stats = MultimodalOutputProcessor._update_stats_from_output.__globals__['IterationStats']
        left, right = Stats(), Stats()
        await old.process(a, iteration_stats=left)
        await new.process(b, iteration_stats=right)
        assert (
            a.detokenizer.__dict__ == b.detokenizer.__dict__
            and a.logprobs_processor.__dict__ == b.logprobs_processor.__dict__
        )
        assert (
            left.__dict__ == right.__dict__
            and old.processor._native_text_metrics_by_request == new.processor._native_text_metrics_by_request
        )
        x = await old.process(a, finish=finish, ids=(1,), iteration_stats=left)
        y = await new.process(b, finish=finish, ids=(1,), iteration_stats=right)
        assert normalized(x) == normalized(y) and left.__dict__ == right.__dict__
        assert old.pool0._output_timestamps_by_request == new.pool0._output_timestamps_by_request
        assert (
            old.pool0._non_empty_first_output_timestamps_by_request
            == new.pool0._non_empty_first_output_timestamps_by_request
        )
        assert not hasattr(b, m._KEY) and 'make_request_output' not in b.__dict__ and not new.processor.request_states
        # old unmarked original route emits both; selected route retains only
        # terminal; genuine terminal payload/metrics must be identical.
        assert old.owner.output_async_queue.qsize() == 2 and new.owner.output_async_queue.qsize() == 1
        old.owner.output_async_queue.get_nowait()
        ox = old.owner.output_async_queue.get_nowait()
        ny = new.owner.output_async_queue.get_nowait()
        assert normalized(ox.engine_outputs) == normalized(ny.engine_outputs) and ox.metrics == ny.metrics

    asyncio.run(run())


@pytest.mark.parametrize(
    'case',
    [
        'empty',
        'stop',
        'transfer',
        'ec_transfer',
        'routed',
        'MM',
        'queue',
        'cumulative',
        'interval',
        'streaming',
        'custom_factory',
        'custom_readout',
    ],
)
def test_bound_unsupported_results_and_mutations_preserve_original_outputs(case):
    async def run():
        w = World()
        msg, state = await w.admit()
        kwargs = {}
        ids = (0,)
        if case == 'empty':
            ids = ()
        if case == 'stop':
            kwargs['stop_reason'] = 42
        if case == 'transfer':
            kwargs['kv_transfer_params'] = {'transfer': 1}
        if case == 'ec_transfer':
            kwargs['ec_transfer_params'] = {'transfer': 1}
        if case == 'routed':
            kwargs['routed_experts'] = torch.zeros((1, 1), dtype=torch.int32)
        if case == 'MM':
            state.add_multimodal_tensor({'codes': {'audio': torch.ones((1, 16), dtype=torch.long)}}, 'latent')
        if case == 'queue':
            state.queue = NS(put=lambda output: None)
        if case == 'cumulative':
            state.output_kind = RequestOutputKind.CUMULATIVE
        if case == 'interval':
            state.stream_interval = 2
        if case == 'streaming':
            state.streaming_input = True
        if case == 'custom_factory':
            state._new_request_output = MethodType(type(state)._new_request_output, state)
        if case == 'custom_readout':
            state.detokenizer.get_next_output_text = MethodType(
                type(state.detokenizer).get_next_output_text, state.detokenizer
            )
        before = m._COUNTS['omitted_nonterminal']
        result = await w.process(state, ids=ids, route=False, **kwargs)
        assert m._COUNTS['omitted_nonterminal'] == before
        assert len(result) == (0 if case == 'queue' else 1)

    asyncio.run(run())


@pytest.mark.parametrize(
    'case',
    ['unmarked', 'forged', 'resumable', 'duplex_plane', 'pd', 'wrong_stage1', 'foreign_frontend', 'outside_admission'],
)
def test_only_initial_trusted_HTTP_exact_topology_can_grant(case):
    async def run():
        w = World(marked=case not in ('unmarked', 'forged'))
        if case == 'duplex_plane':
            w.owner.duplex_control_plane = NS(session_for_identity=lambda identity: None)
        if case == 'pd':
            w.owner._pd_pair = (0, 1)
        if case == 'wrong_stage1':
            w.owner.stage_pools[1].clients[0].final_output_type = 'text'
        if case == 'foreign_frontend':

            class Foreign(MultimodalOutputProcessor):
                pass

            w.pool0._output_processor = Foreign(None, log_stats=False, engine_core_output_type='latent')
            w.processor = w.pool0.output_processor
        original = {'prompt_token_ids': [0] * 71, egress._PROMPT_KEY: 'forged'} if case == 'forged' else None
        if case == 'outside_admission':
            rid = 'direct'
            request = OmniEngineCoreRequest(
                request_id=rid,
                external_req_id=rid,
                prompt_token_ids=[1],
                mm_features=None,
                sampling_params=w.params()[0],
                pooling_params=None,
                arrival_time=1.0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            w.processor.add_request(request, None)
            state = w.processor.request_states[rid]
        else:
            _, state = await w.admit(original=original, resumable=case == 'resumable')
        assert not hasattr(state, m._KEY)

    asyncio.run(run())


def test_direct_make_outside_original_pool_scope_cannot_drop():
    async def run():
        w = World()
        _, state = await w.admit()
        assert isinstance(state.make_request_output([0], None, None, None), RequestOutput)
        assert m._COUNTS['omitted_nonterminal'] == 0

    asyncio.run(run())


def test_updates_revoke_and_cannot_regrant_same_frontend_owner():
    async def run():
        w = World()
        msg, state = await w.admit()
        generation = getattr(state, m._KEY).generation
        w.processor.add_request(msg.prompt, 'stream update')
        assert not hasattr(state, m._KEY) and 'make_request_output' not in state.__dict__
        assert state.request_id not in w.processor.request_states
        # Even a fresh add-message carrying the original private prompt cannot
        # regrant when the frontend object already exists.
        await w.owner._handle_add_request(msg)
        assert not hasattr(w.processor.request_states[state.request_id], m._KEY)
        assert m._GENERATION == generation

    asyncio.run(run())


@pytest.mark.parametrize('method', ['abort_internal', 'abort_external', 'remove', 'outer_cleanup', 'shutdown'])
def test_owner_retirement_detaches_and_releases_all_private_references(method):
    async def run():
        w = World()
        msg, state = await w.admit()
        ref = weakref.ref(state)
        lease_ref = weakref.ref(getattr(state, m._KEY))
        if method == 'abort_internal':
            w.processor.abort_requests([state.request_id], internal=True)
        if method == 'abort_external':
            w.processor.abort_requests([state.external_req_id], internal=False)
        if method == 'remove':
            w.processor.remove_request(state.request_id)
        if method == 'outer_cleanup':
            await w.owner._cleanup_request_ids([msg.request_id], abort=True)
        if method == 'shutdown':
            m.close()
            w.processor.remove_request(state.request_id)
        assert not hasattr(state, m._KEY) and 'make_request_output' not in state.__dict__
        assert m.status()['live_owners'] == 0 and lease_ref() is None
        del state
        gc.collect()
        assert ref() is None

    asyncio.run(run())


def test_submission_failure_revokes_even_when_original_UUID_rollback_misses():
    async def run():
        w = World(fail_send=True, distinct=True)
        with pytest.raises(RuntimeError, match='transport unavailable'):
            await w.admit()
        assert m.status()['live_owners'] == 0
        assert all(not hasattr(s, m._KEY) for s in w.processor.request_states.values())
        assert m._ADMISSION.get() is None

    asyncio.run(run())


def test_processor_exception_clears_scope_pending_and_preserves_exception():
    async def run():
        w = World()
        _, state = await w.admit()

        # Original readout method errors before any timestamp/publication.
        def failure(*args):
            raise LookupError('readout failed')

        state.logprobs_processor.pop_prompt_logprobs = failure
        with pytest.raises(LookupError, match='readout failed'):
            await w.process(state)
        assert m.status()['active_scopes'] == m.status()['pending_timestamp_rows'] == 0
        assert not w.pool0._output_timestamps_by_request and w.owner.output_async_queue.qsize() == 0

    asyncio.run(run())


def test_same_public_ID_reuse_new_UUID_has_new_generation_and_old_state_cannot_drop():
    async def run():
        w = World()
        msg, a = await w.admit(external='same')
        first = getattr(a, m._KEY).generation
        await w.process(a, finish=FinishReason.STOP)
        # Core/orchestrator outer same-ID reuse is explicit new request object.
        msg, b = await w.admit(external='same')
        assert b is not a and getattr(b, m._KEY).generation > first
        assert isinstance(a.make_request_output([0], None, None, None), RequestOutput)
        await w.process(b)
        assert m._COUNTS['omitted_nonterminal'] == 1

    asyncio.run(run())


def test_default_off_and_invalid_flag_leave_every_original_binding(monkeypatch):
    methods = (
        Orchestrator._handle_add_request,
        MultimodalOutputProcessor.add_request,
        OmniRequestState.make_request_output,
        StagePool.process_llm_raw_outputs,
    )
    monkeypatch.setenv(m.ENV, '0')
    assert not m.install_http_unused_talker_output()['enabled']
    assert methods == (
        Orchestrator._handle_add_request,
        MultimodalOutputProcessor.add_request,
        OmniRequestState.make_request_output,
        StagePool.process_llm_raw_outputs,
    )
    monkeypatch.setenv(m.ENV, 'bad')
    with pytest.raises(ValueError):
        m.install_http_unused_talker_output()


def test_original_leaf_function_code_and_class_pool_methods_unchanged():
    from vllm.v1.engine.output_processor import RequestState

    for name in ('make_request_output', '_new_completion_output', '_new_request_output'):
        f = getattr(OmniRequestState, name)
        assert f.__code__.co_filename.endswith('vllm_omni/outputs/output_processor.py')
    assert StagePool.process_llm_raw_outputs.__code__.co_filename.endswith('vllm_omni/engine/stage_pool.py')
    assert MultimodalOutputProcessor.process_outputs.__code__.co_filename.endswith(
        'vllm_omni/outputs/output_processor.py'
    )
    assert m.status()['bindings_matched'] and m.status()['source_method_bodies_unchanged']


@pytest.mark.parametrize('change', ['prompt_cap', 'outer_owner'])
def test_current_association_loss_revokes_permanently_even_when_restored(change):
    async def run():
        w = World()
        msg, state = await w.admit()
        outer = w.owner.request_states[msg.request_id]
        if change == 'prompt_cap':
            saved = outer.prompt.pop(egress._PROMPT_KEY)
        else:
            w.owner.request_states.pop(msg.request_id)
        await w.process(state, route=False)
        assert not hasattr(state, m._KEY) and m._COUNTS['omitted_nonterminal'] == 0
        if change == 'prompt_cap':
            outer.prompt[egress._PROMPT_KEY] = saved
        else:
            w.owner.request_states[msg.request_id] = outer
        await w.process(state, route=False)
        assert m._COUNTS['omitted_nonterminal'] == 0

    asyncio.run(run())


def test_close_mid_batch_keeps_original_recording_of_already_omitted_rows(monkeypatch):
    async def run():
        w = World()
        msg, state = await w.admit()
        original = w.processor._update_stats_from_output
        calls = 0

        def stats(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                m.close()
            return original(*args, **kwargs)

        monkeypatch.setattr(w.processor, '_update_stats_from_output', stats)
        rows = [
            EngineCoreOutput(request_id=state.request_id, new_token_ids=[0]),
            EngineCoreOutput(request_id=state.request_id, new_token_ids=[0]),
        ]
        result = await w.pool0.process_llm_raw_outputs(0, EngineCoreOutputs(outputs=rows, timestamp=2.0))
        assert len(result) == 1 and w.pool0._output_timestamps_by_request[msg.request_id] == [100.0, 100.0]
        assert m._COUNTS['omitted_nonterminal'] == m._COUNTS['timestamp_rows'] == 1
        assert (
            m.status()['closed']
            and m.status()['live_owners'] == m.status()['active_scopes'] == m.status()['pending_timestamp_rows'] == 0
        )
        assert 'process_llm_raw_outputs' not in w.pool0.__dict__ and 'process_outputs' not in w.processor.__dict__

    asyncio.run(run())


def test_mixed_private_foreign_and_same_delta_terminal_order_and_timestamps():
    async def run():
        w = World()
        a_msg, a = await w.admit(external='private')
        w.marked = False
        b_msg, b = await w.admit(external='foreign')
        rows = [
            EngineCoreOutput(request_id=a.request_id, new_token_ids=[0]),
            EngineCoreOutput(request_id=b.request_id, new_token_ids=[0]),
            EngineCoreOutput(request_id=a.request_id, new_token_ids=[1], finish_reason=FinishReason.STOP),
        ]
        outputs = await w.pool0.process_llm_raw_outputs(0, EngineCoreOutputs(outputs=rows, timestamp=2.0))
        assert [output.request_id for output in outputs] == [b_msg.request_id, a_msg.request_id]
        assert w.pool0._output_timestamps_by_request[a_msg.request_id] == [100.0, 100.0]
        assert w.pool0._output_timestamps_by_request[b_msg.request_id] == [100.0]
        assert m._COUNTS['omitted_nonterminal'] == m._COUNTS['timestamp_rows'] == 1
        assert not hasattr(a, m._KEY)

    asyncio.run(run())


def test_original_detokenizer_stop_drives_finish_and_real_client_abort():
    async def run():
        w = World()
        _, state = await w.admit()
        original = state.detokenizer.update

        def stop(ids, stopped):
            original(ids, stopped)
            return 'end'

        state.detokenizer.update = stop
        outputs = await w.process(state, route=False)
        assert len(outputs) == 1 and outputs[0].finished and outputs[0].outputs[0].stop_reason == 'end'
        assert w.aborts == [(0, [state.request_id])] and m._COUNTS['omitted_nonterminal'] == 0
        assert not hasattr(state, m._KEY)

    asyncio.run(run())


def test_first_output_and_closed_status_preserve_ownership_and_timestamps():
    async def run():
        w = World()
        _, state = await w.admit()
        await w.process(state)
        first = m.status()
        assert first['counts']['omitted_nonterminal'] == first['counts']['timestamp_rows'] == 1
        assert first['bindings_matched'] and first['active_scopes'] == first['pending_timestamp_rows'] == 0
        m.close()
        final = m.status()
        assert (
            final['closed'] and final['live_owners'] == final['active_scopes'] == final['pending_timestamp_rows'] == 0
        )
        assert final['counts']['omitted_nonterminal'] == 1

    asyncio.run(run())


def test_foreign_late_factory_binding_falls_back_to_actual_current_method(monkeypatch):
    async def run():
        w = World()
        _, state = await w.admit()
        calls = []
        old = OmniRequestState._new_request_output

        def custom(self, *args, **kwargs):
            calls.append(self)
            return old(self, *args, **kwargs)

        monkeypatch.setattr(OmniRequestState, '_new_request_output', custom)
        outputs = await w.process(state, route=False)
        assert len(outputs) == 1 and calls == [state] and m._COUNTS['omitted_nonterminal'] == 0

    asyncio.run(run())
