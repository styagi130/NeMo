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
import gc
import threading
import weakref
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
from easymagpie_vllm_omni.codec_async_output import (
    DeferredCodecOutput,
    build_codec_async_output,
    codec_async_fallback_reason,
)
from easymagpie_vllm_omni.runner import EasyMagpieCodecGPUGenerationModelRunner
from easymagpie_vllm_omni.scheduler import (
    EasyMagpieAsyncCodecScheduler,
    EasyMagpieCodecScheduler,
    _poll_fenced_codec_chunk,
    _process_fenced_codec_chunks,
)
from test_scheduler import _codec_poll_state
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import RequestStatus
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


class FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device('cuda:0')

    def record_stream(self, stream):
        stream.owned.append(self)


def runner_state():
    runner = object.__new__(EasyMagpieCodecGPUGenerationModelRunner)
    cls = type('EasyMagpieCodecForConditionalGeneration', (), {'__module__': 'easymagpie_vllm_omni.codec.model'})
    runner.model = cls()
    runner._async_chunk = True
    runner.speculative_config = None
    runner.supports_mm_inputs = False
    runner.routed_experts_initialized = False
    runner.kv_connector_output = None
    runner.use_async_scheduling = True
    runner._should_accumulate_full_payload_output = lambda: False
    runner.get_omni_connector_output = lambda: None
    runner.vllm_config = NS(
        parallel_config=NS(),
        scheduler_config=NS(scheduler_cls='easymagpie_vllm_omni.scheduler.EasyMagpieAsyncCodecScheduler'),
    )
    runner.input_batch = NS(num_reqs=2, req_ids=['b', 'a'], req_id_to_index={'b': 0, 'a': 1})
    audio = [torch.tensor([1.0, -2.0]).as_subclass(FakeCudaTensor), torch.empty(0).as_subclass(FakeCudaTensor)]
    state = ExecuteModelState(*([None] * len(ExecuteModelState._fields)))._replace(
        scheduler_output=NS(num_scheduled_tokens={'b': 2, 'a': 2}),
        multimodal_outputs={'model_outputs': audio, 'sr': [torch.tensor(22050), torch.tensor(22050)]},
    )
    runner.execute_model_state = state
    return runner, state


@pytest.fixture
def copies(monkeypatch):
    pending, events, streams = [], [], []
    original_empty, original_copy = torch.empty, torch.Tensor.copy_

    class Event:
        def __init__(self):
            self.allowed = threading.Event()
            self.entered = threading.Event()
            self.copies = []
            events.append(self)

        def record(self, stream):
            self.copies, pending[:] = list(pending), []

        def synchronize(self):
            self.entered.set()
            assert self.allowed.wait(5), 'copy completion was never allowed'
            for target, source in self.copies:
                original_copy(target, source.as_subclass(torch.Tensor))
            self.copies.clear()

    class Stream:
        def __init__(self, **kwargs):
            self.owned = []
            self.waits = []
            streams.append(self)

        def wait_event(self, event):
            self.waits.append(event)

    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)

    def copy(target, source, non_blocking=False):
        if non_blocking:
            pending.append((target, source))
            return target
        return original_copy(target, source)

    monkeypatch.setattr(torch, 'empty', empty)
    monkeypatch.setattr(torch.Tensor, 'copy_', copy)
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: 'producer')
    return events, streams


def test_deferred_output_waits_owns_sources_and_snapshots_request_ids(copies, monkeypatch):
    runner, state = runner_state()
    monkeypatch.setenv('EASYMAGPIE_CODEC_ASYNC_D2H', '1')
    assert codec_async_fallback_reason(runner, state) is None
    result = runner.sample_tokens()
    assert isinstance(result, DeferredCodecOutput) and runner.execute_model_state is None
    ready = result._ready
    source = weakref.ref(result._source)
    state.multimodal_outputs['model_outputs'][0].fill_(99)
    runner.input_batch.req_ids.reverse()
    runner.input_batch.req_id_to_index.clear()
    completed = []
    thread = threading.Thread(target=lambda: completed.append(result.get_output()))
    thread.start()
    assert ready.entered.wait(5)
    assert not completed and source() is not None
    ready.allowed.set()
    thread.join(5)
    assert not thread.is_alive()
    out = completed[0]
    assert out.req_ids == ['b', 'a'] and out.req_id_to_index == {'b': 0, 'a': 1}
    assert torch.equal(out.multimodal_outputs[0]['model_outputs'], torch.tensor([1.0, -2.0]))
    assert out.multimodal_outputs[1]['model_outputs'].numel() == 0
    assert out.inter_stage_outputs is None
    assert result._source is None and result.get_output() is out
    assert runner.codec_async_status()['counters'] == {
        'pending_outputs': 0,
        'peak_pending_outputs': 1,
        'pcm_bytes': 8,
        'deferred_outputs': 1,
        'completed_outputs': 1,
    }


def test_two_inflight_outputs_have_independent_host_and_source_ownership(copies):
    first_runner, first_state = runner_state()
    second_runner, second_state = runner_state()
    second_state.multimodal_outputs['model_outputs'][0].fill_(7)
    first = build_codec_async_output(first_runner, first_state)
    second = build_codec_async_output(second_runner, second_state)
    second._ready.allowed.set()
    second_out = second.get_output()
    first._ready.allowed.set()
    first_out = first.get_output()
    assert torch.equal(first_out.multimodal_outputs[0]['model_outputs'], torch.tensor([1.0, -2.0]))
    assert torch.equal(second_out.multimodal_outputs[0]['model_outputs'], torch.tensor([7.0, 7.0]))
    assert (
        first_out.multimodal_outputs[0]['model_outputs'].data_ptr()
        != second_out.multimodal_outputs[0]['model_outputs'].data_ptr()
    )


@pytest.mark.parametrize(
    'change,expected',
    [
        (lambda r: setattr(r, 'speculative_config', object()), 'speculative'),
        (lambda r: setattr(r, 'routed_experts_initialized', True), 'extra_outputs'),
        (lambda r: setattr(r, 'kv_connector_output', object()), 'connector_output'),
        (lambda r: setattr(r, '_should_accumulate_full_payload_output', lambda: True), 'accumulation'),
        (lambda r: setattr(r.vllm_config.parallel_config, 'pipeline_parallel_size', 2), 'parallel'),
        (lambda r: setattr(r.vllm_config.scheduler_config, 'scheduler_cls', 'old'), 'unfenced_scheduler'),
    ],
)
def test_unsupported_contract_falls_back_without_consuming_state(change, expected, monkeypatch):
    runner, state = runner_state()
    change(runner)
    assert codec_async_fallback_reason(runner, state) == expected
    # Avoid a real copy in the preserved legacy path; this test verifies dispatch.
    monkeypatch.setenv('EASYMAGPIE_CODEC_ASYNC_D2H', '1')
    monkeypatch.setattr('easymagpie_vllm_omni.runner.batch_waveforms_to_cpu', lambda x: x)

    def fallback(self, grammar_output=None):
        assert self.execute_model_state is state
        return 'legacy'

    monkeypatch.setattr(GPUGenerationModelRunner, 'sample_tokens', fallback)
    assert runner.sample_tokens() == 'legacy'


def test_poll_fence_delays_final_metadata_until_previous_output_settles():
    adapter, request = _codec_poll_state(
        [{'codes': {'audio': torch.ones(2, 2, dtype=torch.long)}, 'meta': {'finished': True}}]
    )
    request.num_in_flight_tokens = 3
    before = request.additional_information
    assert _poll_fenced_codec_chunk(adapter, request) is False
    assert request.additional_information is before and not adapter.upstream_exhausted_requests
    assert adapter.get_req_chunk[request.request_id] == 0
    request.num_in_flight_tokens = 0
    assert _poll_fenced_codec_chunk(adapter, request) is True
    assert request.request_id in adapter.upstream_exhausted_requests
    assert request.num_computed_tokens == 3 and request.num_prompt_tokens == 5


def test_terminal_inflight_request_is_parked_and_counts_capacity_while_other_request_runs():
    adapter, request = _codec_poll_state([])
    request.num_in_flight_tokens = 3
    request.status = RequestStatus.RUNNING
    other = NS(request_id='other', num_in_flight_tokens=0, status=RequestStatus.RUNNING)
    adapter.upstream_exhausted_requests.add(request.request_id)
    adapter.requests_with_ready_chunks.add(other.request_id)
    adapter.receives_chunks = True
    adapter._cancelled_load_reqs = set()
    adapter._active_window = 0
    adapter.scheduler_max_num_seqs = 128
    adapter._held_non_active = deque()
    adapter.waiting_for_chunk_running_requests = deque()
    adapter.waiting_for_chunk_waiting_requests = deque()
    adapter._codec_async_process_pending_chunks = adapter.process_pending_chunks
    running = [request, other]
    waiting = FCFSRequestQueue()
    requests = {request.request_id: request, other.request_id: other}
    _process_fenced_codec_chunks(adapter, waiting, running, requests)
    assert running == [other]
    assert adapter.num_running_waiting_for_chunk == 1
    assert len(running) + adapter.num_running_waiting_for_chunk == 2
    adapter.restore_queues(waiting, running, scheduler_requests=requests)
    assert running == [other, request]
    # Cancellation while parked must not resurrect the request.
    _process_fenced_codec_chunks(adapter, waiting, running, requests)
    requests.pop(request.request_id)
    adapter.restore_queues(waiting, running, scheduler_requests=requests)
    assert running == [other]


def test_output_settlement_holds_fence_until_finish_accounting_completes(monkeypatch):
    scheduler = object.__new__(EasyMagpieAsyncCodecScheduler)
    lock = threading.Lock()
    scheduler.chunk_transfer_adapter = NS(_easymagpie_chunk_lock=lock)
    request = NS(num_in_flight_tokens=2, terminal=False)
    decremented, attempting, permit_finish = threading.Event(), threading.Event(), threading.Event()

    def parent(self, scheduled, output):
        request.num_in_flight_tokens = 0
        decremented.set()
        assert attempting.wait(5)
        assert not request.terminal
        assert permit_finish.wait(5)
        return 'settled'

    monkeypatch.setattr(EasyMagpieCodecScheduler, 'update_from_output', parent)
    results = []
    update = threading.Thread(target=lambda: results.append(scheduler.update_from_output(None, None)))

    def poll():
        assert decremented.wait(5)
        attempting.set()
        with lock:
            request.terminal = True

    receiver = threading.Thread(target=poll)
    update.start()
    receiver.start()
    assert attempting.wait(5)
    assert not request.terminal
    permit_finish.set()
    update.join(5)
    receiver.join(5)
    assert results == ['settled'] and request.terminal


def test_merged_configuration_preserves_fenced_scheduler_and_async_flag(monkeypatch):
    import importlib

    import easymagpie_vllm_omni.pipeline as pipeline
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    monkeypatch.setenv('EASYMAGPIE_CODEC_ASYNC_D2H', '1')
    importlib.reload(pipeline)
    stages = merge_pipeline_deploy(
        pipeline.EASYMAGPIE_PIPELINE,
        load_deploy_config(
            str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'deploy/h100_optimized128.yaml')
        ),
    )
    codec = next(stage for stage in stages if stage.stage_id == 1).to_omegaconf()
    assert codec.engine_args.scheduler_cls == 'easymagpie_vllm_omni.scheduler.EasyMagpieAsyncCodecScheduler'
    assert codec.engine_args.async_scheduling is True
    assert codec.engine_args.dtype == 'float32'
    monkeypatch.delenv('EASYMAGPIE_CODEC_ASYNC_D2H')
    importlib.reload(pipeline)
    assert (
        pipeline.EASYMAGPIE_PIPELINE.stages[1].scheduler_cls
        == 'easymagpie_vllm_omni.scheduler.EasyMagpieCodecScheduler'
    )
