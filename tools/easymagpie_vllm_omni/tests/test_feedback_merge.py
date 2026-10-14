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
"""Exercise the actual runner and async eager-postprocess entry point."""
import gc
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
from test_feedback_snapshot import setup as original_setup
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


def setup(device='cpu', enabled=True, phonemes=2, names=('a', 'b', 'c'), live=(0, 2)):
    model, runner, _ = original_setup(phonemes)
    for name in ('_out_codes', '_dec_phoneme_tokens'):
        if hasattr(model, name):
            setattr(model, name, getattr(model, name).to(device))
    state = model._device_decode_state
    for name, value in vars(state).items():
        if isinstance(value, torch.Tensor):
            setattr(state, name, value.to(device))
    state.device = torch.device(device)
    state.row_slots_host = {row: index for index, row in enumerate(live)}
    state.row_slots.copy_(state.inactive_slots)
    for row, slot in state.row_slots_host.items():
        state.row_slots[row] = slot
    state.ended.zero_()
    if live:
        state.ended[0] = True
    model._batched_feedback_merge_enabled = enabled
    runner.input_batch = NS(req_ids=list(names))
    runner.requests = {name: NS() for name in names}
    runner.model_intermediate_buffer = {
        name: {'decode_offset': 10 + i, 'text_tokens': [10, 11], 'keep': object()} for i, name in enumerate(names)
    }
    runner.query_start_loc = NS(cpu=torch.arange(len(names) + 1))
    hidden = torch.zeros(32, 3, device=device)
    output = model.make_omni_output(hidden)
    return model, runner, hidden, output


def apply(runner, hidden, output, **kwargs):
    return runner._process_additional_information_updates(
        hidden, output.multimodal_outputs, np.ones(len(runner.input_batch.req_ids), dtype=np.int32), NS(), **kwargs
    )


def expected(model, runner, hidden, output):
    return {
        name: model.postprocess(hidden[i : i + 1], output.multimodal_outputs)
        for i, name in enumerate(runner.input_batch.req_ids)
    }


def assert_values(runner, wanted):
    for name, values in wanted.items():
        stored = runner.model_intermediate_buffer[name]
        for key, value in values.items():
            torch.testing.assert_close(stored[key], value, rtol=0, atol=0)
        assert runner.requests[name].additional_information_cpu is stored


@pytest.mark.parametrize('names', [('a', 'b', 'c'), ('c', 'a', 'b')])
@pytest.mark.parametrize('phonemes', [0, 2])
def test_batch_merge_exact_rows_and_eos_preserves_unrelated_state(names, phonemes):
    model, runner, hidden, output = setup(names=names, phonemes=phonemes)
    wanted = expected(model, runner, hidden, output)
    keep = {name: dict(info) for name, info in runner.model_intermediate_buffer.items()}
    apply(runner, hidden, output)
    assert runner.get_prepost_status()['feedback_merge']['hits'] == 1
    assert_values(runner, wanted)
    for name, info in keep.items():
        for key, value in info.items():
            assert runner.model_intermediate_buffer[name][key] is value
    if phonemes:
        assert runner.model_intermediate_buffer[names[0]]['phoneme_ended'].item()
        assert 'phoneme_ended' not in runner.model_intermediate_buffer[names[1]]


def test_real_async_eager_hook_with_full_filter_hits_and_preserves_return():
    model, runner, hidden, output = setup()
    runner._resolve_pooler_payload_req_ids = lambda names: ([], names)
    result = runner._maybe_run_eager_omni_postprocess_before_async_output(
        hidden_states=hidden,
        multimodal_outputs=output.multimodal_outputs,
        num_scheduled_tokens_np=np.ones(3, dtype=np.int32),
        scheduler_output=NS(),
        req_ids_output_copy=['a', 'b', 'c'],
        query_start_loc_cpu=torch.arange(4),
    )
    assert result is True
    assert runner.get_prepost_status()['feedback_merge']['hits'] == 1
    assert_values(runner, expected(model, runner, hidden, output))


def test_snapshot_survives_scratch_reuse_async_delay_eviction_and_gc():
    model, runner, hidden, output = setup()
    wanted = {
        name: {key: value.clone() for key, value in values.items()}
        for name, values in expected(model, runner, hidden, output).items()
    }
    apply(runner, hidden, output)
    old_audio = output.multimodal_outputs['codes']['audio']
    ref = weakref.ref(old_audio)
    ptr = runner.model_intermediate_buffer['a']['last_audio_codes'].data_ptr()
    model._out_codes.fill_(777)
    model._dec_phoneme_tokens.fill_(778)
    model._device_decode_state.ended.zero_()
    second = model.make_omni_output(hidden)
    del output, old_audio
    gc.collect()
    assert ref() is not None  # Immutable row view owns its original allocation.
    assert_values(runner, wanted)
    state = model._device_decode_state
    info = dict(runner.model_intermediate_buffer['a'], decode_offset=8, text_tokens=[10, 11])
    model.prepare_decode_batch([(17, info)], torch.device('cpu'), request_keys=[('a', 9)])
    slot = state.row_slots_host[17]
    torch.testing.assert_close(state.audio[slot], wanted['a']['last_audio_codes'].reshape(-1))
    assert state.ended[slot].item()
    assert runner.model_intermediate_buffer['a']['last_audio_codes'].data_ptr() == ptr
    del second


@pytest.mark.parametrize(
    'case',
    [
        'disabled',
        'partial_filter',
        'combined_hidden',
        'combined_mm',
        'noncontiguous_hidden',
        'offset_hidden',
        'missing_request',
        'duplicate_ids',
        'wrong_order',
        'mixed_counts',
        'noncontiguous_spans',
        'gpu_metadata',
        'uncertified_audio',
        'changed_snapshot',
        'wrong_tensor_owner',
        'shape_changed',
    ],
)
def test_guard_falls_back_without_partial_writes(monkeypatch, case):
    model, runner, hidden, output = setup()
    kwargs = {}
    counts = np.ones(3, dtype=np.int32)
    if case == 'disabled':
        model._batched_feedback_merge_enabled = False
    elif case == 'partial_filter':
        kwargs['req_ids_filter'] = {'b'}
    elif case == 'combined_hidden':
        kwargs['combined_hidden_states'] = {}
    elif case == 'combined_mm':
        kwargs['combined_multimodal_outputs'] = {}
    elif case == 'noncontiguous_hidden':
        hidden = hidden[:, ::2]
    elif case == 'offset_hidden':
        hidden = hidden[1:]
    elif case == 'missing_request':
        runner.requests.pop('c')
    elif case == 'duplicate_ids':
        runner.input_batch.req_ids = ['a', 'a', 'c']
    elif case == 'wrong_order':
        kwargs['req_ids'] = ['c', 'b', 'a']
    elif case == 'mixed_counts':
        counts = np.array([1, 2, 1], dtype=np.int32)
    elif case == 'noncontiguous_spans':
        kwargs['query_start_loc_cpu'] = [0, 2, 3, 4]
    elif case == 'gpu_metadata':
        kwargs['query_start_loc_cpu'] = NS(device=NS(type='cuda'))
    elif case == 'uncertified_audio':
        output.multimodal_outputs['codes']['audio'] = model._out_codes.clone()
    elif case == 'changed_snapshot':
        output.multimodal_outputs['codes']['audio']._easymagpie_feedback_snapshot = object()
    elif case == 'wrong_tensor_owner':
        snapshot = output.multimodal_outputs['codes']['audio']._easymagpie_feedback_snapshot
        snapshot.phonemes = snapshot.phonemes.clone()
    elif case == 'shape_changed':
        output.multimodal_outputs['codes']['audio'].transpose_(0, 1)
    before = {key: dict(value) for key, value in runner.model_intermediate_buffer.items()}
    calls = []

    def generic(*args, **kwargs):
        calls.append(True)
        assert runner.model_intermediate_buffer == before
        return 'generic-sentinel'

    monkeypatch.setattr(OmniGPUModelRunner, '_process_additional_information_updates', generic)
    value = runner._process_additional_information_updates(hidden, output.multimodal_outputs, counts, NS(), **kwargs)
    assert value == 'generic-sentinel' and calls == [True]
    assert not runner.get_prepost_status()['feedback_merge']['hits']


def test_partial_filter_fallback_matches_actual_base_and_keeps_other_requests():
    model, runner, hidden, output = setup()
    apply(runner, hidden, output, req_ids_filter={'b'})
    assert 'last_audio_codes' not in runner.model_intermediate_buffer['a']
    assert 'last_audio_codes' not in runner.model_intermediate_buffer['c']
    torch.testing.assert_close(runner.model_intermediate_buffer['b']['last_audio_codes'], model._out_codes[1:2])
    assert not runner.get_prepost_status()['feedback_merge']['hits']


def test_snapshot_and_owner_have_no_cycle_without_stored_views():
    _, _, _, output = setup()
    audio = output.multimodal_outputs['codes']['audio']
    snapshot = audio._easymagpie_feedback_snapshot
    audio_ref, snapshot_ref = weakref.ref(audio), weakref.ref(snapshot)
    del audio, snapshot, output
    assert audio_ref() is None and snapshot_ref() is None


@pytest.mark.parametrize('active,padded', [(1, 1), (64, 64), (65, 128), (128, 128)])
def test_cuda_actual_async_hook_captured_snapshot_ownership(active, padded):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from easymagpie_vllm_omni.device_decode_state import DeviceDecodeState
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    model, fast, _, _ = setup(device='cuda')
    model._out_codes = torch.zeros(160, 4, device='cuda', dtype=torch.long)
    model._dec_phoneme_tokens = torch.zeros(160, 2, device='cuda', dtype=torch.long)
    model._dec_audio_codes = torch.zeros(160, 4, device='cuda', dtype=torch.long)
    model._device_decode_state = state = DeviceDecodeState(model, 128, 32, 160)
    state.row_slots_host = {i: i for i in range(active)}
    state.row_slots[:active] = torch.arange(active, device='cuda')
    audio_source = torch.arange(160 * 4, device='cuda').reshape(160, 4) % 32
    phoneme_source = torch.arange(160 * 2, device='cuda').reshape(160, 2) % 16
    eos_source = torch.arange(state.ended.numel(), device='cuda') % 3 == 0
    hidden = torch.zeros(padded, 3, device='cuda')

    def produce():
        model._out_codes.copy_(audio_source)
        model._dec_phoneme_tokens.copy_(phoneme_source)
        state.ended.copy_(eos_source)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        produce()
        produce()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        produce()
    saved = []
    for iteration in range(6):
        count = active if iteration != 3 else max(1, active // 2)
        ids = [f'req-{i}' for i in range(count)]
        ids = ids[iteration % count :] + ids[: iteration % count]
        state.row_slots_host = {i: i for i in range(count) if i % 4 != 1}
        graph.replay()
        output = model.make_omni_output(hidden)
        # Queue a later producer before either consumer; old feedback must use
        # the immutable snapshot, never the mutable graph/state buffers.
        audio_source.add_(1)
        phoneme_source.add_(1)
        eos_source.logical_not_()
        graph.replay()
        generic = object.__new__(EasyMagpieGPUARModelRunner)
        generic.model = model
        for runner in (fast, generic):
            runner.input_batch = NS(req_ids=ids)
            runner.requests = {key: NS() for key in ids}
            runner.model_intermediate_buffer = {
                key: {'decode_offset': iteration, 'phoneme_ended': False} for key in ids
            }
            runner.query_start_loc = NS(cpu=torch.arange(count + 1))
        fast._resolve_pooler_payload_req_ids = lambda names: ([], names)
        applied = fast._maybe_run_eager_omni_postprocess_before_async_output(
            hidden_states=hidden,
            multimodal_outputs=output.multimodal_outputs,
            num_scheduled_tokens_np=np.ones(count, dtype=np.int32),
            scheduler_output=NS(),
            req_ids_output_copy=ids,
            query_start_loc_cpu=torch.arange(count + 1),
        )
        assert applied and fast.get_prepost_status()['feedback_merge']['hits'] == iteration + 1
        OmniGPUModelRunner._process_additional_information_updates(
            generic,
            hidden,
            output.multimodal_outputs,
            np.ones(count, dtype=np.int32),
            NS(),
            req_ids_filter=set(ids),
            req_ids=ids,
            query_start_loc_cpu=torch.arange(count + 1),
        )
        torch.cuda.synchronize()
        for key in ids:
            actual, reference = fast.model_intermediate_buffer[key], generic.model_intermediate_buffer[key]
            assert actual.keys() == reference.keys()
            for name, value in reference.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
                else:
                    assert actual[name] == value
        saved.append(
            (
                fast.model_intermediate_buffer,
                {
                    key: {
                        name: value.clone() if isinstance(value, torch.Tensor) else value
                        for name, value in info.items()
                    }
                    for key, info in fast.model_intermediate_buffer.items()
                },
            )
        )
        del output, generic
        gc.collect()
    for actual, expected_rows in saved:
        for key, info in expected_rows.items():
            for name, value in info.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key][name], value, rtol=0, atol=0)
    assert fast.get_prepost_status()['feedback_merge']['requests'] == 5 * active + max(1, active // 2)


@pytest.mark.parametrize('payload_kind', ['flattened', 'single_stage', 'legacy'])
def test_payload_forms_preserve_the_certified_original_tensor(payload_kind):
    model, runner, hidden, output = setup()
    audio = output.multimodal_outputs['codes']['audio']
    key = {'flattened': 'codes.audio', 'single_stage': 'model_outputs', 'legacy': 'audio_codes'}[payload_kind]
    output.multimodal_outputs.clear()
    output.multimodal_outputs[key] = audio
    apply(runner, hidden, output)
    assert runner.get_prepost_status()['feedback_merge']['hits'] == 1
    assert runner.model_intermediate_buffer['a']['last_audio_codes'].data_ptr() == audio.data_ptr()
