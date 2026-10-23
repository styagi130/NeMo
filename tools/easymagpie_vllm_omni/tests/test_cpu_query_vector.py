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
"""Actual runner call-site comparison using only CPU integer metadata."""
import ast
import copy
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'easymagpie_vllm_omni/runner.py'
REFERENCE = SOURCE


def runner_class(path, enabled=False):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EasyMagpieGPUARModelRunner')
    names = {
        '_query_starts_cpu',
        '_maybe_run_batch_preprocess',
        '_try_complete_decode_preprocess',
        '_try_complete_q1_preprocess',
        'get_prepost_status',
    }
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]

    class Base:
        def _maybe_run_batch_preprocess(self, req_ids, device):
            self.base_calls.append((list(req_ids), str(device)))

    cls = ast.ClassDef(
        name=original.name, bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {
        'Base': Base,
        'torch': torch,
        'Any': Any,
        'Counter': Counter,
        'CPU_QUERY_VECTOR_ENABLED': enabled,
        'feedback_merge_status': lambda r: None,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[original.name]


class FakeModel:
    def __init__(self):
        self._device_decode_state = SimpleNamespace(lease_status=lambda: {})
        self._batched_decode_inputs_enabled = True
        self._retain_live_decode_leases = True
        self._prefill_graphs_enabled = True
        self._prepared_decode_updates = {}
        self.calls = []

    def prepare_query_dispatch(self, starts, counts, device):
        self.calls.append(('dispatch', starts, counts, str(device)))

    def prepare_decode_batch(self, rows, device, **kwargs):
        self.calls.append(('decode', rows, kwargs))


@pytest.fixture(autouse=True)
def no_cuda_and_exact_model_type(monkeypatch):
    fake = ModuleType('easymagpie_vllm_omni.easymagpie')
    fake.EasyMagpieTTSForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, 'easymagpie_vllm_omni.easymagpie', fake)

    def forbidden(*a, **kw):
        raise AssertionError('CUDA API invoked in CPU vector path')

    for name in ('synchronize', 'current_stream', 'is_current_stream_capturing', 'init'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    assert not torch.cuda.is_initialized()


def make(path, enabled, counts, host=None):
    r = runner_class(path, enabled)()
    r.model = FakeModel()
    r.get_model = lambda: r.model
    r.base_calls = []
    r.query_start_loc = SimpleNamespace(
        cpu=(
            host
            if host is not None
            else torch.tensor([0] + list(__import__('itertools').accumulate(counts)) + [987, 988], dtype=torch.int32)
        )
    )
    r._omni_num_scheduled_tokens_np = counts
    r.model_config = SimpleNamespace(is_encoder_decoder=False)
    r.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    ids = [f'r{i}' for i in range(len(counts))]
    r.requests = {name: SimpleNamespace(prompt_token_ids=[7, 8, 9]) for name in ids}
    r.model_intermediate_buffer = {name: {'decode_offset': 3} for name in ids}
    r.input_batch = SimpleNamespace(req_ids=ids, num_computed_tokens_cpu=[3] * len(ids))
    r._update_intermediate_buffer = lambda name, values: r.model_intermediate_buffer[name].update(values)
    return r


@pytest.mark.parametrize('counts', [[], [1], [71], [1, 1, 1], [71, 1, 71, 1], [1] * 64])
@pytest.mark.parametrize('enabled', [False, True])
def test_actual_batch_preprocess_matches_disabled_path(counts, enabled):
    a = make(REFERENCE, False, counts)
    b = make(SOURCE, enabled, counts)
    # Share request identities: identity lineage must remain unchanged.
    b.requests = a.requests
    ids = list(a.requests)
    for r in (a, b):
        r._maybe_run_batch_preprocess(ids, torch.device('cpu'))
    assert a.base_calls == b.base_calls and a.model.calls == b.model.calls
    assert a.model_intermediate_buffer == b.model_intermediate_buffer
    status = b.get_prepost_status()['cpu_query_vector']
    assert status['enabled'] == enabled
    if enabled:
        assert status['batch_preprocess_calls'] == 1 and status['batch_preprocess_rows'] == len(counts)
    else:
        assert status == {'enabled': False}


@pytest.mark.parametrize('count', [0, 1, 3, 64])
@pytest.mark.parametrize('eligible', [False, True])
def test_actual_decode_fastpath_preserves_acceptance_updates_padding(count, eligible):
    models = [make(REFERENCE, False, [1] * count), make(SOURCE, True, [1] * count)]
    outputs = []
    for r in models:
        r.model._batched_decode_inputs_enabled = eligible
        r.model._prepared_decode_updates = {i: {'decode_offset': 4} for i in range(count)}
        inputs = torch.arange(count + 5, dtype=torch.long)
        embeds = torch.full((count + 5, 4), 123.0)
        result = r._try_complete_decode_preprocess(inputs, embeds, [1] * count)
        outputs.append((result, inputs, embeds, r.model_intermediate_buffer, r.model._prepared_decode_updates))
    assert outputs[0][0] == outputs[1][0] == bool(count and eligible)
    for index in [1, 2]:
        torch.testing.assert_close(outputs[0][index], outputs[1][index], atol=0, rtol=0)
    assert outputs[0][3:] == outputs[1][3:]
    status = models[1].get_prepost_status()['cpu_query_vector']
    assert status.get('decode_fastpath_calls', 0) == int(bool(count and eligible))


def test_multimodal_capability_does_not_disable_completed_prompt_decode():
    runner = make(SOURCE, True, [1, 1])
    runner.supports_mm_inputs = True
    runner.model._prepared_decode_updates = {index: {'decode_offset': 4} for index in range(2)}
    embeds = torch.full((2, 4), 123.0)
    assert runner._try_complete_decode_preprocess(torch.zeros(2, dtype=torch.long), embeds, [1, 1])
    assert torch.equal(embeds, torch.zeros_like(embeds))
    assert not runner.model._prepared_decode_updates


@pytest.mark.parametrize(
    'host,count',
    [
        (torch.tensor([0, 2, 5], dtype=torch.int64), 3),
        ([0, 2, 5], 3),
        (torch.tensor([0.0, 2.9, 5.4]), 3),
        (torch.tensor([[0], [2], [5]], dtype=torch.int32), 3),
    ],
)
def test_unexpected_cpu_buffer_frontends_keep_legacy_conversion(host, count):
    r = make(SOURCE, True, [], host=host)
    assert r._query_starts_cpu(count, 'test') == [int(host[i]) for i in range(count)]
    assert r.get_prepost_status()['cpu_query_vector']['fallback_calls'] == 1


def test_short_buffer_preserves_index_error():
    r = make(SOURCE, True, [], host=torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(IndexError):
        r._query_starts_cpu(3, 'test')
    assert r.get_prepost_status()['cpu_query_vector']['fallback_calls'] == 1


def test_partial_lists_are_independent_of_tensor_and_each_other():
    host = torch.tensor([0, 71, 72, 90, 12345], dtype=torch.int32)
    r = make(SOURCE, True, [], host=host)
    a = r._query_starts_cpu(3, 'test')
    b = r._query_starts_cpu(3, 'test')
    assert a == b == [0, 71, 72] and a is not b
    a[1] = -1
    assert b == [0, 71, 72] and host.tolist() == [0, 71, 72, 90, 12345]
    host[0] = 999
    assert b == [0, 71, 72]
    c = r._query_starts_cpu(0, 'test')
    d = r._query_starts_cpu(0, 'test')
    assert c == d == [] and c is not d


def test_actual_module_flag_defaults_off(monkeypatch):
    import os

    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'CPU_QUERY_VECTOR_ENABLED' for t in n.targets)
    )
    code = compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec')
    monkeypatch.delenv('EASYMAGPIE_CPU_QUERY_VECTOR', raising=False)
    namespace = {'os': os}
    exec(code, namespace)
    assert namespace['CPU_QUERY_VECTOR_ENABLED'] is False
    monkeypatch.setenv('EASYMAGPIE_CPU_QUERY_VECTOR', '1')
    exec(code, namespace)
    assert namespace['CPU_QUERY_VECTOR_ENABLED'] is True
