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
import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from easymagpie_vllm_omni.codec import cuda_graph as candidate

baseline = candidate


class CudaCodes:
    is_cuda = True
    dtype = torch.long
    ndim = 2
    shape = (12, 16)
    device = torch.device('cuda:0')

    def is_contiguous(self):
        return True

    def stride(self):
        return (16, 1)


class Cache:
    is_cuda = True
    shape = (560, 4608)
    dtype = torch.float32
    device = torch.device('cuda:0')
    pointer = 12345

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return 560 * 4608

    def stride(self):
        return (4608, 1)


def setup(module, enabled=True):
    os.environ['EASYMAGPIE_VLLM_CODEC_SHARED_METADATA_GUARD'] = '1' if enabled else '0'
    g = module.CodecCudaGraphs(NS(modules=lambda: []), None)
    g.layers = tuple(NS(prefix='layer' + str(i), kv_cache=[Cache()]) for i in range(26))
    m = module.uniform_codec_metadata(2, 6, torch.device('cpu'))
    raw = {layer.prefix: m for layer in g.layers}
    return g, m, raw, CudaCodes()


@pytest.fixture(autouse=True)
def no_capture(monkeypatch):
    monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_SHARED_METADATA_GUARD', '0')
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)


def outcome(module, mutate=lambda g, m, r, c: None, enabled=True):
    g, m, raw, codes = setup(module, enabled)
    mutate(g, m, raw, codes)
    result = g._metadata(codes, raw)
    return result[1], g, m, raw, codes


def test_shared_aliases_are_identical_to_baseline():
    for enabled in [False, True]:
        reason, g, m, r, c = outcome(candidate, enabled=enabled)
        assert reason is None and g._metadata(c, r)[0] is m
        assert g.metadata_same_object_skips == (52 if enabled else 0)
        assert outcome(baseline, enabled=False)[0] is None


def test_distinct_equivalent_objects_preserve_full_validation():
    def mutate(g, m, raw, c):
        for key in raw:
            raw[key] = copy.copy(m)

    for enabled in [True, False]:
        reason, g, m, raw, c = outcome(candidate, mutate, enabled=enabled)
        assert reason is None
    assert g._metadata(c, raw)[0] is raw['layer0']


@pytest.mark.parametrize('field', candidate._SCALAR_FIELDS)
def test_distinct_scalar_mismatch_falls_back(field):
    def mutate(g, m, raw, c):
        other = copy.copy(m)
        value = getattr(other, field)
        setattr(other, field, not value if isinstance(value, bool) else value + 1)
        raw['layer17'] = other

    assert outcome(candidate, mutate)[0] == outcome(baseline, mutate, enabled=False)[0] == 'per_layer_shape'


@pytest.mark.parametrize('field', candidate._TENSOR_FIELDS[1:])
def test_distinct_nonidentical_routing_falls_back(field):
    def mutate(g, m, raw, c):
        other = copy.copy(m)
        setattr(other, field, getattr(m, field).clone())
        raw['layer13'] = other

    assert outcome(candidate, mutate)[0] == outcome(baseline, mutate, enabled=False)[0] == 'per_layer_routing'


@pytest.mark.parametrize(
    'mutation,reason',
    [
        (lambda g, m, r, c: setattr(m, 'codec_uniform', False), 'ragged'),
        (lambda g, m, r, c: setattr(m, 'codec_prefill_uniform', False), 'unknown_prefill_shape'),
        (lambda g, m, r, c: setattr(m, 'num_prefill_tokens', 13), 'token_count'),
        (lambda g, m, r, c: setattr(m, 'num_decodes', 1), 'mixed_or_empty'),
        (lambda g, m, r, c: r.__setitem__('layer4', None), 'missing_layer_metadata'),
        (lambda g, m, r, c: setattr(g.layers[7].kv_cache[0], 'is_cuda', False), 'unbound_cache'),
    ],
)
def test_mutations_at_each_validation_boundary_remain_visible(mutation, reason):
    assert outcome(candidate, mutation)[0] == outcome(baseline, mutation, enabled=False)[0] == reason


def test_shared_metadata_mutation_between_calls_not_cached():
    g, m, raw, c = setup(candidate)
    assert g._metadata(c, raw)[1] is None
    m.num_prefill_tokens = 13
    assert g._metadata(c, raw)[1] == 'token_count'
    m.num_prefill_tokens = 12
    assert g._metadata(c, raw)[1] is None
    m.codec_uniform = False
    assert g._metadata(c, raw)[1] == 'ragged'


def test_cache_key_still_rechecks_in_place_pointer_and_layout_change():
    for enabled in [True, False]:
        g, m, raw, c = setup(candidate, enabled=enabled)
        key = g._key(c, m)
        g.layers[4].kv_cache[0].pointer += 1
        assert g._key(c, m) != key
        key = g._key(c, m)
        g.layers[4].kv_cache[0].shape = (1120, 2304)
        assert g._key(c, m) != key


def test_key_rechecks_changed_shared_routing_shape_and_metadata_scalar():
    for enabled in [True, False]:
        g, m, r, c = setup(candidate, enabled=enabled)
        key = g._key(c, m)
        m.state_indices_tensor_p = m.state_indices_tensor_p[:, None]
        assert g._key(c, m) != key
        key = g._key(c, m)
        m.codec_max_query_len = 8
        assert g._key(c, m) != key


def test_default_disabled_and_status_is_explicit(monkeypatch):
    monkeypatch.delenv('EASYMAGPIE_VLLM_CODEC_SHARED_METADATA_GUARD', raising=False)
    g = candidate.CodecCudaGraphs(NS(modules=lambda: []), None)
    assert g.status()['shared_metadata_guard'] == {
        'enabled': False,
        'same_object_checks_skipped': 0,
        'calls_with_shared_metadata': 0,
    }


def test_no_cuda_initialized():
    assert not torch.cuda.is_initialized()
