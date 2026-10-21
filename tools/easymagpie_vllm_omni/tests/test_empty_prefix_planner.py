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
"""Native KV planning and mutation-boundary regression tests."""
import gc
import math
import os
import statistics
import threading
import time
from pathlib import Path

import pytest
import torch
from easymagpie_vllm_omni import empty_prefix_planner as p
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request, RequestStatus

KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec)
KVCacheSpecRegistry.register(MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec)


def fixture(mamba_size=16, attention_size=16, blocks=512):
    m = MambaSpec(
        block_size=mamba_size, shapes=((4,), (4,)), dtypes=(torch.float16, torch.float32), page_size_padded=4096
    )
    a = FullAttentionSpec(
        block_size=attention_size, num_kv_heads=1, head_size=1, dtype=torch.float16, page_size_padded=4096
    )
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f'm{i}'], m) for i in range(5)] + [KVCacheGroupSpec(['fa'], a)],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=math.lcm(mamba_size, attention_size),
        hash_block_size=1,
        enable_caching=False,
    )


def ledger(kv):
    return dict(
        free=kv.block_pool.get_num_free_blocks(),
        groups=[
            dict(
                blocks={k: [(b.block_id, b.ref_cnt, b.is_null) for b in v] for k, v in m.req_to_blocks.items()},
                cached=dict(m.num_cached_block),
                partial=dict(m._partial_hit_reqs),
                new=list(m.new_block_ids),
            )
            for m in kv.coordinator.single_type_managers
        ],
    )


@pytest.mark.parametrize('sizes', [(16, 16), (8, 64), (128, 16)])
def test_actual_prediction_matches_exhaustive_empty_prefix_boundary_math(sizes):
    kv = fixture(*sizes)
    planner = p.Planner(kv, shadow=True)
    empty = kv.empty_kv_cache_blocks.blocks
    for held in (0, 1, 2, 5, 16):
        for m in kv.coordinator.single_type_managers:
            m.req_to_blocks['r'] = [kv.block_pool.null_block] * held
        for total in (0, 1, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129, 4096):
            for tokens in (0, 1, 8, 16, 17, 64, 128, 4096):
                before = ledger(kv)
                actual = kv.coordinator.get_num_blocks_to_allocate('r', tokens, empty, 0, total, total, tokens)
                assert actual == planner.original('r', tokens, empty, 0, total, total, tokens)
                assert ledger(kv) == before
    assert planner.counts['fast_hits'] == planner.counts['shadow_comparisons'] == 560
    assert planner.counts['errors'] == 0


def request(rid):
    return Request(rid, [0] * 71, SamplingParams(max_tokens=256), None)


def test_actual_alloc_free_preempt_and_reused_ID_ledgers_remain_exact():
    control, candidate = fixture(), fixture()
    planner = p.Planner(candidate, shadow=True)
    left, right = request('r'), request('r')
    for generation in range(3):
        for i, n in enumerate([71] + [1] * 65):
            x = control.allocate_slots(left, n)
            y = candidate.allocate_slots(right, n)
            assert (x.get_block_ids() if x else None) == (y.get_block_ids() if y else None)
            assert ledger(control) == ledger(candidate)
            left.num_computed_tokens += n
            right.num_computed_tokens += n
            left.status = right.status = RequestStatus.RUNNING
            left.num_in_flight_tokens = right.num_in_flight_tokens = i % 3
        control.free(left)
        candidate.free(right)
        assert ledger(control) == ledger(candidate)
        left, right = request('r'), request('r')
        if generation == 0:
            left.status = right.status = RequestStatus.PREEMPTED
    assert planner.counts['fast_hits'] > 0 and not planner.counts['errors']


def test_original_capacity_failure_is_preserved():
    left, right = fixture(blocks=37), fixture(blocks=37)
    p.Planner(right, shadow=True)
    for rid in ('first', 'no-capacity'):
        x, y = left.allocate_slots(request(rid), 71), right.allocate_slots(request(rid), 71)
        assert (x is None) == (y is None)
        assert ledger(left) == ledger(right)
    assert x is None


@pytest.mark.parametrize(
    'kind',
    [
        'noncanonical',
        'prefixhit',
        'cached',
        'partial',
        'admission',
        'cacheflag',
        'spec',
        'mode',
        'blocksize',
        'replace_spec',
        'replace_manager',
    ],
)
def test_mutated_cache_and_unsupported_inputs_use_exact_original(kind):
    kv = fixture()
    planner = p.Planner(kv)
    m = kv.coordinator.single_type_managers[0]
    args = ['r', 71, kv.empty_kv_cache_blocks.blocks, 0, 0, 0, 71, False]
    if kind == 'noncanonical':
        args[2] = tuple([] for _ in range(6))
    elif kind == 'prefixhit':
        args[2] = tuple([kv.block_pool.null_block] if i == 5 else [] for i in range(6))
        args[5] = 1
    elif kind == 'cached':
        m.num_cached_block['r'] = 0
    elif kind == 'partial':
        m._partial_hit_reqs['r'] = (0, kv.block_pool.null_block)
    elif kind == 'admission':
        args[7] = True
    elif kind == 'cacheflag':
        kv.enable_caching = True
    elif kind == 'spec':
        m.num_speculative_blocks = 1
    elif kind == 'mode':
        m.mamba_cache_mode = 'all'
    elif kind == 'blocksize':
        m.block_size += 1
    elif kind == 'replace_manager':
        kv.coordinator.single_type_managers = (
            fixture().coordinator.single_type_managers[0],
            *kv.coordinator.single_type_managers[1:],
        )
    elif kind == 'replace_spec':
        m.kv_cache_spec = MambaSpec(block_size=16, shapes=((4,),), dtypes=(torch.float16,))
    expected = planner.original(*args)
    assert kv.coordinator.get_num_blocks_to_allocate(*args) == expected
    assert planner.counts['fallback_calls'] == 1 and planner.counts['fast_hits'] == 0
