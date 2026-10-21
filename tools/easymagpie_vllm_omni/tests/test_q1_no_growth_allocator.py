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
"""Actual image KV managers and original Scheduler CPU qualification.

No allocator/group/pool methods or numerical operations are mocked. The reused
Scheduler fixture isolates transport only; its schedule/KV/async accounting is
the original pinned runtime. Ledger comparisons include every manager/pool
field, every block/refcount/hash/free link, cache/event/statistic and return ID.
"""
import ast
import dataclasses
import enum
import inspect
import math
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Omni plugin bootstrap replaces the public Request constructor. Match the
# actual EngineCore constructor after that bootstrap, not a stale early alias.
import vllm.v1.request as current_request
from easymagpie_vllm_omni import q1_no_growth_allocator as H
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, resolve_kv_cache_block_sizes
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import RequestStatus

Request = EngineCore.preprocess_add_request.__globals__['Request']
assert Request is current_request.Request

KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec)
KVCacheSpecRegistry.register(MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec)


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def config(blocks=2048, mamba_size=4096, attention_size=400, mode='none'):
    m = MambaSpec(
        block_size=mamba_size,
        shapes=((4,), (4,)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=4096,
        mamba_cache_mode=mode,
    )
    a = FullAttentionSpec(
        block_size=attention_size, num_kv_heads=1, head_size=1, dtype=torch.float16, page_size_padded=4096
    )
    return KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f'm{i}'], m) for i in range(5)] + [KVCacheGroupSpec(['fa'], a)],
    )


def manager(
    *, blocks=2048, max_model_len=8192, mamba_size=4096, attention_size=400, mode='none', events=True, stats=True
):
    return KVCacheManager(
        config(blocks, mamba_size, attention_size, mode),
        max_model_len=max_model_len,
        scheduler_block_size=math.lcm(mamba_size, attention_size),
        hash_block_size=1,
        enable_caching=False,
        enable_kv_cache_events=events,
        log_stats=stats,
    )


def frozen(value, active=None):
    """Normalize identity-independent original state; never omit mutable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ('bytes', value.hex())
    if isinstance(value, KVCacheBlock):
        return ('block', value.block_id)
    if isinstance(value, enum.Enum):
        return (type(value).__qualname__, value.name)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        return (
            'tensor',
            str(value.dtype),
            tuple(value.shape),
            value.detach().contiguous().reshape(-1).view(torch.uint8).tolist(),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(frozen(x, active) for x in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted((frozen(x, active) for x in value), key=repr)))
    if isinstance(value, dict):
        factory = getattr(value, 'default_factory', None)
        return (
            type(value).__qualname__,
            frozen(factory, active),
            tuple(sorted(((frozen(k, active), frozen(v, active)) for k, v in value.items()), key=repr)),
        )
    if callable(value):
        return (
            'callable',
            getattr(value, '__module__', None),
            getattr(value, '__qualname__', type(value).__qualname__),
        )
    active = set() if active is None else active
    if id(value) in active:
        raise AssertionError('unexpected non-block state cycle: ' + type(value).__qualname__)
    active.add(id(value))
    try:
        if hasattr(type(value), '__struct_fields__'):
            items = {name: getattr(value, name) for name in type(value).__struct_fields__}
        elif dataclasses.is_dataclass(value):
            items = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        elif hasattr(value, '__dict__'):
            items = vars(value)
        else:
            slots = getattr(type(value), '__slots__', ())
            if isinstance(slots, str):
                slots = (slots,)
            items = {name: getattr(value, name) for name in slots if name != '__weakref__' and hasattr(value, name)}
            assert items, ('unreviewed ledger type', type(value))
        return (type(value).__module__, type(value).__qualname__, frozen(items, active))
    finally:
        active.remove(id(value))


def ledger(kv):
    pool = kv.block_pool
    coordinator = kv.coordinator
    return dict(
        # allocate_slots is the sole intentional instance delegate difference.
        kv=frozen(
            {
                key: value
                for key, value in vars(kv).items()
                if key not in ('coordinator', 'block_pool', 'allocate_slots')
            }
        ),
        coordinator=frozen(
            {
                key: value
                for key, value in vars(coordinator).items()
                if key not in ('single_type_managers', 'block_pool', 'get_num_blocks_to_allocate')
            }
        ),
        groups=[
            frozen({key: value for key, value in vars(m).items() if key not in ('block_pool', '_null_block')})
            for m in coordinator.single_type_managers
        ],
        pool=frozen({key: value for key, value in vars(pool).items() if key not in ('blocks', 'free_block_queue')}),
        all_blocks=[
            frozen({field.name: getattr(block, field.name) for field in dataclasses.fields(block)})
            for block in pool.blocks
        ],
        free_queue=frozen(vars(pool.free_block_queue)),
        free_sentinels={
            key: frozen({field.name: getattr(value, field.name) for field in dataclasses.fields(value)})
            for key, value in vars(pool.free_block_queue).items()
            if isinstance(value, KVCacheBlock)
        },
        free_order=[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        free_count=pool.get_num_free_blocks(),
        usage=kv.usage,
        canonical_empty=frozen(kv.empty_kv_cache_blocks),
    )


def request(rid='r', computed=0):
    # The Omni bootstrap may update the public constructor during collection.
    # Resolve the same constructor that a new EngineCore request uses now.
    constructor = EngineCore.preprocess_add_request.__globals__['Request']
    assert constructor is current_request.Request
    value = constructor(rid, list(range(71)), SamplingParams(max_tokens=8192), None)
    value.num_computed_tokens = computed
    return value


def request_state(r):
    return frozen(
        {
            key: getattr(r, key)
            for key in (
                'request_id',
                'status',
                'num_computed_tokens',
                'num_tokens',
                'num_tokens_with_spec',
                'num_output_placeholders',
                'num_in_flight_tokens',
                'num_preemptions',
                'resumable',
                'prompt_token_ids',
                'output_token_ids',
            )
        }
    )


def returned(value, kv):
    return dict(
        is_none=value is None,
        canonical_empty=value is kv.empty_kv_cache_blocks,
        blocks=None if value is None else frozen(value.blocks),
    )


def prime(kv, r, tokens=71):
    assert kv.allocate_slots(r, tokens) is not None
    r.num_computed_tokens = tokens
    r.status = RequestStatus.RUNNING


def paired(left, right, lr, rr, tokens=1, **kwargs):
    before_left, before_right = request_state(lr), request_state(rr)
    x, y = left.allocate_slots(lr, tokens, **kwargs), right.allocate_slots(rr, tokens, **kwargs)
    assert returned(x, left) == returned(y, right)
    assert ledger(left) == ledger(right)
    assert request_state(lr) == before_left and request_state(rr) == before_right
    return x, y


def test_actual_no_growth_preserves_all_ledgers_and_canonical_empty():
    left, right = manager(), manager()
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    assert type(rr) is candidate.types[6]
    for computed, in_flight, placeholders in [(71, 0, 0), (72, 1, 2), (150, 7, 8), (398, 2, 3), (399, 0, 1)]:
        for r in (lr, rr):
            r.num_computed_tokens = computed
            r.num_in_flight_tokens = in_flight
            r.num_output_placeholders = placeholders
        before = ledger(right)
        x, y = paired(left, right, lr, rr)
        assert x is left.empty_kv_cache_blocks and y is right.empty_kv_cache_blocks
        assert ledger(right) == before
    assert candidate.counts['fast_hits'] == 5 and candidate.counts['errors'] == 0, dict(
        counts=dict(candidate.counts), fallbacks=dict(candidate.fallbacks), layout=candidate.layout_reason_at_install
    )
    done = candidate.close()
    assert done['closed'] and done['binding_restored'] and done['retained_Request_references'] == 0
    assert right.allocate_slots.__func__ is KVCacheManager.allocate_slots


@pytest.mark.parametrize('computed', [400, 800, 4096, 4097])
def test_actual_attention_and_mamba_boundaries_use_original_before_mutation(computed):
    left, right = manager(), manager()
    lr, rr = request(), request()
    prime(left, lr, computed)
    prime(right, rr, computed)
    candidate = H.NoGrowthAllocator(right)
    paired(left, right, lr, rr)
    assert candidate.counts['fast_hits'] == 0 and candidate.counts['fallback_calls'] == 1
    candidate.close()


@pytest.mark.parametrize(
    'kwargs,tokens',
    [
        ({'num_new_computed_tokens': 1}, 1),
        ({'new_computed_blocks': 'canonical'}, 1),
        ({'new_computed_blocks': 'noncanonical'}, 1),
        ({'num_lookahead_tokens': 1}, 1),
        ({'num_external_computed_tokens': 1}, 1),
        ({'delay_cache_blocks': True}, 1),
        ({'num_encoder_tokens': 1}, 1),
        ({'full_sequence_must_fit': True}, 1),
        ({'reserved_blocks': 1}, 1),
        ({'has_scheduled_reqs': False}, 1),
        ({}, 2),
        ({}, 0),
    ],
)
def test_unsupported_optional_semantics_follow_original_return_or_exception(kwargs, tokens):
    left, right = manager(), manager()
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    kwleft = dict(kwargs)
    kwright = dict(kwargs)
    if kwargs.get('new_computed_blocks') == 'canonical':
        kwleft['new_computed_blocks'] = left.empty_kv_cache_blocks
        kwright['new_computed_blocks'] = right.empty_kv_cache_blocks
    elif kwargs.get('new_computed_blocks') == 'noncanonical':
        kwleft['new_computed_blocks'] = KVCacheBlocks(tuple([] for _ in range(6)))
        kwright['new_computed_blocks'] = KVCacheBlocks(tuple([] for _ in range(6)))
    outcomes = []
    for kv, r, kw in ((left, lr, kwleft), (right, rr, kwright)):
        try:
            outcomes.append(('return', returned(kv.allocate_slots(r, tokens, **kw), kv)))
        except Exception as exc:
            outcomes.append(('error', type(exc).__module__, type(exc).__qualname__, str(exc)))
    assert outcomes[0] == outcomes[1] and ledger(left) == ledger(right)
    assert candidate.counts['fast_hits'] == 0 and candidate.counts['fallback_calls'] == 1
    candidate.close()


@pytest.mark.parametrize('kind', ['waiting', 'preempted', 'resumable', 'cached', 'partial', 'missing_request'])
def test_request_lifetime_or_prefix_state_never_uses_fast_path(kind):
    left, right = manager(), manager()
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    if kind == 'waiting':
        lr.status = rr.status = RequestStatus.WAITING
    elif kind == 'preempted':
        lr.num_preemptions = rr.num_preemptions = 1
    elif kind == 'resumable':
        lr.resumable = rr.resumable = True
    elif kind == 'cached':
        for kv in (left, right):
            kv.coordinator.single_type_managers[0].num_cached_block['r'] = 0
    elif kind == 'partial':
        for kv in (left, right):
            m = kv.coordinator.single_type_managers[0]
            m._partial_hit_reqs['r'] = (0, m.req_to_blocks['r'][0])
    else:
        left.free(lr)
        right.free(rr)
    outcomes = []
    for kv, r in ((left, lr), (right, rr)):
        try:
            outcomes.append(('return', returned(kv.allocate_slots(r, 1), kv)))
        except Exception as exc:
            outcomes.append(('error', type(exc).__module__, type(exc).__qualname__, str(exc)))
    assert outcomes[0] == outcomes[1] and ledger(left) == ledger(right)
    assert candidate.counts['fast_hits'] == 0 and candidate.counts['fallback_calls'] == 1
    candidate.close()


def test_exhausted_pool_reserved_capacity_and_reused_ID_exact():
    left, right = manager(blocks=7), manager(blocks=7)
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    assert right.block_pool.get_num_free_blocks() == 0
    paired(left, right, lr, rr)  # No-growth remains legal even with no free blocks.
    x, y = paired(left, right, lr, rr, reserved_blocks=1)
    assert x is y is None
    left.free(lr)
    right.free(rr)
    assert ledger(left) == ledger(right)
    # An externally reused ID must obtain pages from the original allocator.
    lr, rr = request(), request()
    paired(left, right, lr, rr, tokens=71)
    lr.status = rr.status = RequestStatus.RUNNING
    lr.num_computed_tokens = rr.num_computed_tokens = 71
    paired(left, right, lr, rr)
    candidate.close()
    left.free(lr)
    right.free(rr)
    assert ledger(left) == ledger(right)


def test_maximum_model_length_clamping_remains_original():
    left, right = manager(max_model_len=399), manager(max_model_len=399)
    lr, rr = request(), request()
    prime(left, lr, 399)
    prime(right, rr, 399)
    candidate = H.NoGrowthAllocator(right)
    paired(left, right, lr, rr)
    candidate.close()


@pytest.mark.parametrize('options', [{'mamba_size': 2048}, {'mode': 'all'}])
def test_unsupported_static_geometry_is_original_fallback_only(options):
    left, right = manager(**options), manager(**options)
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    paired(left, right, lr, rr)
    assert candidate.status()['eligible_layout'] is False
    assert candidate.counts['fast_hits'] == 0 and candidate.counts['fallback_calls'] == 1
    candidate.close()


@pytest.mark.parametrize('kind', ['prefix_enabled', 'replacement_empty'])
def test_dynamic_prefix_or_canonical_empty_change_uses_original_before_mutation(kind):
    left, right = manager(), manager()
    lr, rr = request(), request()
    prime(left, lr)
    prime(right, rr)
    candidate = H.NoGrowthAllocator(right)
    for kv in (left, right):
        if kind == 'prefix_enabled':
            kv.enable_caching = True
        else:
            kv.empty_kv_cache_blocks = KVCacheBlocks(tuple(() for _ in range(6)))
    outcomes = []
    for kv, r in ((left, lr), (right, rr)):
        try:
            outcomes.append(('return', returned(kv.allocate_slots(r, 1), kv)))
        except Exception as exc:
            outcomes.append(('error', type(exc).__module__, type(exc).__qualname__, str(exc)))
    assert outcomes[0] == outcomes[1] and ledger(left) == ledger(right)
    assert candidate.counts['fast_hits'] == 0 and candidate.counts['fallback_calls'] == 1
    candidate.close()
