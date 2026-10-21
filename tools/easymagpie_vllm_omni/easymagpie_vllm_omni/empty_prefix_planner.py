# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Exact empty-prefix planning for the uncached EasyMagpie KV layout."""
import os
import threading
from collections import Counter
from types import MethodType


class Planner:
    def __init__(self, kv, *, shadow=False):
        from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator, KVCacheCoordinatorNoPrefixCache
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        if type(kv) is not KVCacheManager or type(kv.coordinator) is not KVCacheCoordinatorNoPrefixCache:
            raise TypeError('exact retained uncached KV manager/coordinator required')
        self.kv, self.coordinator = kv, kv.coordinator
        self.managers = self.coordinator.single_type_managers
        self.empty = kv.empty_kv_cache_blocks.blocks
        if type(self.empty) is not tuple or len(self.empty) != 6 or any(type(x) is not tuple or x for x in self.empty):
            raise ValueError('exact six immutable empty-prefix groups required')
        if (
            len(self.managers) != 6
            or sum(type(m) is MambaManager for m in self.managers) != 5
            or sum(type(m) is FullAttentionManager for m in self.managers) != 1
        ):
            raise ValueError('exact five Mamba plus one FullAttention managers required')
        if kv.enable_caching or self.coordinator.enable_caching or kv.use_eagle or self.coordinator.eagle_group_ids:
            raise ValueError('prefix caching and speculation must remain disabled')
        if (
            getattr(self.coordinator.get_num_blocks_to_allocate, '__func__', None)
            is not KVCacheCoordinator.get_num_blocks_to_allocate
        ):
            raise RuntimeError('original coordinator predictor required')
        self.original = self.coordinator.get_num_blocks_to_allocate
        self.pid, self.tid = os.getpid(), threading.get_ident()
        self.shadow = shadow
        self.counts = Counter()
        self.fallbacks = Counter()
        self.plan = []
        self.closed = False
        for manager in self.managers:
            mamba = type(manager) is MambaManager
            spec = manager.kv_cache_spec
            if type(spec) is not (MambaSpec if mamba else FullAttentionSpec):
                raise TypeError('exact retained cache spec required')
            if (
                manager.enable_caching
                or manager.block_pool is not kv.block_pool
                or self.coordinator.block_pool is not kv.block_pool
            ):
                raise ValueError('uncached original shared block pool required')
            if (
                manager.dcp_world_size != 1
                or manager.pcp_world_size != 1
                or manager.block_size <= 0
                or manager._max_admission_blocks_per_request is not None
            ):
                raise ValueError('unsupported cache geometry')
            mode = manager.mamba_cache_mode if mamba else None
            if mamba and (
                mode not in ('none', 'all') or manager.num_speculative_blocks != 0 or spec.num_speculative_blocks != 0
            ):
                raise ValueError('nonalign nonspeculative Mamba required')
            functions = {
                name: getattr(type(manager), name)
                for name in (
                    'get_num_blocks_to_allocate',
                    'allocate_new_blocks',
                    'remove_skipped_blocks',
                    'pop_blocks_for_free',
                    'get_num_skipped_tokens',
                )
            }
            self.plan.append((manager, mamba, manager.block_size, spec, mode, functions))
        self.pool = kv.block_pool
        self.kv_coordinator = self.coordinator
        self.manager_types = (FullAttentionManager, MambaManager)
        self.protected = {
            name: getattr(type(kv), name)
            for name in ('allocate_slots', 'free', 'remove_skipped_blocks', 'pop_blocks_for_free')
        }
        self.wrapper = MethodType(lambda coordinator, *args, **kwargs: self.predict(*args, **kwargs), self.coordinator)
        self.coordinator.get_num_blocks_to_allocate = self.wrapper

    def _fallback(self, reason, args):
        self.counts['fallback_calls'] += 1
        self.fallbacks[reason] += 1
        try:
            return self.original(*args)
        except BaseException:
            self.counts['errors'] += 1
            raise

    def predict(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        if (os.getpid(), threading.get_ident()) != (self.pid, self.tid):
            raise RuntimeError('original core predictor thread required')
        self.counts['calls'] += 1
        args = (
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        if (
            self.kv.coordinator is not self.kv_coordinator
            or self.coordinator.single_type_managers is not self.managers
            or len(self.managers) != 6
            or self.kv.empty_kv_cache_blocks.blocks is not self.empty
        ):
            return self._fallback('layout_changed', args)
        if (
            self.kv.enable_caching
            or self.coordinator.enable_caching
            or self.kv.use_eagle
            or self.coordinator.eagle_group_ids
        ):
            return self._fallback('prefix_caching_or_speculation_changed', args)
        if new_computed_blocks is not self.empty:
            return self._fallback('noncanonical_prefix', args)
        if num_encoder_tokens != 0 or apply_admission_cap is not False:
            return self._fallback('encoder_or_admission', args)
        if any(
            type(v) is not int or v < 0
            for v in (num_tokens, total_computed_tokens, num_local_computed_tokens, num_tokens_main_model)
        ):
            return self._fallback('unsupported_tokens', args)
        result = 0
        for index, (manager, mamba, size, spec, mode, functions) in enumerate(self.plan):
            if (
                self.managers[index] is not manager
                or type(manager) is not self.manager_types[int(mamba)]
                or manager.kv_cache_spec is not spec
                or manager.block_size != size
                or manager.block_pool is not self.pool
                or manager.enable_caching
                or manager._max_admission_blocks_per_request is not None
                or manager.dcp_world_size != 1
                or manager.pcp_world_size != 1
            ):
                return self._fallback('manager_configuration_changed', args)
            if mamba and (
                manager.mamba_cache_mode != mode
                or manager.num_speculative_blocks != 0
                or spec.num_speculative_blocks != 0
            ):
                return self._fallback('mamba_configuration_changed', args)
            if request_id in manager.num_cached_block or request_id in manager._partial_hit_reqs:
                return self._fallback('cached_or_partial_request', args)
            required = (num_tokens + size - 1) // size
            skipped = (total_computed_tokens - 1) // size if mamba else 0
            held = len(manager.req_to_blocks.get(request_id, ()))
            result += max(required - max(skipped, held), 0)
        self.counts['fast_hits'] += 1
        if self.shadow:
            self.counts['shadow_comparisons'] += 1
            try:
                expected = self.original(*args)
            except BaseException:
                self.counts['errors'] += 1
                raise
            if result != expected:
                self.counts['prediction_mismatches'] += 1
                self.counts['errors'] += 1
                raise RuntimeError('empty-prefix prediction mismatch before allocation')
        return result

    def status(self):
        if (os.getpid(), threading.get_ident()) != (self.pid, self.tid):
            raise RuntimeError('original core status thread required')
        if not self.closed and self.coordinator.get_num_blocks_to_allocate is not self.wrapper:
            raise RuntimeError('planner binding changed')
        if self.closed and self.coordinator.get_num_blocks_to_allocate != self.original:
            raise RuntimeError('planner restoration changed')
        if any(getattr(type(self.kv), name) is not original for name, original in self.protected.items()):
            raise RuntimeError('original allocator methods changed')
        for manager, mamba, size, spec, mode, functions in self.plan:
            if any(getattr(type(manager), name) is not original for name, original in functions.items()):
                raise RuntimeError('original group manager methods changed')
        return dict(
            counts={
                k: self.counts[k]
                for k in (
                    'calls',
                    'fast_hits',
                    'fallback_calls',
                    'shadow_comparisons',
                    'prediction_mismatches',
                    'errors',
                )
            },
            fallbacks=dict(self.fallbacks),
            shadow=self.shadow,
            installed=not self.closed,
            closed=self.closed,
        )

    def close(self):
        if self.closed:
            return
        self.status()
        del self.coordinator.get_num_blocks_to_allocate
        self.closed = True
