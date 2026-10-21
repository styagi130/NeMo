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
"""Skip a proven no-op Q1 allocation; preserve the complete native fallback."""
import os
import threading
import weakref
from collections import Counter
from types import MethodType


class NoGrowthAllocator:
    """Skip only a proven no-op original allocation; no Request retention.

    The actual original create_kv_cache_blocks returns the exact immutable
    empty_kv_cache_blocks singleton for six empty result lists. This delegate
    returns that same object, not a new mutable object or cached payload.
    """

    def __init__(self, kv):
        from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
        from vllm.v1.request import Request, RequestStatus

        if type(kv) is not KVCacheManager:
            raise TypeError('exact original KVCacheManager required')
        original = kv.allocate_slots
        if (
            'allocate_slots' in vars(kv)
            or getattr(original, '__self__', None) is not kv
            or original.__func__ is not KVCacheManager.allocate_slots
        ):
            raise RuntimeError('unmodified inherited original allocator required')
        self.kv = weakref.ref(kv)
        self.pid, self.tid = os.getpid(), threading.get_ident()
        self.counts, self.fallbacks = Counter(), Counter()
        self.closed = False
        self.original = KVCacheManager.allocate_slots
        self.coordinator = kv.coordinator
        self.group_list = kv.coordinator.single_type_managers
        self.managers = tuple(self.group_list)
        self.pool = kv.block_pool
        self.empty = kv.empty_kv_cache_blocks
        self.empty_blocks = self.empty.blocks
        self.types = (
            KVCacheManager,
            KVCacheCoordinatorNoPrefixCache,
            FullAttentionManager,
            MambaManager,
            FullAttentionSpec,
            MambaSpec,
            Request,
            RequestStatus,
        )
        self.protected = [
            (KVCacheManager, name, getattr(KVCacheManager, name))
            for name in (
                'allocate_slots',
                'create_kv_cache_blocks',
                'free',
                'remove_skipped_blocks',
                'pop_blocks_for_free',
            )
        ]
        self.manager_methods = [
            (cls, name, getattr(cls, name))
            for cls in (FullAttentionManager, MambaManager)
            for name in (
                'allocate_new_blocks',
                'remove_skipped_blocks',
                'get_num_blocks_to_allocate',
                'get_num_skipped_tokens',
            )
        ]
        self.original_methods = tuple(self.protected + self.manager_methods)
        self.group_plan = tuple(
            (manager, manager.block_size, manager.kv_cache_spec, type(manager) is MambaManager)
            for manager in self.managers
        )
        self.layout_reason_at_install = self._layout_reason(kv)
        self.eligible_layout = self.layout_reason_at_install is None
        state_ref = weakref.ref(self)
        original_function = self.original

        def allocate_slots(owner, *args, **kwargs):
            state = state_ref()
            if state is None:
                return original_function(owner, *args, **kwargs)
            return state.allocate(*args, **kwargs)

        self.function = allocate_slots
        kv.allocate_slots = MethodType(allocate_slots, kv)

    def _core(self):
        if (os.getpid(), threading.get_ident()) != (self.pid, self.tid):
            raise RuntimeError('original allocator core PID/thread required')

    def _layout_reason(self, kv):
        KV, Coordinator, FA, Mamba, FASpec, MambaSpec, Request, RequestStatus = self.types
        if (
            type(kv.coordinator) is not Coordinator
            or kv.coordinator is not self.coordinator
            or kv.block_pool is not self.pool
            or kv.coordinator.block_pool is not self.pool
            or kv.num_kv_cache_groups != 6
            or len(self.managers) != 6
            or len(kv.coordinator.single_type_managers) != 6
        ):
            return 'unsupported_layout'
        if kv.enable_caching or kv.coordinator.enable_caching or kv.use_eagle or kv.coordinator.eagle_group_ids:
            return 'prefix_or_speculation'
        if (
            kv.empty_kv_cache_blocks is not self.empty
            or self.empty.blocks is not self.empty_blocks
            or type(self.empty_blocks) is not tuple
            or len(self.empty_blocks) != 6
            or any(type(group) is not tuple or group for group in self.empty_blocks)
        ):
            return 'noncanonical_empty'
        mamba_count = attention_count = 0
        for index, manager in enumerate(self.managers):
            mamba = type(manager) is Mamba
            attention = type(manager) is FA
            if (
                not (mamba or attention)
                or kv.coordinator.single_type_managers[index] is not manager
                or manager.block_pool is not self.pool
                or manager.enable_caching
                or manager.dcp_world_size != 1
                or manager.pcp_world_size != 1
                or manager._max_admission_blocks_per_request is not None
            ):
                return 'unsupported_group'
            expected_size = 4096 if mamba else 400
            if (
                type(manager.kv_cache_spec) is not (MambaSpec if mamba else FASpec)
                or manager.block_size != expected_size
                or manager.kv_cache_spec.block_size != expected_size
            ):
                return 'group_geometry'
            if mamba and (
                manager.mamba_cache_mode != 'none'
                or manager.num_speculative_blocks != 0
                or manager.kv_cache_spec.num_speculative_blocks != 0
            ):
                return 'mamba_mode_or_speculation'
            mamba_count += mamba
            attention_count += attention
        if mamba_count != 5 or attention_count != 1:
            return 'mixed_groups'
        if any(getattr(cls, name) is not original for cls, name, original in self.original_methods):
            return 'original_method_changed'
        return None

    def _guard(
        self,
        kv,
        request,
        num_new_tokens,
        num_new_computed_tokens,
        new_computed_blocks,
        num_lookahead_tokens,
        num_external_computed_tokens,
        delay_cache_blocks,
        num_encoder_tokens,
        full_sequence_must_fit,
        reserved_blocks,
        has_scheduled_reqs,
    ):
        # Every unsupported case returns to the complete original allocator,
        # before touching its manager/pool/request state.
        if (
            type(num_new_tokens) is not int
            or num_new_tokens != 1
            or type(num_new_computed_tokens) is not int
            or num_new_computed_tokens != 0
            or new_computed_blocks is not None
            or type(num_lookahead_tokens) is not int
            or num_lookahead_tokens != 0
            or type(num_external_computed_tokens) is not int
            or num_external_computed_tokens != 0
            or delay_cache_blocks is not False
            or type(num_encoder_tokens) is not int
            or num_encoder_tokens != 0
            or full_sequence_must_fit is not False
            or type(reserved_blocks) is not int
            or reserved_blocks != 0
            or has_scheduled_reqs is not True
        ):
            return 'unsupported_arguments'
        Request, RequestStatus = self.types[6:]
        if (
            type(request) is not Request
            or request.status != RequestStatus.RUNNING
            or request.num_preemptions != 0
            or request.resumable is not False
            or request.is_prefill_chunk
            or request.has_encoder_inputs
        ):
            return 'unsupported_request'
        computed, inflight = request.num_computed_tokens, request.num_in_flight_tokens
        if (
            type(computed) is not int
            or type(inflight) is not int
            or type(kv.max_model_len) is not int
            or not 0 <= inflight <= computed < kv.max_model_len
        ):
            return 'unsupported_token_state'
        target = computed + 1
        # With target<=4096 the processed Mamba prefix has zero whole skipped
        # blocks. FullAttention never skips. Hence remove_skipped_blocks is a
        # complete no-op even with optimistic async num_computed_tokens.
        if target > 4096:
            return 'mamba_reclamation_boundary'
        # Source/class/full immutable-configuration validation occurs only at
        # install/control status. These hot checks cover the mutable fields
        # needed for the exact no-op allocation proof.
        if not self.eligible_layout:
            return self.layout_reason_at_install
        if (
            kv.coordinator is not self.coordinator
            or kv.block_pool is not self.pool
            or self.coordinator.block_pool is not self.pool
            or self.coordinator.single_type_managers is not self.group_list
            or len(self.group_list) != 6
        ):
            return 'layout_changed'
        if kv.enable_caching or self.coordinator.enable_caching or kv.use_eagle or self.coordinator.eagle_group_ids:
            return 'prefix_or_speculation'
        if kv.empty_kv_cache_blocks is not self.empty or self.empty.blocks is not self.empty_blocks:
            return 'noncanonical_empty'
        rid = request.request_id
        if type(rid) is not str:
            return 'unsupported_request_id'
        for index, (manager, size, spec, mamba) in enumerate(self.group_plan):
            if (
                self.group_list[index] is not manager
                or manager.block_size != size
                or manager.kv_cache_spec is not spec
                or manager.block_pool is not self.pool
                or manager.enable_caching
            ):
                return 'group_changed'
            if mamba and (
                manager.mamba_cache_mode != 'none'
                or manager.num_speculative_blocks != 0
                or spec.num_speculative_blocks != 0
            ):
                return 'mamba_mode_or_speculation'
            if rid in manager.num_cached_block or rid in manager._partial_hit_reqs:
                return 'cached_or_partial_request'
            blocks = manager.req_to_blocks.get(rid)
            if blocks is None or len(blocks) * size < target:
                return 'capacity_growth'
        return None

    def allocate(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        self._core()
        if self.closed:
            raise RuntimeError('allocator already closed')
        kv = self.kv()
        if kv is None:
            raise RuntimeError('original allocator owner retired')
        self.counts['calls'] += 1
        args = (
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )
        try:
            reason = self._guard(kv, *args)
            if reason is not None:
                self.counts['fallback_calls'] += 1
                self.fallbacks[reason] += 1
                return self.original(kv, *args)
            self.counts['fast_hits'] += 1
            return self.empty
        except BaseException:
            self.counts['errors'] += 1
            raise

    def status(self):
        self._core()
        kv = self.kv()
        bound = None if kv is None else getattr(kv, 'allocate_slots')
        installed = (
            not self.closed
            and getattr(bound, '__self__', None) is kv
            and getattr(bound, '__func__', None) is self.function
        )
        restored = (
            self.closed
            and kv is not None
            and 'allocate_slots' not in vars(kv)
            and getattr(bound, '__func__', None) is self.original
        )
        if not (installed or restored):
            raise RuntimeError('allocator callable ownership changed')
        return dict(
            enabled=True,
            installed=installed,
            closed=self.closed,
            binding_matches=installed,
            binding_restored=restored,
            eligible_layout=self.eligible_layout,
            current_layout_matches=None if self.closed else self._layout_reason(kv) is None,
            counts={name: self.counts[name] for name in ('calls', 'fast_hits', 'fallback_calls', 'errors')},
            fallbacks=dict(self.fallbacks),
            retained_Request_references=0,
        )

    def close(self):
        self._core()
        if self.closed:
            return self.status()
        self.status()  # Refuse to restore a foreign callable.
        kv = self.kv()
        delattr(kv, 'allocate_slots')
        self.closed = True
        # Drop manager/pool references after restoration; retain scalar receipts.
        self.managers = self.group_plan = ()
        self.group_list = None
        self.coordinator = self.pool = self.empty = self.empty_blocks = None
        return self.status()
