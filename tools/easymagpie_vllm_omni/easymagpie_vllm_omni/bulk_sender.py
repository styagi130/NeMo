# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-update CPU chunk construction, with request-owned accumulation slots.

The scheduler transfers one owning task batch per update, with no batch-fill wait.
Regular frames never enter the generic transfer adapter/processor. The connector
still serializes independently owned chunks in original task order.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.request import RequestStatus
from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.engine.serialization import deserialize_additional_information

from . import stage_processors
from .chunk_schedule import startup_schedule

logger = logging.getLogger(__name__)
_BATCH = '_easymagpie_bulk_tasks_v1'
HAS_FRAME, SKIP_ZERO, TERMINAL, SEGMENT = 1, 2, 4, 8
_FAILED = (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR)


@dataclass
class ChunkEvent:
    index: int
    kind: str
    codes: torch.Tensor | None
    frames: int = 0


class PythonChunkAccumulator:
    """Reference bulk backend: one preallocated row pool; no per-frame tensors.

    Copying batches with NumPy keeps the fixed storage separate from immutable
    published outputs. The native backend implements the same state machine.
    """

    def __init__(self, capacity=2048, width=16, startup_frames=(2, 6)):
        self.capacity, self.width = capacity, width
        self.startup_frames = startup_schedule(startup_frames)
        self.pool = np.empty((capacity, 8, width), dtype=np.int64)
        self.generations = [0] * capacity
        self.seen = [0] * capacity
        self.delays = [0] * capacity
        self.pending = [0] * capacity
        self.emitted = [0] * capacity
        self.chunks = [0] * capacity

    def reset(self, slot, generation, delay):
        self.generations[slot] = generation
        self.seen[slot] = self.pending[slot] = self.emitted[slot] = self.chunks[slot] = 0
        self.delays[slot] = delay

    def release(self, slot, generation):
        if self.generations[slot] == generation:
            self.generations[slot] = 0

    def export_state(self, slot):
        return {
            'seen': self.seen[slot],
            'delay': self.delays[slot],
            'pending': self.pending[slot],
            'emitted': self.emitted[slot],
            'chunks': self.chunks[slot],
            'buffer': torch.from_numpy(self.pool[slot, : self.pending[slot]].copy()),
        }

    def append(self, rows, slots, generations, flags):
        events = []
        for index, (row, slot, generation, flag) in enumerate(zip(rows, slots, generations, flags)):
            if generation != self.generations[slot]:
                events.append(ChunkEvent(index, 'stale', None))
                continue
            terminal, segment = bool(flag & TERMINAL), bool(flag & SEGMENT)
            if flag & HAS_FRAME:
                values = row.numpy().reshape(-1)
                if not (flag & SKIP_ZERO) or values.any():
                    self.seen[slot] += 1
                    if self.seen[slot] > self.delays[slot]:
                        self.pool[slot, self.pending[slot]] = values
                        self.pending[slot] += 1
            pending, count = self.pending[slot], self.chunks[slot]
            target = self.startup_frames[count] if count < len(self.startup_frames) else 8
            if pending and (terminal or pending >= target):
                codes = self.pool[slot].copy() if terminal else self.pool[slot, :pending].copy()
                if terminal and pending < 8:
                    codes[pending:] = -1
                events.append(ChunkEvent(index, 'chunk', torch.from_numpy(codes), pending))
                self.emitted[slot] += pending
                self.pending[slot] = 0
                self.chunks[slot] += 1
            elif terminal or segment:
                kind = 'empty' if self.emitted[slot] + pending == 0 else 'marker'
                codes = torch.empty(0, dtype=torch.long) if kind == 'empty' else None
                events.append(ChunkEvent(index, kind, codes))
        return events


@dataclass
class RequestState:
    request: object
    slot: int
    generation: int
    delay: int
    enqueue_ns: int
    fallback: bool = False


class BulkSender:
    def __init__(self, adapter, capacity=2048, max_tasks=512, backend='python'):
        self.adapter = adapter
        self._send_original = adapter._send_single_request
        self.max_tasks = max_tasks
        if not 1 <= max_tasks <= 4096:
            raise ValueError('bulk sender max_tasks must be 1..4096')
        cfg = getattr(adapter.connector, 'config', {})
        cfg = cfg.get('extra', cfg)
        if cfg.get('codec_chunk_frames') != 8:
            raise ValueError('bulk sender requires steady chunk 8')
        self.startup_frames = startup_schedule(cfg.get('codec_startup_chunk_frames', []))
        self.backend_name = backend
        self.frame_viewless = os.environ.get('EASYMAGPIE_BULK_FRAME_VIEWLESS', '0') == '1'
        if backend == 'python':
            self.backend = PythonChunkAccumulator(capacity, startup_frames=self.startup_frames)
        elif backend == 'native':
            from .bulk_native import BulkChunkAccumulator

            self.backend = BulkChunkAccumulator(capacity=capacity, width=16, startup_frames=self.startup_frames)
        else:
            raise ValueError('unknown bulk sender backend')
        self.free = list(range(capacity - 1, -1, -1))
        self.states = {}
        self.current_owner = {}
        self.next_generation = 0
        self._pending = []
        self._scope_owner = None
        self._scope_depth = 0
        self._lock = threading.RLock()
        self.consumer = Counter()
        self.deadline_policy = None
        self._true = torch.tensor(True, dtype=torch.bool)
        self._false = torch.tensor(False, dtype=torch.bool)

    @contextmanager
    def scheduler_update(self):
        with self._lock:
            owner = threading.get_ident()
            if self._scope_owner not in (None, owner):
                raise RuntimeError('concurrent scheduler updates are unsupported')
            self._scope_owner = owner
            self._scope_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._scope_depth -= 1
                if not self._scope_depth:
                    self._flush()
                    self._scope_owner = None

    def save_async(self, multimodal_output=None, request=None, is_segment_finished=False):
        with self._lock:
            confirmed = self.adapter._confirmed_num_computed_tokens(request)
            rid = request.external_req_id
            if confirmed < self.adapter.requests_num_chunks_sent.get(rid, 0):
                return
            self.adapter.requests_num_chunks_sent[rid] = confirmed
            task = {
                'request': request,
                'multimodal_output': multimodal_output,
                'is_finished': request.is_finished() and not request.resumable,
                'is_segment_finished': is_segment_finished,
                'enqueue_ns': time.monotonic_ns(),
            }
            if self._scope_owner == threading.get_ident():
                self._pending.append(task)
                if len(self._pending) >= self.max_tasks:
                    self._flush()
            else:
                self._flush()
                self._enqueue((task,))

    def enqueue_cancellations(self, requests):
        with self._lock:
            self._flush()
            tasks = tuple({'request': req, '_bulk_cancel_only': True} for req in requests)
            if tasks:
                self._enqueue(tasks)

    def _flush(self):
        if self._pending:
            tasks, self._pending = tuple(self._pending), []
            self._enqueue(tasks)

    def _enqueue(self, tasks):
        self.adapter._pending_save_reqs.append({_BATCH: tasks, 'request_id': 'bulk-sender'})
        with self.adapter._save_cond:
            self.adapter._save_cond.notify()

    def _release(self, state):
        key = id(state.request)
        if self.states.get(key) is not state:
            return
        self.states.pop(key)
        if self.deadline_policy is not None and getattr(state.request, 'status', None) in _FAILED:
            self.deadline_policy.cancel((state.request.external_req_id, state.generation), reason='cancelled')
        self.backend.release(state.slot, state.generation)
        self.free.append(state.slot)
        rid = state.request.external_req_id
        if self.current_owner.get(rid) is state:
            self.current_owner.pop(rid)

    def _state(self, task):
        request = task['request']
        key = id(request)
        state = self.states.get(key)
        if state is not None:
            return state
        current = self.current_owner.get(request.external_req_id)
        # The scheduler copies a resumable Request for a late input-close
        # sentinel. It closes the existing acoustic generation, including a
        # buffered tail, rather than registering an empty replacement stream.
        if (
            current is not None
            and task.get('is_finished')
            and task.get('multimodal_output') is None
            and request.request_id == current.request.request_id
            and getattr(current.request, 'status', None) not in _FAILED
        ):
            return current
        # A new request is a low-frequency point for reclaiming cancellations
        # whose scheduler control message has not yet reached this sender.
        for previous in tuple(self.states.values()):
            if getattr(previous.request, 'status', None) in _FAILED:
                self._release(previous)
        if not self.free:
            raise RuntimeError('bulk sender slot capacity exhausted')
        base = stage_processors._resolve_speech_delay(self.adapter)
        info = deserialize_additional_information(getattr(request, 'additional_information', None))
        prefill = int(info.get('text_prefill_num', 0) or 0)
        if not 0 <= prefill <= base:
            raise ValueError('invalid text_prefill_num for speech delay')
        self.next_generation += 1
        state = RequestState(
            request, self.free.pop(), self.next_generation, base - prefill, task.get('enqueue_ns', time.monotonic_ns())
        )
        self.states[key] = state
        if self.deadline_policy is not None:
            self.deadline_policy.register(
                (request.external_req_id, state.generation),
                request,
                arrival_ns=state.enqueue_ns,
                anchor_scope='sender-enqueue',
            )
        self.current_owner[request.external_req_id] = state
        self.adapter.put_req_chunk[request.external_req_id] += 0
        self.backend.reset(state.slot, state.generation, state.delay)
        if info.get('reset_codec_on_segment', False):
            self._bridge_fallback(state)
        return state

    @staticmethod
    def _frame(task, viewless=False):
        raw = task['multimodal_output']
        if raw is None:
            return None, 0
        if type(raw) is not dict:
            raise ValueError('non-dict payload')
        if 'codes.audio' in raw:
            if any(key.startswith('codes.audio.') for key in raw):
                raise ValueError('ambiguous flattened payload')
            value = raw['codes.audio']
        elif type(raw.get('codes')) is dict:
            value = raw['codes'].get('audio')
        else:
            value = raw.get('audio_codes')
        if value is None or (isinstance(value, torch.Tensor) and value.numel() == 0):
            return None, 0
        if (
            type(value) is not torch.Tensor
            or value.device.type != 'cpu'
            or value.dtype != torch.long
            or value.requires_grad
            or value.ndim not in (1, 2)
            or value.shape[-1] != 16
            or not value.is_contiguous()
        ):
            raise ValueError('unsupported frame contract')
        if viewless and value.ndim == 2 and value.shape[0] == 1:
            return value, HAS_FRAME | SKIP_ZERO
        return (value[-1] if value.ndim == 2 else value), HAS_FRAME | (SKIP_ZERO if value.ndim == 2 else 0)

    def _bridge_fallback(self, state):
        if state.fallback:
            return
        if self.deadline_policy is not None:
            self.deadline_policy.cancel((state.request.external_req_id, state.generation), reason='fallback')
        exported = self.backend.export_state(state.slot)
        rid = state.request.external_req_id
        adapter = self.adapter
        stage_processors._persistent_state(adapter, '_emp_seen_frames')[rid] = int(exported['seen'])
        stage_processors._persistent_state(adapter, '_emp_request_speech_delay')[rid] = state.delay
        emitted = int(exported['emitted'])
        for name in ('_emp_frame_buffer_base', '_emp_emitted_frames'):
            stage_processors._persistent_state(adapter, name)[rid] = emitted
        stage_processors._persistent_state(adapter, '_emp_emitted_chunks')[rid] = int(exported['chunks'])
        buffer = exported['buffer']
        stage_processors._persistent_list_state(adapter, '_emp_frame_buffer')[rid] = (
            list(buffer.unbind(0)) if isinstance(buffer, torch.Tensor) else list(buffer)
        )
        if not hasattr(adapter, '_emp_requests'):
            adapter._emp_requests = {}
        adapter._emp_requests[rid] = state.request
        state.fallback = True

    def send_envelope(self, envelope):
        tasks = envelope.get(_BATCH)
        if tasks is None:
            tasks = (envelope,)
        # Flush before uncommon fallback/control work to preserve exact global
        # publication order, including multiple rows for one request in a batch.
        pending = []
        for task in tasks:
            req = task['request']
            state = self.states.get(id(req))
            if task.get('_bulk_cancel_only') or getattr(req, 'status', None) in _FAILED:
                self._append_pending(pending)
                pending.clear()
                if state is not None:
                    self._release(state)
                if not task.get('_bulk_cancel_only') and (task['is_finished'] or task['is_segment_finished']):
                    self._send_original(task)
                continue
            try:
                state = self._state(task)
                if state.fallback:
                    raise ValueError('generation uses fallback')
                frame, flags = self._frame(task, self.frame_viewless)
                flags |= TERMINAL if task['is_finished'] else 0
                flags |= SEGMENT if task['is_segment_finished'] else 0
                pending.append((task, state, frame, flags))
            except Exception:
                self._append_pending(pending)
                pending.clear()
                if state is not None:
                    self._bridge_fallback(state)
                self._send_original(task)
                if state is not None and task['is_finished']:
                    self._release(state)
        self._append_pending(pending)

    def _append_pending(self, pending):
        if not pending:
            return
        rows, slots, generations, flags = [], [], [], []
        for task, state, row, flag in pending:
            rows.append(row)
            slots.append(state.slot)
            generations.append(state.generation)
            flags.append(flag)
        events = self.backend.append(rows, slots, generations, flags)
        ready = []
        for event in events:
            if event.kind == 'stale':
                continue
            task, state, _, _ = pending[event.index]
            meta = MetaStruct(
                finished=self._true if task['is_finished'] else self._false,
                is_segment_finished=self._true if task['is_segment_finished'] else self._false,
            )
            if event.kind == 'chunk':
                meta.left_context_size = 0
            payload = OmniPayloadStruct(meta=meta)
            if event.codes is not None:
                payload.codes = CodesStruct(audio=event.codes)
            ready.append(
                {
                    'task': task,
                    'request': state.request,
                    'state': state,
                    'owner': (task['request'].external_req_id, state.generation),
                    'payload': payload,
                    'produced_frames': event.frames,
                    'is_finished': task['is_finished'],
                    'is_segment_finished': task['is_segment_finished'],
                    'enqueue_ns': task.get('enqueue_ns', state.enqueue_ns),
                    'ready_ns': time.monotonic_ns(),
                }
            )
        self.publish_ready(ready)

    def publish_ready(self, ready):
        for record in ready:
            try:
                self.publish_record(record)
            except Exception as exc:
                # The pinned save loop isolates failed requests. Preserve that
                # behavior when a queue item now contains many ready chunks.
                self.consumer['publication_exceptions'] += 1
                logger.warning('Bulk chunk publication failed for %s: %s', record['owner'], exc)

    def publish_record(self, record):
        task, state = record['task'], record['state']
        request = task['request']
        # A concurrent abort invalidates unpublished ordinary audio. Finish
        # markers retain the original adapter's cancellation behavior.
        if getattr(request, 'status', None) in _FAILED and not (task['is_finished'] or task['is_segment_finished']):
            self._release(state)
            self.consumer['cancelled_ready_chunks'] += 1
            return False
        adapter, rid = self.adapter, request.external_req_id
        stage = adapter.connector.stage_id
        key = f'{rid}_{stage}_{adapter.put_req_chunk[rid]}'
        self.consumer['chunk_serialization_calls'] += 1
        success, size, metadata = adapter.connector.put(
            from_stage=str(stage), to_stage=str(stage + 1), put_key=key, data=record['payload']
        )
        self.consumer['published_chunks' if success else 'publication_failures'] += 1
        self.consumer['serialized_bytes'] += int(size or 0)
        if success:
            adapter.put_req_chunk[rid] += 1
            adapter.ramp_chunk_count[rid] += 1
            if task['is_finished']:
                adapter.cleanup(request.request_id, rid)
                self._release(state)
        if task['is_segment_finished']:
            for name in ('code_prompt_token_ids', 'requests_num_chunks_sent', 'ramp_chunk_count', '_cached_ic'):
                getattr(adapter, name, {}).pop(rid, None)
        return bool(success)

    def status(self):
        pool = getattr(self.adapter.connector, 'pool', None)
        return {
            'pooled_publication': pool.status() if pool is not None else {'enabled': False},
            'enabled': True,
            'backend': self.backend_name,
            'frame_viewless': self.frame_viewless,
            'version': 2,
            'startup_frames': list(self.startup_frames),
            'steady_frames': 8,
            'consumer': dict(self.consumer),
            'live_generations': len(self.states),
            'capacity': self.backend.capacity,
            'collector_pending_tasks': len(self._pending),
            'no_batch_fill_wait': True,
            'viewless_backend_calls': getattr(self.backend, 'viewless_batch_calls', 0),
            'viewless_backend_rows': getattr(self.backend, 'viewless_rows', 0),
        }


def install_bulk_sender(scheduler):
    if os.environ.get('EASYMAGPIE_BULK_SENDER', '0') != '1':
        return None
    adapter = scheduler.chunk_transfer_adapter
    if scheduler.vllm_config.model_config.stage_id != 0 or adapter is None:
        return None
    if adapter.custom_process_next_stage_input_func is not stage_processors.talker2code2wav_async_chunk:
        raise RuntimeError('bulk sender requires the supported EasyMagpie processor')
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter

    if not isinstance(adapter, OmniChunkTransferAdapter):
        raise TypeError('bulk sender requires the vLLM-Omni chunk transfer adapter')
    controller = BulkSender(
        adapter,
        capacity=int(os.environ.get('EASYMAGPIE_BULK_SENDER_SLOTS', '2048')),
        max_tasks=int(os.environ.get('EASYMAGPIE_BULK_SENDER_MAX_TASKS', '512')),
        backend=os.environ.get('EASYMAGPIE_BULK_SENDER_BACKEND', 'python'),
    )
    from .host_fastpath import install_metadata_wire_cache

    install_metadata_wire_cache(controller)
    from .ready_deadlines import install_deadline_policy

    install_deadline_policy(controller)
    from .pooled_publication import install_pooled_publication

    install_pooled_publication(adapter, role='sender', controller=controller)
    from .pooled_publication_v11 import install_owned_fallback_publication

    install_owned_fallback_publication(controller)
    from .pooled_publication_v2 import install_batched_publication

    install_batched_publication(controller)
    update = scheduler.update_from_output

    def update_from_output(*args, **kwargs):
        with controller.scheduler_update():
            return update(*args, **kwargs)

    scheduler.update_from_output = update_from_output
    finish = getattr(scheduler, 'finish_requests', None)
    if finish is not None:

        def finish_requests(request_ids, finished_status):
            if request_ids is None:
                affected = list(scheduler.requests.values())
            else:
                ids = (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
                request_ids = ids
                affected = [scheduler.requests[rid] for rid in ids if rid in scheduler.requests]
            result = finish(request_ids, finished_status)
            if finished_status in _FAILED:
                controller.enqueue_cancellations(affected)
            return result

        scheduler.finish_requests = finish_requests
    adapter.save_async = controller.save_async
    adapter._send_single_request = controller.send_envelope
    scheduler.bulk_sender = controller
    return controller
