# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-fill-wait batching on top of the frozen persistent publication pool.

Already-ready deadline heads are serialized before taking the shared lock. Up to
16 distinct owners are published together. A failed publication keeps its exact
record inflight for a later callback; successful publication is never reported
before bytes have committed. Partial commits are not re-sent on retry.
"""
import fcntl
import os
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager

from .pooled_publication import (
    _OCCUPIED_OFFSET,
    _U64,
    PoolBusy,
    PooledConnector,
    SharedPublicationPool,
    StaleDescriptor,
)


class BatchedPublicationPool(SharedPublicationPool):
    @contextmanager
    def _lock(self):
        with self._mutex:
            if self._closed:
                raise RuntimeError('publication pool is closed')
            depth = getattr(self._batch_local, 'depth', 0)
            if depth == 0:
                fcntl.flock(self.fd, fcntl.LOCK_EX)
                self.counts['persistent_lock_acquisitions'] += 1
            self._batch_local.depth = depth + 1
            try:
                yield
            finally:
                self._batch_local.depth -= 1
                if depth == 0:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)

    @classmethod
    def upgrade(cls, pool):
        if isinstance(pool, cls):
            return pool
        if type(pool) is not SharedPublicationPool:
            raise ValueError('batching requires the supported v1 publication pool')
        pool._batch_local = threading.local()
        pool._batch_poisoned = False
        pool.__class__ = cls
        return pool

    def put_many(self, items, *, timeout=None):
        """Return one result per item; preserve actual partial commits on failure.

        Valid results contain descriptor and commit_ns; cancelled records contain
        cancelled=True. A store exception produces error entries for that and all
        unattempted records, keeping earlier descriptors available for honest ACK.
        Validation and capacity failures happen before any write. The caller must
        retain failed records and must not acknowledge them as successful.
        """
        if self._batch_poisoned:
            raise RuntimeError('publication pool has an uncertain partial commit; restart required')
        if not items:
            return []
        if len(items) > min(64, self.slots):
            raise ValueError('publication batches are bounded to 64 and pool capacity')
        keys = set()
        owners = {}
        for item in items:
            key, payload, generation = item['key'], item['payload'], item['request_generation']
            rid, _ = self.key_parts(key)
            if key in keys or (rid in owners and owners[rid] != generation):
                raise ValueError('duplicate key or mixed generations in one batch')
            keys.add(key)
            owners[rid] = generation
            if (
                type(payload) is not bytes
                or len(payload) > self.payload_bytes
                or len(key.encode()) > self.key_bytes
                or type(generation) is not int
                or generation <= 0
                or item.get('flags', 0) & ~3
            ):
                raise ValueError('invalid bounded batch payload')
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        waited = False
        while True:
            with self._lock():
                active = []
                blocked = False
                for item in items:
                    rid, _ = self.key_parts(item['key'])
                    generation = item['request_generation']
                    owner = rid, generation
                    if owner in self._cancelled and not item.get('flags', 0):
                        active.append(False)
                        continue
                    active.append(True)
                    previous = self._owners.get(rid)
                    if previous is not None and generation < previous[0]:
                        raise StaleDescriptor('batch request generation precedes current owner')
                    if (
                        previous is not None
                        and previous[0] != generation
                        and any(self._live_descriptor(d) for d in previous[1])
                    ):
                        blocked = True
                    if self._find(item['key'].encode()) is not None:
                        blocked = True
                occupied = _U64.unpack_from(self.memory, _OCCUPIED_OFFSET)[0]
                if not blocked and self.slots - occupied >= sum(active):
                    results = []
                    failure = None
                    for item, live in zip(items, active):
                        if failure is not None:
                            results.append({'error': failure, 'attempted': False})
                        elif not live:
                            results.append({'cancelled': True})
                        else:
                            before_puts = self.counts['puts']
                            try:
                                descriptor = self._store_one(item)
                                results.append({'descriptor': descriptor, 'commit_ns': time.monotonic_ns()})
                            except Exception as exc:
                                # A wrapper can fail after the underlying store
                                # completed. Recover only a fully accounted own
                                # commit; ambiguous partial writes fail closed.
                                found = self._find(item['key'].encode())
                                recovered = False
                                if found is not None:
                                    index, _, header = found
                                    descriptor = self._descriptor(index, header)
                                    rid, sequence = self.key_parts(item['key'])
                                    owner = self._owners.get(rid)
                                    accounted = (
                                        owner is not None
                                        and owner[0] == item['request_generation']
                                        and descriptor in owner[1]
                                        and self.counts['puts'] == before_puts + 1
                                        and header[2:4] == (item['request_generation'], sequence)
                                    )
                                    if accounted:
                                        results.append(
                                            {
                                                'descriptor': descriptor,
                                                'commit_ns': time.monotonic_ns(),
                                                'recovered_exception': str(exc),
                                            }
                                        )
                                        self.counts['recovered_postcommit_store_exceptions'] += 1
                                        recovered = True
                                    else:
                                        self._batch_poisoned = True
                                        self.counts['uncertain_partial_commits'] += 1
                                if not recovered:
                                    failure = exc
                                    results.append({'error': exc, 'attempted': True})
                    self.counts['batched_put_calls'] += 1
                    self.counts['batched_put_records'] += len(items)
                    self.counts['maximum_put_batch'] = max(self.counts['maximum_put_batch'], len(items))
                    if failure is not None:
                        self.counts['partial_batch_exceptions'] += 1
                    if waited:
                        self.counts['backpressured_batches'] += 1
                    return results
                self.counts['batch_backpressure_polls'] += 1
            waited = True
            if time.monotonic() >= deadline:
                self.counts['batch_backpressure_timeouts'] += 1
                raise PoolBusy('publication batch requires receiver acknowledgement before reuse')
            time.sleep(0.0002)

    def _store_one(self, item):
        return super().put(
            item['key'],
            item['payload'],
            request_generation=item['request_generation'],
            flags=item.get('flags', 0),
            timeout=0,
        )


class BatchedReadyPublisher:
    def __init__(self, controller, max_batch=16):
        if not 1 <= max_batch <= 64:
            raise ValueError('batch size must be between one and 64')
        self.controller = controller
        self.policy = controller.deadline_policy
        if self.policy is None:
            raise ValueError('batched publication requires the supported ready deadline queue')
        self.proxy = controller.adapter.connector
        if not isinstance(self.proxy, PooledConnector) or self.proxy.role != 'sender':
            raise ValueError('install the v1 sender pool before batching')
        self.pool = BatchedPublicationPool.upgrade(self.proxy.pool)
        self.max_batch = min(max_batch, self.pool.slots)
        self.retry = deque()
        self.counts = Counter()
        self._lock = threading.RLock()

    @staticmethod
    def _failed(request):
        return getattr(getattr(request, 'status', None), 'name', None) in ('FINISHED_ABORTED', 'FINISHED_ERROR')

    def _error(self, record, error, phase):
        self.retry.append(record)
        self.counts['retained_failed_records'] += 1
        self.counts['error:' + phase] += 1
        self.policy.counters['publication_exceptions'] += 1
        self.controller.consumer['publication_exceptions'] += 1
        record['_pooled_publication_last_error'] = f'{type(error).__name__}: {error}'

    def _finish(self, record):
        """Idempotent post-commit bookkeeping; never re-publishes committed bytes."""
        commit = record['_pooled_publication_commit']
        controller, adapter = self.controller, self.controller.adapter
        request, state, task = record['task']['request'], record['state'], record['task']
        rid = request.external_req_id
        if not commit.get('counted'):
            controller.consumer['published_chunks'] += 1
            controller.consumer['serialized_bytes'] += commit['size']
            self.counts['committed_records'] += 1
            commit['counted'] = True
        if not commit.get('sequence_advanced'):
            if adapter.put_req_chunk[rid] != commit['sequence']:
                raise RuntimeError('request chunk sequence changed before publication acknowledgement')
            adapter.put_req_chunk[rid] += 1
            commit['sequence_advanced'] = True
        if not commit.get('ramp_advanced'):
            adapter.ramp_chunk_count[rid] += 1
            commit['ramp_advanced'] = True
        if task['is_finished']:
            if not commit.get('cleaned'):
                adapter.cleanup(request.request_id, rid)
                commit['cleaned'] = True
            if not commit.get('released'):
                controller._release(state)
                commit['released'] = True
        if task['is_segment_finished'] and not commit.get('segment_cleaned'):
            for name in ('code_prompt_token_ids', 'requests_num_chunks_sent', 'ramp_chunk_count', '_cached_ic'):
                getattr(adapter, name, {}).pop(rid, None)
            commit['segment_cleaned'] = True
        if not self.policy.complete(record, True, now_ns=commit['commit_ns']):
            raise RuntimeError('publication succeeded but its deadline owner rejected acknowledgement')
        record.pop('_pooled_publication_last_error', None)

    def _dispatch(self, records):
        prepared, items = [], []
        for record in records:
            try:
                if '_pooled_publication_commit' in record:
                    self._finish(record)
                    self.counts['bookkeeping_retries_completed'] += 1
                    continue
                task, state = record['task'], record['state']
                if self._failed(task['request']) and not (task['is_finished'] or task['is_segment_finished']):
                    self.controller._release(state)
                    self.controller.consumer['cancelled_ready_chunks'] += 1
                    self.policy.complete(record, False)
                    continue
                adapter = self.controller.adapter
                rid = task['request'].external_req_id
                sequence = adapter.put_req_chunk[rid]
                self.controller.consumer['chunk_serialization_calls'] += 1
                payload = self.proxy.original.serialize_obj(record['payload'])
                flags = int(bool(task['is_finished'])) | (int(bool(task['is_segment_finished'])) << 1)
                item = {
                    'key': f'{rid}_{adapter.connector.stage_id}_{sequence}',
                    'payload': payload,
                    'request_generation': record['owner'][1],
                    'flags': flags,
                }
                prepared.append((record, sequence, len(payload)))
                items.append(item)
            except Exception as exc:
                self._error(record, exc, 'prepare_or_bookkeeping')
        if not items:
            return
        try:
            results = self.pool.put_many(items)
        except Exception as exc:
            for record, _, _ in prepared:
                self._error(record, exc, 'transport_precommit')
            return
        for (record, sequence, size), result in zip(prepared, results):
            if 'error' in result:
                self._error(record, result['error'], 'transport_partial')
            elif result.get('cancelled'):
                self.policy.complete(record, False)
                self.counts['cancelled_publications'] += 1
            else:
                record['_pooled_publication_commit'] = {
                    'descriptor': result['descriptor'],
                    'commit_ns': result['commit_ns'],
                    'sequence': sequence,
                    'size': size,
                }
                try:
                    self._finish(record)
                except Exception as exc:
                    self._error(record, exc, 'postcommit_bookkeeping')

    def publish_ready(self, records):
        with self._lock:
            self.policy.counters['ready_lists'] += 1
            self.policy.counters['maximum_ready_list'] = max(self.policy.counters['maximum_ready_list'], len(records))
            self.policy.offer_many(records)
            # Retried heads already own policy.inflight and cannot be selected
            # twice. Try each once per callback; hard failures cannot spin here.
            old_retries = self.retry
            self.retry = deque()
            while old_retries:
                batch = [old_retries.popleft() for _ in range(min(self.max_batch, len(old_retries)))]
                self._dispatch(batch)
            while True:
                batch = []
                for _ in range(self.max_batch):
                    record = self.policy.take_next()
                    if record is None:
                        break
                    batch.append(record)
                if not batch:
                    break
                self.counts['ready_batches'] += 1
                self.counts['selected_records'] += len(batch)
                self.counts['maximum_ready_batch'] = max(self.counts['maximum_ready_batch'], len(batch))
                self._dispatch(batch)

    def status(self):
        return {
            'enabled': True,
            'version': 2,
            'max_batch': self.max_batch,
            'counts': dict(self.counts),
            'retry_records': len(self.retry),
            'no_batch_fill_wait': True,
            'batch_scope': 'currently ready heads; one inflight record per owner',
            'ack_boundary': 'actual pool commit plus completed publication bookkeeping',
            'failure_policy': 'retain inflight record; retry on next callback; never false-ACK',
            'pool_poisoned': self.pool._batch_poisoned,
        }


def install_batched_publication(controller):
    if os.environ.get('EASYMAGPIE_POOLED_PUBLICATION_BATCH', '0') != '1':
        return None
    if hasattr(controller, 'batched_publisher'):
        return controller.batched_publisher
    publisher = BatchedReadyPublisher(
        controller, max_batch=int(os.environ.get('EASYMAGPIE_PUBLICATION_BATCH_MAX', '16'))
    )
    controller.batched_publisher = publisher
    controller.publish_ready = publisher.publish_ready
    original_status = controller.status

    def status():
        result = original_status()
        result['batched_publication'] = publisher.status()
        return result

    controller.status = status
    return publisher
