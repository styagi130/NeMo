# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in persistent, bounded transport for the local talker -> codec edge.

Payload bytes and the decoder are unchanged. A successful read owns its decoded
bytes before acknowledging the slot. Acknowledgement, cancellation and slot
reuse are serialized with one persistent file lock. No polling sleep is added
unless the producer has exhausted the bounded pool or reuses an undrained ID.
This module never patches a connector class or an unrelated pipeline edge.
"""
from __future__ import annotations

import fcntl
import hashlib
import mmap
import os
import struct
import threading
import time
import uuid
import zlib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

_MAGIC = b'EMPPUB01'
_HEADER = struct.Struct('<8sIIIIQQqQ')
# next bucket/free entry, slot generation, owner generation, sequence,
# lifecycle state (0 free, 1 ready, 2 receiver decoding), key length, payload
# length, flags (terminal=1, segment=2)
_SLOT = struct.Struct('<qQQQqIIQ')
_I64 = struct.Struct('<q')
_U64 = struct.Struct('<Q')
_FREE_OFFSET, _OCCUPIED_OFFSET, _GEN_OFFSET = 40, 48, 32
_HEADER_BYTES = 128


class PoolBusy(TimeoutError):
    pass


class StaleDescriptor(ValueError):
    pass


class SharedPublicationPool:
    """Single producer / one receiver process, with thread-safe API calls.

    Multiple reader threads may claim independent keys. The OS lock excludes
    processes and a local RLock excludes threads sharing the same file handle.
    Cancellation is explicit producer acknowledgement: already published control
    records remain readable, while ordinary queued audio is retired. The caller
    must first stop generating new ordinary records for that generation.
    """

    def __init__(self, namespace, *, slots=4096, payload_bytes=4096, key_bytes=256, directory='/dev/shm', timeout=5.0):
        if not namespace or len(namespace) > 200:
            raise ValueError('a nonempty unique run namespace is required')
        if not 2 <= slots <= 65536 or not 128 <= payload_bytes <= 1048576:
            raise ValueError('invalid bounded pool dimensions')
        if not 16 <= key_bytes <= 4096 or timeout < 0:
            raise ValueError('invalid key capacity or timeout')
        self.slots, self.payload_bytes, self.key_bytes = slots, payload_bytes, key_bytes
        self.timeout = timeout
        self.path = Path(directory) / ('emp-pub-' + hashlib.sha256(namespace.encode()).hexdigest()[:24] + '.pool')
        self._mutex = threading.RLock()
        self.counts = Counter()
        self._cancelled = set()
        self._owners = {}
        self._closed = False
        self._slot_base = _HEADER_BYTES + slots * 8
        self._stride = _SLOT.size + key_bytes + payload_bytes
        size = self._slot_base + slots * self._stride
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            fresh = os.fstat(self.fd).st_size == 0
            if fresh:
                os.ftruncate(self.fd, size)
            elif os.fstat(self.fd).st_size != size:
                raise ValueError('pool dimensions differ between endpoints')
            self.memory = mmap.mmap(self.fd, size)
            if fresh:
                epoch = uuid.uuid4().int & ((1 << 64) - 1)
                _HEADER.pack_into(self.memory, 0, _MAGIC, 1, slots, payload_bytes, key_bytes, epoch, 0, 0, 0)
                for index in range(slots):
                    _I64.pack_into(self.memory, _HEADER_BYTES + index * 8, -1)
                    _SLOT.pack_into(
                        self.memory, self._offset(index), index + 1 if index + 1 < slots else -1, 0, 0, 0, 0, 0, 0, 0
                    )
            h = _HEADER.unpack_from(self.memory, 0)
            if h[:5] != (_MAGIC, 1, slots, payload_bytes, key_bytes):
                raise ValueError('pool protocol or shape mismatch')
            self.epoch = h[5]
        except BaseException:
            if hasattr(self, 'memory'):
                self.memory.close()
            os.close(self.fd)
            raise
        finally:
            if not getattr(self, '_closed', False):
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                except OSError:
                    pass

    def _offset(self, index):
        return self._slot_base + index * self._stride

    @contextmanager
    def _lock(self):
        with self._mutex:
            if self._closed:
                raise RuntimeError('publication pool is closed')
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _bucket(self, key):
        return _HEADER_BYTES + (zlib.crc32(key) % self.slots) * 8

    def _find(self, key):
        link = self._bucket(key)
        index = _I64.unpack_from(self.memory, link)[0]
        traversed = 0
        while index != -1:
            if not 0 <= index < self.slots or traversed >= self.slots:
                raise RuntimeError('corrupt publication pool index')
            off = self._offset(index)
            h = _SLOT.unpack_from(self.memory, off)
            if h[4] == 0:
                raise RuntimeError('free publication slot appears in index')
            start = off + _SLOT.size
            if h[5] == len(key) and self.memory[start : start + h[5]] == key:
                return index, link, h
            link, index = off, h[0]
            traversed += 1
        return None

    def _descriptor(self, index, h):
        return {
            'epoch': self.epoch,
            'slot': index,
            'slot_generation': h[1],
            'request_generation': h[2],
            'chunk_sequence': h[3],
        }

    def _live_descriptor(self, descriptor):
        if descriptor['epoch'] != self.epoch or not 0 <= descriptor['slot'] < self.slots:
            return False
        h = _SLOT.unpack_from(self.memory, self._offset(descriptor['slot']))
        return h[4] != 0 and h[1:4] == (
            descriptor['slot_generation'],
            descriptor['request_generation'],
            descriptor['chunk_sequence'],
        )

    def _retire(self, index, link, h):
        _I64.pack_into(self.memory, link, h[0])
        free = _I64.unpack_from(self.memory, _FREE_OFFSET)[0]
        _SLOT.pack_into(self.memory, self._offset(index), free, h[1], 0, 0, 0, 0, 0, 0)
        _I64.pack_into(self.memory, _FREE_OFFSET, index)
        occupied = _U64.unpack_from(self.memory, _OCCUPIED_OFFSET)[0]
        _U64.pack_into(self.memory, _OCCUPIED_OFFSET, occupied - 1)

    @staticmethod
    def key_parts(key):
        rid, stage, sequence = key.rsplit('_', 2)
        if not rid or stage != '0' or int(sequence) < 0:
            raise ValueError('pool only accepts talker stage 0 chunk keys')
        return rid, int(sequence)

    def put(self, key, payload, *, request_generation, flags=0, timeout=None):
        key_bytes = key.encode('utf-8')
        rid, sequence = self.key_parts(key)
        if len(key_bytes) > self.key_bytes or len(payload) > self.payload_bytes:
            raise ValueError('payload or key exceeds fixed slot capacity')
        if type(payload) is not bytes or type(request_generation) is not int or request_generation <= 0:
            raise ValueError('immutable bytes and positive request generation required')
        if flags & ~3:
            raise ValueError('unknown chunk flags')
        owner = rid, request_generation
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        waited = False
        while True:
            with self._lock():
                if owner in self._cancelled and not flags:
                    self.counts['rejected_cancelled_audio'] += 1
                    return None
                previous = self._owners.get(rid)
                if previous is not None and request_generation < previous[0]:
                    self.counts['rejected_old_generations'] += 1
                    raise StaleDescriptor('request generation precedes the current owner')
                old_live = (
                    previous is not None
                    and previous[0] != request_generation
                    and any(self._live_descriptor(d) for d in previous[1])
                )
                existing = self._find(key_bytes)
                free = _I64.unpack_from(self.memory, _FREE_OFFSET)[0]
                if not old_live and existing is None and free != -1:
                    off = self._offset(free)
                    old = _SLOT.unpack_from(self.memory, off)
                    bucket = self._bucket(key_bytes)
                    head = _I64.unpack_from(self.memory, bucket)[0]
                    generation = _U64.unpack_from(self.memory, _GEN_OFFSET)[0] + 1
                    _U64.pack_into(self.memory, _GEN_OFFSET, generation)
                    start = off + _SLOT.size
                    self.memory[start : start + len(key_bytes)] = key_bytes
                    data_start = start + self.key_bytes
                    self.memory[data_start : data_start + len(payload)] = payload
                    h = (head, generation, request_generation, sequence, 1, len(key_bytes), len(payload), flags)
                    _SLOT.pack_into(self.memory, off, *h)
                    _I64.pack_into(self.memory, bucket, free)
                    _I64.pack_into(self.memory, _FREE_OFFSET, old[0])
                    occupied = _U64.unpack_from(self.memory, _OCCUPIED_OFFSET)[0] + 1
                    _U64.pack_into(self.memory, _OCCUPIED_OFFSET, occupied)
                    descriptor = self._descriptor(free, h)
                    if previous is None or previous[0] != request_generation:
                        self._owners[rid] = (request_generation, [descriptor])
                    else:
                        # Keep only live descriptors: state is bounded by slots.
                        previous[1][:] = [d for d in previous[1] if self._live_descriptor(d)]
                        previous[1].append(descriptor)
                    if len(self._owners) > self.slots * 2:
                        self._owners = {
                            r: (g, live)
                            for r, (g, entries) in self._owners.items()
                            if (live := [d for d in entries if self._live_descriptor(d)])
                        }
                    self.counts['puts'] += 1
                    self.counts['bytes'] += len(payload)
                    self.counts['maximum_occupied'] = max(self.counts['maximum_occupied'], occupied)
                    if waited:
                        self.counts['backpressured_puts'] += 1
                    return descriptor
                self.counts['backpressure_polls'] += 1
            waited = True
            if time.monotonic() >= deadline:
                self.counts['backpressure_timeouts'] += 1
                raise PoolBusy('publication pool full, duplicate key, or prior request generation undrained')
            time.sleep(0.0002)

    def get(self, key, decoder, *, descriptor=None):
        key_bytes = key.encode('utf-8')
        with self._lock():
            found = self._find(key_bytes)
            if found is None:
                self.counts['empty_polls'] += 1
                return None
            index, _, h = found
            current = self._descriptor(index, h)
            if descriptor is not None and descriptor != current:
                self.counts['stale_descriptors'] += 1
                raise StaleDescriptor('descriptor does not own this key generation')
            if h[4] != 1:
                self.counts['claimed_polls'] += 1
                return None
            _SLOT.pack_into(self.memory, self._offset(index), *(*h[:4], 2, *h[5:]))
            start = self._offset(index) + _SLOT.size + self.key_bytes
            payload = bytes(self.memory[start : start + h[6]])
        try:
            result = decoder(payload)
        except BaseException:
            with self._lock():
                found = self._find(key_bytes)
                if found and self._descriptor(found[0], found[2]) == current:
                    hh = found[2]
                    _SLOT.pack_into(self.memory, self._offset(index), *(*hh[:4], 1, *hh[5:]))
                self.counts['decode_exceptions'] += 1
            raise
        with self._lock():
            found = self._find(key_bytes)
            if not found or self._descriptor(found[0], found[2]) != current:
                raise RuntimeError('claimed slot was reused before receiver acknowledgement')
            self._retire(*found)
            self.counts['gets'] += 1
            self.counts['acknowledged_bytes'] += len(payload)
        return result, len(payload), current

    def cancel(self, rid, request_generation):
        """Producer cancellation; preserve published segment and EOS controls.

        Claimed records remain owned until decoder completion. This call must be
        made after the caller has stopped ordinary production for this owner.
        """
        owner = rid, request_generation
        with self._lock():
            self._cancelled.add(owner)
            if len(self._cancelled) > self.slots * 2:
                self._cancelled = {
                    o for o in self._cancelled if o[0] in self._owners and self._owners[o[0]][0] == o[1]
                }
            previous = self._owners.get(rid)
            if previous is None or previous[0] != request_generation:
                return 0
            removed = 0
            for descriptor in previous[1]:
                if not self._live_descriptor(descriptor):
                    continue
                off = self._offset(descriptor['slot'])
                h = _SLOT.unpack_from(self.memory, off)
                if h[4] != 1 or h[7]:
                    continue
                key = bytes(self.memory[off + _SLOT.size : off + _SLOT.size + h[5]])
                found = self._find(key)
                self._retire(*found)
                removed += 1
            self.counts['cancelled_ordinary_records'] += removed
            return removed

    def status(self):
        with self._lock():
            return {
                'enabled': True,
                'protocol': 1,
                'path': str(self.path),
                'epoch': self.epoch,
                'slots': self.slots,
                'payload_bytes': self.payload_bytes,
                'occupied': _U64.unpack_from(self.memory, _OCCUPIED_OFFSET)[0],
                'counts': dict(self.counts),
                'retained_owner_entries': len(self._owners),
                'retained_cancel_entries': len(self._cancelled),
                'owner_entry_limit': self.slots * 2,
            }

    def close(self):
        with self._mutex:
            if not self._closed:
                self.memory.close()
                os.close(self.fd)
                self._closed = True

    def unlink(self):
        """Only call after both endpoints have stopped; never on request cleanup."""
        self.path.unlink(missing_ok=True)


class PooledConnector:
    """Instance-scoped connector proxy; all unrelated edges use the original."""

    def __init__(self, original, pool, role):
        self.original, self.pool, self.role = original, pool, role
        self._local = threading.local()
        self._fallback_generations = {}
        self._next_generation = 1 << 48

    def __getattr__(self, name):
        return getattr(self.original, name)

    @contextmanager
    def publication_context(self, owner, flags=0):
        previous = getattr(self._local, 'context', None)
        self._local.context = (owner, flags)
        try:
            yield
        finally:
            self._local.context = previous

    def put(self, from_stage, to_stage, put_key, data):
        if self.role != 'sender' or (str(from_stage), str(to_stage)) != ('0', '1'):
            return self.original.put(from_stage, to_stage, put_key, data)
        context = getattr(self._local, 'context', None)
        rid, seq = self.pool.key_parts(put_key)
        if context is None:
            if rid not in self._fallback_generations:
                self._next_generation += 1
                self._fallback_generations[rid] = self._next_generation
            owner, flags = (rid, self._fallback_generations[rid]), 0
            if len(self._fallback_generations) > self.pool.slots * 2:
                self._fallback_generations = {
                    r: g for r, g in self._fallback_generations.items() if r in self.pool._owners
                }
            self.pool.counts['fallback_context_puts'] += 1
        else:
            owner, flags = context
            if owner[0] != rid:
                raise ValueError('publication context does not match wire key')
        payload = self.original.serialize_obj(data)
        descriptor = self.pool.put(put_key, payload, request_generation=owner[1], flags=flags)
        if descriptor is None:
            return False, 0, None
        return True, len(payload), {'pooled_publication': descriptor, 'size': len(payload)}

    def get(self, from_stage, to_stage, get_key, metadata=None):
        if self.role != 'receiver' or (str(from_stage), str(to_stage)) != ('0', '1'):
            return self.original.get(from_stage, to_stage, get_key, metadata)
        if isinstance(metadata, dict) and get_key in metadata:
            metadata = metadata[get_key]
        descriptor = metadata.get('pooled_publication') if isinstance(metadata, dict) else None
        result = self.pool.get(get_key, self.original.deserialize_obj, descriptor=descriptor)
        if result is None:
            return None
        obj, size, receipt = result
        return obj, size

    def cleanup(self, request_id):
        # Lifecycle cleanup must not invalidate already published terminal audio.
        # Sender cancellation uses the explicit generation-tagged controller hook.
        return self.original.cleanup(request_id)

    def close(self):
        self.pool.close()
        self.original.close()

    def health(self):
        return {**self.original.health(), 'pooled_publication': self.pool.status()}


def install_pooled_publication(target, *, role, controller=None):
    """Call for BOTH scheduler adapters before requests; default is disabled.

    For the sender, pass its BulkSender as ``controller`` to attach immutable
    owner/sequence context and generation-tagged cancellation.
    """
    if os.environ.get('EASYMAGPIE_POOLED_PUBLICATION', '0') != '1':
        return None
    if role not in ('sender', 'receiver'):
        raise ValueError('invalid connector role')
    adapter = getattr(target, 'chunk_transfer_adapter', target)
    if adapter is None:
        raise ValueError('scheduler has no chunk transfer adapter')
    if isinstance(adapter.connector, PooledConnector):
        return adapter.connector
    if role == 'sender' and controller is None:
        raise ValueError('sender pool requires BulkSender lifecycle ownership')
    original = adapter.connector
    expected_stage = 0 if role == 'sender' else 1
    if original.stage_id != expected_stage or type(original).__name__ != 'SharedMemoryConnector':
        raise ValueError('pool is limited to the local SharedMemoryConnector talker/codec pair')
    namespace = os.environ.get('EASYMAGPIE_PUBLICATION_NAMESPACE', '')
    pool = SharedPublicationPool(
        namespace,
        slots=int(os.environ.get('EASYMAGPIE_PUBLICATION_SLOTS', '4096')),
        payload_bytes=int(os.environ.get('EASYMAGPIE_PUBLICATION_SLOT_BYTES', '4096')),
        timeout=float(os.environ.get('EASYMAGPIE_PUBLICATION_TIMEOUT_S', '5')),
    )
    proxy = PooledConnector(original, pool, role)
    adapter.connector = proxy
    if role == 'sender':
        publish = controller.publish_record

        def publish_record(record):
            flags = int(bool(record.get('is_finished'))) | (int(bool(record.get('is_segment_finished'))) << 1)
            with proxy.publication_context(record['owner'], flags):
                return publish(record)

        controller.publish_record = publish_record
        release = controller._release

        def release_state(state):
            from vllm.v1.request import RequestStatus

            if getattr(state.request, 'status', None) in (
                RequestStatus.FINISHED_ABORTED,
                RequestStatus.FINISHED_ERROR,
            ):
                pool.cancel(state.request.external_req_id, state.generation)
            return release(state)

        controller._release = release_state
    return proxy
