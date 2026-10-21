"""Ready-chunk ordering using startup deadlines and an ESTIMATED playback budget.

This module never changes payloads or runs a model.  The default audio clock is
connector-publication time plus valid-code-frame duration, NOT PCM delivery or
client playback.

Only explicit registration creates a generation.  Scheduler request retirement
does not retire sender ownership: terminal publication acknowledgement or explicit
cancellation does, and only after queued/inflight publications have drained.
"""

from __future__ import annotations

import heapq
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _Item:
    record: dict
    sequence: int
    serial: int
    ready_ns: int


@dataclass
class _Generation:
    owner: tuple[str, int]
    request: Any
    arrival_ns: int
    anchor_scope: str
    ready: deque = field(default_factory=deque)
    inflight: _Item | None = None
    next_sequence: int = 0
    last_dispatched: int = -1
    closed: bool = False
    cancelled: bool = False
    terminal_acknowledged: bool = False
    first_publication_ns: int | None = None
    estimated_end_ns: int | None = None
    estimated_samples: int = 0
    priority_version: int = 0
    head_serial: int = -1


class ReadyDeadlineQueue:
    """Thread-safe metadata queue; payload publication itself occurs outside its lock.

    Each owner is (external_request_id, monotonically assigned generation).
    The builder must register each generation exactly once and must not re-register
    a retired generation.  Offers and acknowledgements cannot create state.  Queued
    record dictionaries and request objects are strongly retained through ack.
    """

    def __init__(
        self,
        *,
        startup_ms=150.0,
        reserve_ms=80.0,
        max_ready_wait_ms=50.0,
        max_consecutive_owner=1,
        sample_rate=22050,
        samples_per_frame=1764,
        clock: Callable[[], int] = time.monotonic_ns,
    ):
        values = (startup_ms, reserve_ms, max_ready_wait_ms)
        if any(not isinstance(x, (int, float)) or not 0 <= x < 60000 for x in values):
            raise ValueError('deadline intervals must be finite nonnegative milliseconds below 60000')
        if max_ready_wait_ms == 0:
            raise ValueError('max_ready_wait_ms must be positive')
        if type(max_consecutive_owner) is not int or max_consecutive_owner < 1:
            raise ValueError('max_consecutive_owner must be a positive integer')
        if any(type(x) is not int or x <= 0 for x in (sample_rate, samples_per_frame)):
            raise ValueError('sample rate and frame sample count must be positive integers')
        self.startup_ns = int(startup_ms * 1_000_000)
        self.reserve_ns = int(reserve_ms * 1_000_000)
        self.max_ready_wait_ns = int(max_ready_wait_ms * 1_000_000)
        self.max_consecutive_owner = max_consecutive_owner
        self.sample_rate, self.samples_per_frame = sample_rate, samples_per_frame
        self.clock = clock
        self._states = {}
        self._rid_owners = {}
        self._heads = {}
        self._deadline_heap = []
        self._age_heap = []
        self._fifo_heap = []
        self._queued_count = 0
        self._lock = threading.RLock()
        self._serial = 0
        self._last_owner = None
        self._owner_streak = 0
        self._closed = False
        self.counters = Counter()
        self.anchor_scopes = Counter()

    @staticmethod
    def _owner(owner):
        if (
            type(owner) is not tuple
            or len(owner) != 2
            or type(owner[0]) is not str
            or type(owner[1]) is not int
            or owner[1] <= 0
        ):
            raise ValueError('owner must be (external request id, positive generation integer)')
        return owner

    def known(self, owner):
        with self._lock:
            return owner in self._states

    def register(self, owner, request, *, arrival_ns, anchor_scope='sender-enqueue'):
        owner = self._owner(owner)
        if type(arrival_ns) is not int or arrival_ns < 0:
            raise ValueError('arrival_ns must be a nonnegative monotonic timestamp')
        if type(anchor_scope) is not str or not anchor_scope:
            raise ValueError('anchor_scope must be explicit')
        with self._lock:
            if self._closed:
                raise RuntimeError('deadline queue is closed')
            if owner in self._states:
                raise ValueError('generation is already registered')
            self._states[owner] = _Generation(owner, request, arrival_ns, anchor_scope)
            self._rid_owners.setdefault(owner[0], deque()).append(owner)
            self.counters['registered_generations'] += 1
            self.anchor_scopes[anchor_scope] += 1
            self.counters['maximum_live_generations'] = max(
                self.counters['maximum_live_generations'], len(self._states)
            )

    @staticmethod
    def _control(record):
        return bool(record.get('is_finished') or record.get('is_segment_finished'))

    def offer(self, record):
        owner = self._owner(record['owner'])
        ready_ns, frames = record['ready_ns'], record['produced_frames']
        if type(ready_ns) is not int or ready_ns < 0:
            raise ValueError('ready_ns must be a nonnegative monotonic timestamp')
        if type(frames) is not int or frames < 0:
            raise ValueError('produced_frames must be a nonnegative integer')
        with self._lock:
            state = self._states.get(owner)
            generation_request = record.get('generation_request', record['request'])
            if state is None or state.request is not generation_request:
                self.counters['stale_ready_records'] += 1
                return False
            if state.closed:
                self.counters['records_after_close'] += 1
                return False
            self._serial += 1
            item = _Item(record, state.next_sequence, self._serial, ready_ns)
            state.next_sequence += 1
            state.ready.append(item)
            self._queued_count += 1
            self._refresh_head(state)
            state.closed = bool(record.get('is_finished'))
            self.counters['offered_records'] += 1
            self.counters['offered_frames'] += frames
            self.counters['maximum_queued_records'] = max(self.counters['maximum_queued_records'], self._queued())
            return True

    def offer_many(self, records):
        # Validate all records individually; rejected stale records remain counted.
        return sum(self.offer(record) for record in records)

    def _queued(self):
        return self._queued_count

    def _refresh_head(self, state):
        # The connector key uses external ID, so a reused ID must not publish
        # its new generation before the old generation's terminal/cancel drain.
        owners = self._rid_owners.get(state.owner[0])
        if state.ready and state.inflight is None and owners and owners[0] == state.owner:
            serial = state.ready[0].serial
            if self._heads.get(state.owner) is state and state.head_serial == serial:
                return
            state.priority_version += 1
            state.head_serial = serial
            self._heads[state.owner] = state
            suffix = (serial, state.owner, state.priority_version)
            heapq.heappush(self._deadline_heap, (self._deadline(state)[0], *suffix))
            heapq.heappush(self._age_heap, (state.ready[0].ready_ns, *suffix))
            heapq.heappush(self._fifo_heap, (serial, *suffix))
            self._compact_heaps_if_needed()
        else:
            self._heads.pop(state.owner, None)

    def _peek_valid(self, heap):
        while heap:
            _, serial, owner, version = heap[0]
            state = self._heads.get(owner)
            if state is not None and state.priority_version == version and state.head_serial == serial:
                return state
            heapq.heappop(heap)
        return None

    def _compact_heaps_if_needed(self):
        limit = 4 * len(self._heads) + 1024
        if max(len(self._deadline_heap), len(self._age_heap), len(self._fifo_heap)) <= limit:
            return
        self._deadline_heap.clear()
        self._age_heap.clear()
        self._fifo_heap.clear()
        for state in self._heads.values():
            suffix = (state.head_serial, state.owner, state.priority_version)
            self._deadline_heap.append((self._deadline(state)[0], *suffix))
            self._age_heap.append((state.ready[0].ready_ns, *suffix))
            self._fifo_heap.append((state.head_serial, *suffix))
        for heap in (self._deadline_heap, self._age_heap, self._fifo_heap):
            heapq.heapify(heap)
        self.counters['heap_compactions'] += 1

    def _deadline(self, state):
        if state.estimated_end_ns is not None:
            return state.estimated_end_ns - self.reserve_ns, 'estimated_playback'
        return state.arrival_ns + self.startup_ns, 'startup'

    def take_next(self, now_ns=None):
        now = self.clock() if now_ns is None else now_ns
        with self._lock:
            if not self._heads:
                return None
            oldest = self._peek_valid(self._fifo_heap)
            oldest_ready = self._peek_valid(self._age_heap)
            if now - oldest_ready.ready[0].ready_ns >= self.max_ready_wait_ns:
                chosen = oldest_ready
                reason = 'age_guard'
            else:
                chosen = self._peek_valid(self._deadline_heap)
                if (
                    self._owner_streak >= self.max_consecutive_owner
                    and len(self._heads) > 1
                    and chosen.owner == self._last_owner
                ):
                    excluded = heapq.heappop(self._deadline_heap)
                    chosen = self._peek_valid(self._deadline_heap)
                    heapq.heappush(self._deadline_heap, excluded)
                    self.counters['owner_cap_choices'] += 1
                reason = self._deadline(chosen)[1]
            competing = len(self._heads) > 1
            item = chosen.ready.popleft()
            self._queued_count -= 1
            self._heads.pop(chosen.owner)
            if item.sequence <= chosen.last_dispatched:
                self.counters['fifo_violations'] += 1
                raise RuntimeError('per-generation publication order violated')
            chosen.last_dispatched = item.sequence
            chosen.inflight = item
            self.counters['selected_records'] += 1
            self.counters['choice:' + reason] += 1
            if competing:
                self.counters['competing_choices'] += 1
            if chosen is not oldest:
                self.counters['reordered_choices'] += 1
            if chosen.owner == self._last_owner:
                self._owner_streak += 1
            else:
                self._last_owner, self._owner_streak = chosen.owner, 1
            self.counters['maximum_ready_wait_ns'] = max(
                self.counters['maximum_ready_wait_ns'], max(0, now - item.ready_ns)
            )
            if now > self._deadline(chosen)[0]:
                self.counters['choices_past_deadline'] += 1
            return item.record

    def _retire_if_drained(self, state):
        if (
            state.closed
            and (state.cancelled or state.terminal_acknowledged)
            and not state.ready
            and state.inflight is None
        ):
            self._states.pop(state.owner)
            self._heads.pop(state.owner, None)
            owners = self._rid_owners[state.owner[0]]
            owners.remove(state.owner)
            if owners:
                self._refresh_head(self._states[owners[0]])
            else:
                self._rid_owners.pop(state.owner[0])
            self.counters['retired_generations'] += 1
            if state.terminal_acknowledged:
                self.counters['terminal_retirements'] += 1
            if not self._states:
                self._deadline_heap.clear()
                self._age_heap.clear()
                self._fifo_heap.clear()
                self._last_owner, self._owner_streak = None, 0

    def complete(self, record, success, now_ns=None):
        now = self.clock() if now_ns is None else now_ns
        with self._lock:
            state = self._states.get(record['owner'])
            if state is None or state.inflight is None or state.inflight.record is not record:
                self.counters['stale_acknowledgements'] += 1
                return False
            state.inflight = None
            self.counters['acknowledged_records'] += 1
            if success:
                self.counters['successful_publications'] += 1
                frames = record['produced_frames']
                if frames > 0 and not state.cancelled:
                    samples = frames * self.samples_per_frame
                    duration_ns = samples * 1_000_000_000 // self.sample_rate
                    if state.first_publication_ns is None:
                        state.first_publication_ns = now
                        self.counters['first_audio_publications'] += 1
                        self.counters['first_publication_latency_ns_sum'] += max(0, now - state.arrival_ns)
                    state.estimated_end_ns = max(state.estimated_end_ns or now, now) + duration_ns
                    state.estimated_samples += samples
                    self.counters['estimated_published_samples'] += samples
                if record.get('is_finished'):
                    state.terminal_acknowledged = True
                    state.closed = True
            elif state.cancelled:
                self.counters['cancelled_publications'] += 1
            else:
                # Match the pinned adapter's per-record failure isolation:
                # never advance this record's sample budget, but do not discard
                # other ready records.  Any put failure invalidates the run.
                self.counters['failed_publications'] += 1
            self._refresh_head(state)
            self._retire_if_drained(state)
            return True

    def cancel(self, owner, reason='cancelled'):
        """Drop unpublished ordinary audio, retaining captured EOS/segment controls.

        A put already running remains owned until its acknowledgement.  Existing
        terminal/segment markers preserve the bulk adapter's abort behavior.
        """
        with self._lock:
            state = self._states.get(owner)
            if state is None:
                self.counters['cancellation_after_retirement'] += 1
                return False
            if not state.cancelled:
                self.counters['cancelled_generations'] += 1
                self.counters['cancel_reason:' + reason] += 1
            state.cancelled = state.closed = True
            controls = deque(item for item in state.ready if self._control(item.record))
            self.counters['cancelled_queued_records'] += len(state.ready) - len(controls)
            self._queued_count -= len(state.ready) - len(controls)
            state.ready = controls
            self._refresh_head(state)
            self._retire_if_drained(state)
            return True

    def close(self):
        """Close only after sender drain; live states are reported, never erased."""
        with self._lock:
            self._closed = True
            self.counters['close_calls'] += 1
            return not self._states

    def status(self):
        with self._lock:
            return {
                'enabled': True,
                'version': 1,
                'closed': self._closed,
                'policy': 'ready-chunk startup / ESTIMATED playback budget with FIFO heads and age guard',
                'playback_clock': 'connector-publication estimate',
                'client_playback_observed': False,
                'observation_transport_installed': False,
                'startup_budget_ms': self.startup_ns / 1_000_000,
                'reserve_ms': self.reserve_ns / 1_000_000,
                'max_ready_wait_ms': self.max_ready_wait_ns / 1_000_000,
                'max_consecutive_owner': self.max_consecutive_owner,
                'sample_rate': self.sample_rate,
                'samples_per_frame': self.samples_per_frame,
                'anchor_scopes': dict(self.anchor_scopes),
                'counters': dict(self.counters),
                'live_generations': len(self._states),
                'queued_records': self._queued(),
                'external_ids_retained': len(self._rid_owners),
                'ready_owner_heads': len(self._heads),
                'index_entries': {
                    'deadline': len(self._deadline_heap),
                    'age': len(self._age_heap),
                    'fifo': len(self._fifo_heap),
                },
                'blocked_reused_id_generations': sum(
                    bool(state.ready) and self._rid_owners[state.owner[0]][0] != state.owner
                    for state in self._states.values()
                ),
                'inflight_records': sum(state.inflight is not None for state in self._states.values()),
                'retained_closed_generations': sum(state.closed for state in self._states.values()),
            }


def install_deadline_policy(controller):
    """Install on the bulk builder before its first generation is registered.

    The builder calls `deadline_policy.register(...)` at generation creation and
    `cancel(...)` for cancellations/fallback.  This wrapper operates on one ready
    list at a time, with no batching wait and no change to payload serialization.
    """
    if os.environ.get('EASYMAGPIE_READY_CHUNK_DEADLINES', '0') != '1':
        return None
    if getattr(controller, 'deadline_policy', None) is not None or controller.states:
        raise RuntimeError('install ready deadlines once, before generation creation')
    policy = ReadyDeadlineQueue(
        startup_ms=float(os.environ.get('EASYMAGPIE_READY_STARTUP_MS', '150')),
        reserve_ms=float(os.environ.get('EASYMAGPIE_READY_RESERVE_MS', '80')),
        max_ready_wait_ms=float(os.environ.get('EASYMAGPIE_READY_MAX_WAIT_MS', '50')),
        max_consecutive_owner=int(os.environ.get('EASYMAGPIE_READY_MAX_CONSECUTIVE', '1')),
    )
    controller.deadline_policy = policy
    original_status = controller.status

    def publish_ready(records):
        policy.counters['ready_lists'] += 1
        policy.counters['maximum_ready_list'] = max(policy.counters['maximum_ready_list'], len(records))
        policy.offer_many(records)
        while (record := policy.take_next()) is not None:
            try:
                success = controller.publish_record(record)
            except Exception:
                policy.counters['publication_exceptions'] += 1
                policy.complete(record, False)
                continue
            policy.complete(record, success)

    def status():
        result = original_status()
        result['ready_deadlines'] = policy.status()
        result['ready_deadlines']['reordering_scope'] = 'one built ready list; no cross-envelope lookahead'
        result['ready_deadlines']['counter_scope'] = 'whole worker boot including warmups and canaries'
        return result

    controller.publish_ready = publish_ready
    controller.status = status
    return policy
