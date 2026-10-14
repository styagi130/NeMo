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
"""Execute the actual pool class with delayed CPU stream/event stand-ins.

The queues deliberately retain references until retirement, so early buffer
reuse corrupts the destination. This is ownership/exception evidence, not a
CUDA driver or pinned-memory performance test.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'easymagpie_vllm_omni/dds_routing_pool.py'


class Device:
    def __init__(self, value):
        self.value = value.value if isinstance(value, Device) else str(value)
        self.type = self.value.split(':')[0]

    def __eq__(self, other):
        return isinstance(other, Device) and self.value == other.value


class Stream:
    def __init__(self):
        self.operations = []

    def drain(self):
        work, self.operations = self.operations, []
        for operation in work:
            operation()


def actual_pool():
    runtime = NS(current=Stream(), capture=False, fail_copy=None, fail_index=None, fail_record=False, query_calls=0)

    class Tensor:
        def __init__(self, n, device='cuda:0', storage=None, offset=0):
            self.storage = [0] * n if storage is None else storage
            self.offset, self.n = offset, n
            self.device, self.dtype, self.ndim = Device(device), 'long', 1

        def is_contiguous(self):
            return True

        def __getitem__(self, key):
            assert isinstance(key, slice) and key.step in (None, 1)
            start, stop, _ = key.indices(self.n)
            return Tensor(stop - start, self.device, self.storage, self.offset + start)

        def __setitem__(self, key, values):
            start, stop, _ = key.indices(self.n)
            assert stop - start == len(values)
            self.storage[self.offset + start : self.offset + stop] = values

        def values(self):
            return self.storage[self.offset : self.offset + self.n]

        def numpy(self):
            return self

        def copy_(self, source, non_blocking=False):
            assert non_blocking
            if runtime.fail_copy == 'before':
                raise RuntimeError('copy before')
            runtime.current.operations.append(lambda: self.__setitem__(slice(None), source.values()))
            if runtime.fail_copy == 'after':
                raise RuntimeError('copy after')

        def index_copy_(self, dimension, rows, slots):
            assert dimension == 0
            if runtime.fail_index == 'before':
                raise RuntimeError('index before')

            def apply():
                for row, slot in zip(rows.values(), slots.values(), strict=True):
                    self.storage[self.offset + row] = slot

            runtime.current.operations.append(apply)
            if runtime.fail_index == 'after':
                raise RuntimeError('index after')

    class Event:
        def __init__(self):
            self.done = False

        def query(self):
            runtime.query_calls += 1
            return self.done

        def record(self, stream):
            if runtime.fail_record:
                raise RuntimeError('record failure')
            self.done = False
            stream.operations.append(lambda: setattr(self, 'done', True))

    torch = NS(
        Tensor=Tensor,
        long='long',
        device=Device,
        empty=lambda n, *, dtype, device, pin_memory: Tensor(n, device),
        empty_like=lambda source, *, device: Tensor(source.n, device),
        cuda=NS(
            device=lambda _: nullcontext(),
            Event=Event,
            is_current_stream_capturing=lambda: runtime.capture,
            current_stream=lambda _: runtime.current,
        ),
    )
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.Import) or not any(a.name == 'torch' for a in n.names)]
    namespace = {'torch': torch}
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace['RoutingUploadPool']('cuda:0', 8), runtime, Tensor


def test_four_pending_entries_are_never_reused_and_busy_falls_back_without_wait():
    pool, runtime, Tensor = actual_pool()
    target = Tensor(8)
    for i in range(4):
        assert pool.try_upload(target, {i: 20 + i})
    pending = [entry['host'].values() for entry in pool.entries]
    assert not pool.try_upload(target, {7: 99})
    assert [entry['host'].values() for entry in pool.entries] == pending
    assert target.values() == [0] * 8
    queries = runtime.query_calls
    assert pool.status()['busy_fallbacks'] == 1 and runtime.query_calls == queries
    runtime.current.drain()
    assert target.values() == [20, 21, 22, 23, 0, 0, 0, 0]
    assert pool.try_upload(target, {7: 99})
    runtime.current.drain()
    assert target.values()[7] == 99
    assert pool.counts['retired_reuses'] == 1


def test_retired_slot_can_move_streams_without_reusing_pending_peers():
    pool, runtime, Tensor = actual_pool()
    a = runtime.current
    b = Stream()
    target = Tensor(8)
    assert pool.try_upload(target, {0: 11})
    runtime.current = b
    for i in range(1, 4):
        assert pool.try_upload(target, {i: 11 + i})
    assert not pool.try_upload(target, {4: 55})
    a.drain()
    assert pool.try_upload(target, {4: 55})  # Slot0 now retires onto streamB.
    assert pool.counts['retired_reuses'] == 1
    b.drain()
    assert target.values()[:5] == [11, 12, 13, 14, 55]


@pytest.mark.parametrize(
    'operation,when', [('copy', 'before'), ('copy', 'after'), ('index', 'before'), ('index', 'after')]
)
def test_enqueue_failures_protect_staging_until_recorded_event_settles(operation, when):
    pool, runtime, Tensor = actual_pool()
    target = Tensor(8)
    setattr(runtime, 'fail_' + operation, when)
    with pytest.raises(RuntimeError):
        pool.try_upload(target, {0: 41})
    assert pool.entries[0]['busy'] and pool.entries[0]['recorded']
    assert not pool.entries[0]['event'].done
    setattr(runtime, 'fail_' + operation, None)
    for i in range(1, 4):
        assert pool.try_upload(target, {i: 50 + i})
    assert not pool.try_upload(target, {7: 99})
    runtime.current.drain()
    assert pool.try_upload(target, {7: 99})
    runtime.current.drain()
    assert target.values()[7] == 99


def test_event_record_failure_poisoned_slot_stays_unavailable_even_after_stream_drain():
    pool, runtime, Tensor = actual_pool()
    target = Tensor(8)
    runtime.fail_record = True
    with pytest.raises(RuntimeError, match='record failure'):
        pool.try_upload(target, {0: 40})
    assert pool.entries[0]['busy'] and not pool.entries[0]['recorded']
    runtime.fail_record = False
    runtime.current.drain()
    assert target.values()[0] == 40
    for i in range(1, 4):
        assert pool.try_upload(target, {i: i})
    assert not pool.try_upload(target, {7: 70})
    runtime.current.drain()
    assert pool.try_upload(target, {7: 70})
    assert pool.entries[0]['busy'] and not pool.entries[0]['recorded']
    assert pool.next_slot == 2  # Reused slot1; poisoned slot0 was skipped.


@pytest.mark.parametrize('case', ['capture', 'empty', 'oversized', 'dtype', 'rank', 'device'])
def test_fallbacks_make_no_queue_or_host_staging_changes(case):
    pool, runtime, Tensor = actual_pool()
    target = Tensor(12)
    routing = {0: 3}
    if case == 'capture':
        runtime.capture = True
    elif case == 'empty':
        routing = {}
    elif case == 'oversized':
        routing = {i: i for i in range(9)}
    elif case == 'dtype':
        target.dtype = 'float'
    elif case == 'rank':
        target.ndim = 2
    else:
        target.device = Device('cpu')
    before = [entry['host'].values() for entry in pool.entries]
    assert not pool.try_upload(target, routing)
    assert not runtime.current.operations and not any(entry['busy'] for entry in pool.entries)
    assert before == [entry['host'].values() for entry in pool.entries]
