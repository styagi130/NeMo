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
import ast
import bisect
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'easymagpie_vllm_omni'


class Device:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.split(':')[0]

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(self) == str(other)


class Tensor:
    def __init__(self, values, owner=None, start=0, stop=None):
        self.owner = list(values) if owner is None else owner
        self.start = start
        self.stop = len(self.owner) if stop is None else stop
        self.shape = (self.stop - self.start,)
        self.device = Device('cuda:0')

    def __getitem__(self, idx):
        assert isinstance(idx, slice)
        start, stop, step = idx.indices(self.shape[0])
        assert step == 1
        return Tensor([], self.owner, self.start + start, self.start + stop)

    def __setitem__(self, idx, values):
        start, stop, step = idx.indices(self.shape[0])
        assert step == 1
        self.owner[self.start + start : self.start + stop] = values

    def numpy(self):
        return self

    def copy_(self, value, non_blocking=False):
        if FAKE.copy_failure:
            raise RuntimeError('injected copy failure')
        self.owner[self.start : self.stop] = value.values()
        FAKE.log.append(('copy', non_blocking))
        return self

    def zero_(self):
        self.owner[self.start : self.stop] = [0] * self.shape[0]
        return self

    def values(self):
        return self.owner[self.start : self.stop]

    def numel(self):
        return self.shape[0]


class Event:
    def __init__(self):
        self.done = False
        self.recorded = False
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.done

    def record(self, stream):
        FAKE.log.append(('record', stream.cuda_stream))
        if FAKE.record_failure:
            raise RuntimeError('injected record failure')
        self.recorded = True
        self.done = False


class Fake:
    long = 'int64'

    def __init__(self):
        self.stream = SimpleNamespace(cuda_stream=1)
        self.capture = False
        self.copy_failure = self.record_failure = False
        self.log = []
        self.cuda = SimpleNamespace(
            device=lambda device: nullcontext(),
            Event=Event,
            current_stream=lambda device: self.stream,
            is_current_stream_capturing=lambda: self.capture,
        )

    device = staticmethod(Device)

    def empty(self, n, **kwargs):
        return Tensor([0] * n)

    def empty_like(self, t, **kwargs):
        return Tensor([0] * t.numel())

    def tensor(self, values, **kwargs):
        self.log.append(('legacy', list(values)))
        return Tensor(values)


FAKE = Fake()


def helper_class():
    tree = ast.parse((SOURCE / 'query_dispatch_pool.py').read_text())
    ns = {'torch': FAKE, 'Counter': Counter}
    exec(
        compile(
            ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), 'helper', 'exec'
        ),
        ns,
    )
    return ns['QueryDispatchPool']


def methods(source, names, context=None):
    tree = ast.parse((source / 'easymagpie.py').read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EasyMagpieTTSForConditionalGeneration'
    )
    ns = dict(
        torch=FAKE,
        bisect=bisect,
        CUDAGraphMode=SimpleNamespace(NONE=0),
        get_forward_context=lambda: context or SimpleNamespace(attn_metadata={'value': 1}),
        Optional=SimpleNamespace(),
        IntermediateTensors=object,
        Any=object,
    )
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.returns = None
        for arg in n.args.args:
            arg.annotation = None
        if n.args.kwarg:
            n.args.kwarg.annotation = None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'model-methods', 'exec'), ns)
    return ns


class Contracts(unittest.TestCase):
    def setUp(self):
        global FAKE
        FAKE = Fake()
        self.Pool = helper_class()
        self.pool = self.Pool('cuda:0', 128)

    def upload(self, decode=(0, 2), prefill=(70, 142)):
        self.pool.begin_step()
        return self.pool.try_upload(decode, prefill, Device('cuda:0'))

    def complete(self):
        self.pool.before_forward()
        self.pool.complete_forward()

    def test_constructor_bounds(self):
        for device, cap, slots in [('cpu', 128, 4), ('cuda:0', 0, 4), ('cuda:0', 128, 3), ('cuda:0', True, 4)]:
            with self.assertRaises(ValueError):
                self.Pool(device, cap, slots)

    def test_zero_offset_independent_storage_values(self):
        a, b = self.upload()
        self.assertEqual(a.values(), [0, 2])
        self.assertEqual(b.values(), [70, 142])
        self.assertEqual(a.start, 0)
        self.assertEqual(b.start, 0)
        self.assertIsNot(a.owner, b.owner)
        self.assertFalse(self.pool.current['recorded'])
        self.complete()
        self.assertEqual(FAKE.log[-1], ('record', 1))

    def test_pending_all_four_fallback_unchanged(self):
        owners = []
        for i in range(4):
            owners.append(self.upload((i,), (70 + i,)))
            self.complete()
        snapshots = [(a.values(), b.values()) for a, b in owners]
        self.assertIsNone(self.upload())
        self.assertEqual(self.pool.counts['busy_fallbacks'], 1)
        self.assertEqual(snapshots, [(a.values(), b.values()) for a, b in owners])

    def test_retired_reuse_other_stream(self):
        self.upload()
        self.complete()
        entry = self.pool.entries[0]
        entry['event'].done = True
        self.pool.next_slot = 0
        FAKE.stream = SimpleNamespace(cuda_stream=2)
        self.upload((4,), (9,))
        self.complete()
        self.assertEqual(self.pool.counts['retired_reuses'], 1)
        self.assertEqual(FAKE.log[-1], ('record', 2))

    def test_no_retirement_at_upload_completion(self):
        self.upload()
        entry = self.pool.current
        entry['event'].done = True
        self.pool.begin_step()
        self.pool.next_slot = 0
        self.upload((5,), (6,))
        self.assertIsNot(self.pool.current, entry)
        self.assertEqual(self.pool.counts['quarantined_steps'], 1)

    def test_exception_after_upload_quarantines(self):
        FAKE.copy_failure = True
        with self.assertRaises(RuntimeError):
            self.upload()
        entry = self.pool.current
        self.assertTrue(entry['busy'])
        self.assertFalse(entry['recorded'])
        FAKE.copy_failure = False
        self.pool.begin_step()
        self.pool.next_slot = 0
        self.upload()
        self.assertIsNot(self.pool.current, entry)

    def test_record_failure_quarantines(self):
        self.upload()
        self.pool.before_forward()
        FAKE.record_failure = True
        with self.assertRaises(RuntimeError):
            self.pool.complete_forward()
        entry = self.pool.current
        self.assertFalse(entry['recorded'])
        FAKE.record_failure = False
        self.pool.begin_step()
        self.pool.next_slot = 0
        self.upload()
        self.assertIsNot(self.pool.current, entry)

    def test_wrong_stream_capture_and_double_consumer_rejected(self):
        self.upload()
        FAKE.stream = SimpleNamespace(cuda_stream=2)
        with self.assertRaises(RuntimeError):
            self.pool.before_forward()
        FAKE.stream = SimpleNamespace(cuda_stream=1)
        FAKE.capture = True
        with self.assertRaises(RuntimeError):
            self.pool.before_forward()
        FAKE.capture = False
        self.pool.before_forward()
        with self.assertRaises(RuntimeError):
            self.pool.before_forward()

    def test_retire_requires_consumer_same_stream(self):
        self.upload()
        with self.assertRaises(RuntimeError):
            self.pool.complete_forward()
        self.pool.before_forward()
        FAKE.stream = SimpleNamespace(cuda_stream=3)
        with self.assertRaises(RuntimeError):
            self.pool.complete_forward()
        self.assertFalse(self.pool.current['recorded'])

    def test_unsupported_and_capture_fallback_before_mutation(self):
        for decode, prefill, dev in [([0] * 129, [1], 'cuda:0'), ([], [], 'cuda:0'), ([], [1], 'cpu')]:
            self.assertIsNone(self.pool.try_upload(decode, prefill, dev))
        FAKE.capture = True
        self.assertIsNone(self.upload())
        self.assertFalse(any(x['busy'] for x in self.pool.entries))
        self.assertFalse(FAKE.log)

    def test_status_no_event_queries_or_retirement(self):
        self.upload()
        self.complete()
        before = [e['event'].queries for e in self.pool.entries]
        state = self.pool.status()
        self.assertEqual(state['completed_forwards'], 1)
        self.assertEqual(before, [e['event'].queries for e in self.pool.entries])

    def test_actual_prepare_matches_disabled_path_varied_shapes_and_modes(self):
        candidate = methods(SOURCE, ['prepare_query_dispatch'])['prepare_query_dispatch']
        parent = methods(SOURCE, ['prepare_query_dispatch'])['prepare_query_dispatch']
        for mode in [0, 1]:
            for counts in [[], [1], [1, 1], [71], [71, 1], [1, 71, 1], [2, 6, 8], [1, 1, 71, 1, 1], [71] * 128]:
                for enabled in [False, True]:
                    with self.subTest(mode=mode, counts=counts, enabled=enabled):
                        config = SimpleNamespace(
                            compilation_config=SimpleNamespace(
                                cudagraph_mode=mode, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
                            )
                        )
                        c = SimpleNamespace(
                            vllm_config=config, _query_dispatch_pool=self.Pool('cuda:0', 128) if enabled else None
                        )
                        p = SimpleNamespace(vllm_config=config)
                        starts = []
                        offset = 0
                        for n in counts:
                            starts.append(offset)
                            offset += n
                        candidate(c, starts, counts, 'cuda:0')
                        parent(p, starts, counts, 'cuda:0')
                        if p._prepared_query_dispatch is None:
                            self.assertIsNone(c._prepared_query_dispatch)
                        else:
                            a, n, b = c._prepared_query_dispatch
                            x, m, y = p._prepared_query_dispatch
                            self.assertEqual((a.values(), n, b.values()), (x.values(), m, y.values()))

    def test_actual_prepare_busy_fallback_and_pure_decode(self):
        prepare = methods(SOURCE, ['prepare_query_dispatch'])['prepare_query_dispatch']
        m = SimpleNamespace(
            vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=0)),
            _query_dispatch_pool=self.pool,
        )
        for _ in range(4):
            prepare(m, [0], [71], 'cuda:0')
            self.complete()
        before = len(FAKE.log)
        prepare(m, [0, 71], [71, 1], 'cuda:0')
        self.assertEqual([x[0] for x in FAKE.log[before:]], ['legacy', 'legacy'])
        prepare(m, [0, 1], [1, 1], 'cuda:0')
        self.assertIsNone(m._prepared_query_dispatch)
        FAKE.capture = True
        self.pool.before_forward()
        self.pool.complete_forward()  # pure decode no event/runtime calls

    def test_actual_forward_last_consumer_then_event_and_exception(self):
        fn = methods(SOURCE, ['forward'])['forward']
        for fail in [False, True]:
            pool = self.Pool('cuda:0', 128)
            pool.begin_step()
            indices = pool.try_upload([], [70], 'cuda:0')

            def dispatch():
                pool.before_forward()
                return indices[0], 0, indices[1]

            def phonemes(hidden, idx):
                FAKE.log.append(('phoneme_consumer', idx.values()))
                if fail:
                    raise RuntimeError('injected phoneme failure')

            m = SimpleNamespace(
                _combined_embeddings=Tensor([0] * 71),
                _token_stop=Tensor([0] * 71),
                _sample_stop=Tensor([0] * 71),
                _get_query_dispatch=dispatch,
                _device_decode_state=None,
                _prefill_graphs_enabled=False,
                _out_codes=Tensor([0] * 71),
                backbone=lambda **kw: Tensor([3] * 71),
                has_phoneme=True,
                _predict_phonemes=phonemes,
                _query_dispatch_pool=pool,
            )
            if fail:
                with self.assertRaises(RuntimeError):
                    fn(m, Tensor([0] * 71), Tensor([0] * 71))
                self.assertIsNotNone(pool.current)
                self.assertFalse(pool.current['recorded'])
            else:
                fn(m, Tensor([0] * 71), Tensor([0] * 71))
                self.assertEqual(FAKE.log[-1], ('record', 1))
                self.assertLess(FAKE.log.index(('phoneme_consumer', [70])), len(FAKE.log) - 1)
