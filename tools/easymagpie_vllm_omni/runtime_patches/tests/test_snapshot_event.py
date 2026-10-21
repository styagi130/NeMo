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
"""Run exact upstream snapshot functions with CPU stream/event stand-ins.

Use the installed pinned distribution or EASYMAGPIE_RUNTIME_ORIGINALS. Neither
path imports torch/vLLM or modifies the installation. These tests check host
control flow and source ownership; actual CUDA copy correctness is a GPU gate.
"""

import ast
import hashlib
import importlib.metadata
import importlib.util
import os
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

TOOL = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('snapshot_patch_installer', TOOL / 'scripts/apply_runtime_patches.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
RUNTIME_PATH = 'vllm_omni/worker/gpu_ar_model_runner.py'


def sources():
    manifest, patches = installer._load_bundle()
    row = next(item for item in manifest['files'] if item['path'] == RUNTIME_PATH)
    root = os.environ.get('EASYMAGPIE_RUNTIME_ORIGINALS')
    if root:
        path = Path(root) / RUNTIME_PATH
    else:
        try:
            path = Path(importlib.metadata.distribution('vllm-omni').locate_file(RUNTIME_PATH))
        except importlib.metadata.PackageNotFoundError:
            raise unittest.SkipTest('Requires pinned runtime sources or EASYMAGPIE_RUNTIME_ORIGINALS') from None
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest == row['original_sha256']:
        original = content
        patched = installer._apply_hunks(original, patches[RUNTIME_PATH])
    else:
        if digest != row['patched_sha256']:
            raise AssertionError('Unsupported snapshot runtime source')
        patched = content
        original = patched.replace(b'    blocking: bool = False,\n', b'', 1)
        original = original.replace(
            b'    ready_event = torch.cuda.Event(blocking=True) if blocking else torch.cuda.Event()\n',
            b'    ready_event = torch.cuda.Event()\n',
            1,
        ).replace(b'                blocking=getattr(self, "_omni_snapshot_blocking_event", False),\n', b'', 1)
    assert hashlib.sha256(original).hexdigest() == row['original_sha256']
    assert hashlib.sha256(patched).hexdigest() == row['patched_sha256']
    return original.decode(), patched.decode()


def runtime(source):
    events, created = [], []
    state = SimpleNamespace(fail_copy=False, fail_wait=False)

    class Tensor:
        def __init__(self, value, device='cuda'):
            self.value = value
            self.device = SimpleNamespace(type=device)

        def detach(self):
            events.append(('detach', self.device.type))
            return self

        def clone(self):
            events.append(('clone', self.device.type))
            return Tensor(self.value, self.device.type)

        def copy_(self, other, *, non_blocking):
            events.append(('copy', non_blocking))
            if state.fail_copy:
                raise RuntimeError('copy failed')
            self.value = other.value
            return self

        def __getitem__(self, key):
            assert key == slice(None, 0)
            return Tensor((), self.device.type)

    class Event:
        def __init__(self, **kwargs):
            events.append(('event', kwargs))
            self.kwargs = kwargs
            created.append(self)

        def record(self, stream):
            assert stream is copy_stream
            events.append(('record',))

        def synchronize(self):
            events.append(('synchronize',))
            if state.fail_wait:
                raise RuntimeError('wait failed')

    source_stream = object()

    class Stream:
        def wait_stream(self, other):
            assert other is source_stream
            events.append(('wait_stream',))

    copy_stream = Stream()

    @contextmanager
    def stream(value):
        assert value is copy_stream
        events.append(('enter_stream',))
        try:
            yield
        finally:
            events.append(('exit_stream',))

    def empty_like(value, *, device, pin_memory):
        assert device == 'cpu'
        events.append(('empty_like', pin_memory))
        return Tensor(None, device)

    torch = SimpleNamespace(
        Tensor=Tensor,
        empty_like=empty_like,
        cuda=SimpleNamespace(Event=Event, Stream=Stream, current_stream=lambda: source_stream, stream=stream),
    )
    tree = ast.parse(source)
    names = {
        '_clone_cuda_tensor_payload',
        '_copy_tensor_payload_to_cpu',
        '_AsyncCPUPayloadSnapshot',
        '_snapshot_tensor_payload_to_cpu_async',
        '_OmniOutputTensorSnapshot',
    }
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert len(nodes) == len(names)
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GPUARModelRunner')
    method = next(
        node for node in runner.body if getattr(node, 'name', None) == '_snapshot_omni_output_tensors_for_async_output'
    )
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    executable = ast.fix_missing_locations(ast.Module(body=[future, *nodes, method], type_ignores=[]))
    namespace = dict(
        torch=torch, Any=Any, NamedTuple=NamedTuple, record_function_or_nullcontext=lambda _: nullcontext()
    )
    namespace['is_pin_memory_available'] = lambda: True
    exec(compile(executable, '<pinned-snapshot-runtime>', 'exec'), namespace)
    return namespace, events, created, state, Tensor, copy_stream


class SnapshotEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.patched = sources()

    def test_default_preserves_complete_original_operation_sequence(self):
        sequences = []
        for source in (self.original, self.patched):
            namespace, events, _, _, Tensor, stream = runtime(source)
            snapshot = namespace['_snapshot_tensor_payload_to_cpu_async'](
                {'gpu': Tensor(7), 'cpu': Tensor(9, 'cpu')}, copy_stream=stream, pin_memory=True
            )
            self.assertEqual(snapshot.payload['gpu'].value, 7)
            self.assertEqual(snapshot.payload['cpu'].value, 9)
            self.assertEqual(snapshot.payload['gpu'].device.type, 'cpu')
            snapshot.wait()
            snapshot.wait()
            sequences.append(events)
        self.assertEqual(*sequences)

    def test_blocking_changes_only_event_constructor_and_preserves_source_lifetime(self):
        sequences = []
        for blocking in (False, True):
            namespace, events, created, _, Tensor, stream = runtime(self.patched)
            original = Tensor(13)
            snapshot = namespace['_snapshot_tensor_payload_to_cpu_async'](
                [original], copy_stream=stream, pin_memory=True, blocking=blocking
            )
            self.assertEqual(created[0].kwargs, {'blocking': True} if blocking else {})
            self.assertEqual(len(snapshot._cuda_sources), 1)
            self.assertIsNot(snapshot._cuda_sources[0], original)
            self.assertFalse(snapshot._waited)
            self.assertNotIn(('synchronize',), events)
            snapshot.wait()
            snapshot.wait()
            self.assertEqual(snapshot._cuda_sources, [])
            self.assertTrue(snapshot._waited)
            self.assertEqual(events.count(('synchronize',)), 1)
            sequences.append([event for event in events if event[0] != 'event'])
        self.assertEqual(*sequences)

    def test_cpu_only_payload_creates_no_event_or_stream_work(self):
        namespace, events, created, _, Tensor, stream = runtime(self.patched)
        value = Tensor(3, 'cpu')
        snapshot = namespace['_snapshot_tensor_payload_to_cpu_async'](
            (value, 4), copy_stream=stream, pin_memory=True, blocking=True
        )
        self.assertIsNot(snapshot.payload[0], value)
        self.assertEqual(snapshot.payload[0].value, 3)
        self.assertIsNone(snapshot._ready_event)
        snapshot.wait()
        self.assertEqual(created, [])
        self.assertEqual(events, [('detach', 'cpu'), ('clone', 'cpu')])

    def test_wait_failure_keeps_sources_until_a_successful_wait(self):
        namespace, _, _, state, Tensor, stream = runtime(self.patched)
        snapshot = namespace['_snapshot_tensor_payload_to_cpu_async'](
            Tensor(4), copy_stream=stream, pin_memory=True, blocking=True
        )
        state.fail_wait = True
        with self.assertRaisesRegex(RuntimeError, 'wait failed'):
            snapshot.wait()
        self.assertEqual(len(snapshot._cuda_sources), 1)
        self.assertFalse(snapshot._waited)
        state.fail_wait = False
        snapshot.wait()
        self.assertEqual(snapshot._cuda_sources, [])

    def test_copy_failure_propagates_without_record_or_wait(self):
        namespace, events, _, state, Tensor, stream = runtime(self.patched)
        state.fail_copy = True
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            namespace['_snapshot_tensor_payload_to_cpu_async'](
                Tensor(4), copy_stream=stream, pin_memory=True, blocking=True
            )
        self.assertNotIn(('record',), events)
        self.assertNotIn(('synchronize',), events)
        self.assertEqual(events[-1], ('exit_stream',))

    def test_runner_opt_in_is_instance_local_and_sync_path_does_not_snapshot(self):
        namespace, _, created, _, Tensor, stream = runtime(self.patched)
        method = namespace['_snapshot_omni_output_tensors_for_async_output']
        for enabled in (None, True, False):
            owner = SimpleNamespace(
                _build_omni_async_snapshot_payload=lambda **kwargs: kwargs,
                _get_or_create_omni_payload_copy_stream=lambda: stream,
            )
            if enabled is not None:
                owner._omni_snapshot_blocking_event = enabled
            result = method(
                owner,
                use_async_omni_output=True,
                hidden_states=Tensor(1),
                staged_hidden_states_cpu=None,
                multimodal_outputs={'audio': Tensor(2)},
            )
            self.assertEqual(created[-1].kwargs, {'blocking': True} if enabled else {})
            self.assertIsNotNone(result.async_payload)
            result.async_payload.wait()
            before = len(created)
            hidden, payload = Tensor(5), {'audio': Tensor(6)}
            result = method(
                owner,
                use_async_omni_output=False,
                hidden_states=hidden,
                staged_hidden_states_cpu=None,
                multimodal_outputs=payload,
            )
            self.assertIs(result.hidden_states, hidden)
            self.assertIs(result.multimodal_outputs, payload)
            self.assertIsNone(result.async_payload)
            self.assertEqual(len(created), before)


if __name__ == '__main__':
    unittest.main()
