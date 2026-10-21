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
"""Execute the startup module with CPU-only runtime stand-ins."""
import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'easymagpie_vllm_omni/optimized_workers.py'


def load_workers():
    events = []
    state = SimpleNamespace(inference=False, fail_base=False, fail_prewarm=False)

    class Base:
        def __init__(self):
            self.model_runner = SimpleNamespace(model=SimpleNamespace(prewarm_codec_graphs=self.prewarm))

        def init_device(self):
            events.append(('init_device',))
            return 'device'

        def load_model(self, *args, **kwargs):
            events.append(('load_model', args, kwargs))
            return 'model'

        def compile_or_warm_up_model(self):
            events.append(('base_warmup',))
            if state.fail_base:
                raise RuntimeError('base warmup failed')
            return 7.25

        def prewarm(self, batches, *, frame_sizes, descending=False):
            assert state.inference
            events.append(('codec_prewarm', tuple(batches), frame_sizes))
            if descending:
                events.append(('descending',))
            if state.fail_prewarm:
                raise RuntimeError('codec prewarm failed')

    @contextmanager
    def inference_mode():
        assert not state.inference
        state.inference = True
        events.append(('inference_enter',))
        try:
            yield
        finally:
            events.append(('inference_exit',))
            state.inference = False

    torch = SimpleNamespace(
        set_float32_matmul_precision=lambda value: events.append(('precision', value)),
        inference_mode=inference_mode,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
    )
    namespace = dict(
        os=os,
        sys=SimpleNamespace(setswitchinterval=lambda value: events.append(('switch_interval', value))),
        torch=torch,
        gc=SimpleNamespace(freeze=lambda: events.append(('freeze',))),
        EasyMagpieGPUARWorker=Base,
        EasyMagpieCodecGPUGenerationWorker=Base,
    )
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace, events, state, torch


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE', raising=False)
    monkeypatch.delenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', raising=False)
    return load_workers()


def test_precision_precedes_each_parent_call_and_returns_are_preserved(workers):
    module, events, _, torch = workers
    worker = module['H100OptimizedTalkerWorker']()
    assert worker.init_device() == 'device'
    assert worker.load_model('checkpoint', revision=3) == 'model'
    assert worker.compile_or_warm_up_model() == 7.25
    assert events == [
        ('precision', 'highest'),
        ('init_device',),
        ('precision', 'highest'),
        ('load_model', ('checkpoint',), {'revision': 3}),
        ('precision', 'highest'),
        ('base_warmup',),
    ]
    assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32


@pytest.mark.parametrize('value', [None, '0', 'false'])
def test_codec_disabled_path_delegates_without_extra_graphs_or_gc(workers, monkeypatch, value):
    module, events, _, _ = workers
    if value is not None:
        monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', value)
    worker = module['H100OptimizedCodecWorker']()
    assert worker.compile_or_warm_up_model() == 7.25
    assert events == [('precision', 'highest'), ('base_warmup',)]
    assert not hasattr(worker, 'codec_prewarm_completed')


def test_enabled_codec_prewarm_finishes_exact192_shapes_before_freeze(workers, monkeypatch):
    module, events, state, _ = workers
    monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', '1')
    worker = module['H100OptimizedCodecWorker']()
    assert worker.compile_or_warm_up_model() == 7.25
    assert events == [
        ('precision', 'highest'),
        ('base_warmup',),
        ('inference_enter',),
        ('codec_prewarm', tuple(range(1, 65)), (2, 6, 8)),
        ('inference_exit',),
        ('freeze',),
    ]
    assert worker.codec_prewarm_completed and not state.inference


def test_talker_never_runs_codec_prewarm_even_with_codec_flag(workers, monkeypatch):
    module, events, _, _ = workers
    monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', '1')
    worker = module['H100OptimizedTalkerWorker']()
    worker.compile_or_warm_up_model()
    assert events == [('precision', 'highest'), ('base_warmup',)]


@pytest.mark.parametrize('operation', ['init_device', 'load_model', 'compile_or_warm_up_model'])
def test_external_tf32_override_rejected_before_parent_mutation(workers, monkeypatch, operation):
    module, events, _, _ = workers
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    with pytest.raises(RuntimeError, match='Unset NVIDIA_TF32_OVERRIDE'):
        getattr(module['H100OptimizedCodecWorker'](), operation)()
    assert not events


@pytest.mark.parametrize('failure', ['fail_base', 'fail_prewarm'])
def test_failure_cannot_publish_success_or_freeze_gc(workers, monkeypatch, failure):
    module, events, state, _ = workers
    monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', '1')
    setattr(state, failure, True)
    worker = module['H100OptimizedCodecWorker']()
    with pytest.raises(RuntimeError, match='failed'):
        worker.compile_or_warm_up_model()
    assert not hasattr(worker, 'codec_prewarm_completed')
    assert ('freeze',) not in events and not state.inference
    if failure == 'fail_base':
        assert not any(event[0] == 'codec_prewarm' for event in events)


@pytest.mark.parametrize('blocking', ['0', '1'])
def test_c256_talker_sets_copy_event_policy_and_switch_interval(workers, monkeypatch, blocking):
    module, events, _, torch = workers
    monkeypatch.setenv('EASYMAGPIE_TALKER_BLOCKING_COPY_EVENT', blocking)
    worker = module['H100C256TalkerWorker']()
    assert worker.init_device() == 'device'
    assert worker.load_model('checkpoint') == 'model'
    assert events[:3] == [('precision', 'highest'), ('switch_interval', 0.001), ('init_device',)]
    assert worker.model_runner._omni_snapshot_blocking_event == (blocking == '1')
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_c256_invalid_copy_flag_rejected_before_loading_model(workers, monkeypatch):
    module, events, _, _ = workers
    monkeypatch.setenv('EASYMAGPIE_TALKER_BLOCKING_COPY_EVENT', 'true')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        module['H100C256TalkerWorker']().load_model()
    assert not events


def test_c256_codec_captures384_shapes_descending_with_retained_precision(workers, monkeypatch):
    module, events, _, torch = workers
    monkeypatch.setenv('EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS', '1')
    worker = module['H100C256CodecWorker']()
    worker.init_device()
    assert worker.compile_or_warm_up_model() == 7.25
    assert events[-5:] == [
        ('inference_enter',),
        ('codec_prewarm', tuple(range(1, 129)), (2, 6, 8)),
        ('descending',),
        ('inference_exit',),
        ('freeze',),
    ]
    assert not any(event[0] == 'switch_interval' for event in events)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation
    assert not torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert worker.codec_prewarm_completed
