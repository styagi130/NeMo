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
"""Exact query dispatch through the pinned runner CPU-span handoff."""

from itertools import accumulate
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni import easymagpie
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@pytest.mark.parametrize("lengths", [(3, 7), (1, 3), (1, 1, 1, 3), (71, 1), (1, 3, 1), (1, 1)])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dispatch_matches_existing_rows(lengths, graphs, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, context, spans = _model(lengths, graphs=graphs, device=device)
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    expected = model._get_query_dispatch()
    actual = model._get_query_dispatch(spans)
    _assert_dispatch(actual, expected)


@pytest.mark.parametrize("count", [0, 1, 3, 32, 64, 128])
def test_forward_avoids_dynamic_indices_for_leading_decode_rows(count, monkeypatch):
    lengths = [1] * count + [3, 7]
    model, context, spans = _model(lengths)
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model(**_inputs(sum(lengths)), request_token_spans=spans)
    assert not any(event.key == "aten::nonzero" for event in profile.key_averages())
    assert model._out_codes[:count, 0].tolist() == list(range(1000, 1000 + count))
    assert not model._out_codes[count:].any()


@pytest.mark.parametrize(
    "spans",
    [
        None,
        [],
        [(0, 3)],
        [(1, 4), (4, 11)],
        [(0, 0), (0, 10)],
        [(0, 3), (4, 11)],
        [(0, 3), (3, 9)],
        [(0.0, 3), (3, 10)],
        [(False, 3), (3, 10)],
        [(0, 3, 4), (3, 10)],
        "invalid",
    ],
)
def test_invalid_spans_keep_existing_path(spans, monkeypatch):
    model, context, _ = _model([3, 7])
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    expected = model._get_query_dispatch()
    _assert_dispatch(model._get_query_dispatch(spans), expected)


def test_metadata_absent_or_unavailable_keeps_dummy_path(monkeypatch):
    model, context, spans = _model([3, 7])
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    for metadata in (None, {}, {"mamba": SimpleNamespace(num_prefills=2, num_decodes=0)}):
        context.attn_metadata = metadata
        assert model._get_query_dispatch(spans) == (None, 0, None)


def test_microbatch_layout_keeps_existing_path(monkeypatch):
    model, context, spans = _model([1, 3])
    context.ubatch_slices = [object()]
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    expected = model._get_query_dispatch()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = model._get_query_dispatch(spans)
    _assert_dispatch(actual, expected)
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::nonzero") == 2


def test_real_runner_spans_are_owned_and_passed_to_each_forward(monkeypatch):
    model, context, _ = _model([1, 3])
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    runner = EasyMagpieGPUARModelRunner.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.model_config = SimpleNamespace(has_sampling_extra_args=False)
    runner._omni_query_start_loc_model_kwarg = False
    runner._sync_local_stage_payloads = lambda: None
    runner._gather_runtime_additional_information = lambda: []
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda self, **kwargs: self.model(**kwargs))
    owned = []
    for lengths in ([1, 3], [3, 1], [2, 2]):
        starts = np.array([0, *accumulate(lengths)])
        runner.input_batch = SimpleNamespace(req_ids=["a", "b"])
        runner.query_start_loc = SimpleNamespace(cpu=starts)
        runner._omni_num_scheduled_tokens_np = np.array(lengths)
        context.attn_metadata["attention"].query_start_loc = torch.tensor(starts, dtype=torch.int32)
        context.attn_metadata["attention"].max_query_len = max(lengths)
        spans = runner._build_model_kwargs_extra()["request_token_spans"]
        owned.append(spans)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            runner._model_forward(**_inputs(4))
        nonzero = sum(event.count for event in profile.key_averages() if event.key == "aten::nonzero")
        assert nonzero == (2 if lengths == [3, 1] else 0)
        starts.fill(-1)
        assert spans == [(0, lengths[0]), (lengths[0], 4)]
    assert owned == [[(0, 1), (1, 4)], [(0, 3), (3, 4)], [(0, 2), (2, 4)]]
    context.attn_metadata = None
    assert model._get_query_dispatch() == (None, 0, None)


def test_forward_without_spans_cannot_reuse_previous_layout(monkeypatch):
    model, context, spans = _model([1, 3])
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    model(**_inputs(4), request_token_spans=spans)
    context.attn_metadata["attention"].query_start_loc.copy_(torch.tensor([0, 3, 4]))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model(**_inputs(4))
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::nonzero") == 2
    assert model._out_codes[:, 0].tolist() == [0, 0, 0, 1003]


def test_forward_preserves_phoneme_rows_and_batch_descriptor(monkeypatch):
    model, context, spans = _model([1, 1, 1, 3, 7])
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    model.has_phoneme = True
    rows = []
    model._predict_phonemes = lambda hidden, indices: rows.append(indices.tolist())
    descriptor = context.batch_descriptor
    model(**_inputs(13), request_token_spans=spans)
    assert rows == [[0, 1, 2], [5, 12]]
    assert context.batch_descriptor is descriptor


def test_cuda_slice_capture_replays_updated_query_starts(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, context, spans = _model([1, 1, 1, 4, 8], device="cuda")
    monkeypatch.setattr(easymagpie, "get_forward_context", lambda: context)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model._get_query_dispatch(spans)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model._get_query_dispatch(spans)
    starts = context.attn_metadata["attention"].query_start_loc
    starts.copy_(torch.tensor([0, 1, 2, 3, 11, 15], device="cuda", dtype=starts.dtype))
    graph.replay()
    _assert_dispatch(captured, model._get_query_dispatch())
    context.attn_metadata = None
    assert model._get_query_dispatch() == (None, 0, None)


def _assert_dispatch(actual, expected):
    assert actual[1] == expected[1]
    for value, reference in ((actual[0], expected[0]), (actual[2], expected[2])):
        if reference is None:
            assert value is None
        else:
            assert value.dtype == reference.dtype and torch.equal(value, reference)


class _Backbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return inputs_embeds


def _model(lengths, graphs=True, device="cpu"):
    model = easymagpie.EasyMagpieTTSForConditionalGeneration.__new__(easymagpie.EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    mode = CUDAGraphMode.FULL_AND_PIECEWISE if graphs else CUDAGraphMode.NONE
    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode, cudagraph_capture_sizes=sizes)
    )
    total = sum(lengths)
    model._combined_embeddings = torch.zeros(total, 2, device=device)
    model._token_stop = torch.zeros(total, dtype=torch.bool, device=device)
    model._sample_stop = torch.zeros(total, dtype=torch.bool, device=device)
    model._out_codes = torch.zeros(total, 1, dtype=torch.long, device=device)
    model.has_phoneme = model._single_stage_audio = False
    model.backbone = _Backbone()
    model.code_predictor = SimpleNamespace(generate_codes=lambda hidden: hidden[:, :1].long() + 1000)
    model._assemble_decode_embeddings = lambda combined, idx: None
    model._flag_audio_eos = lambda codes, idx: None
    starts = [0, *accumulate(lengths)]
    metadata = {
        "mamba": SimpleNamespace(num_prefills=len(lengths), num_decodes=0),
        "attention": SimpleNamespace(
            max_query_len=max(lengths), query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device)
        ),
    }
    context = SimpleNamespace(attn_metadata=metadata, batch_descriptor=BatchDescriptor(num_tokens=total))
    return model, context, list(zip(starts, starts[1:]))


def _inputs(total):
    return {
        "input_ids": torch.zeros(total, dtype=torch.long),
        "positions": torch.arange(total),
        "inputs_embeds": torch.arange(total, dtype=torch.float32).unsqueeze(1).expand(total, 2),
    }
