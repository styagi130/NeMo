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
"""Codec waveform transport and worker integration tests."""
from __future__ import annotations

from types import SimpleNamespace

import easymagpie_vllm_omni.runner as runner_module
import pytest
import torch
import yaml
from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.runner import (
    EasyMagpieCodecGPUGenerationModelRunner,
    EasyMagpieCodecGPUGenerationWorker,
    batch_waveforms_to_cpu,
)
from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_waveforms_preserve_bits_shapes_and_order(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    first = torch.tensor([0, -2147483648, 1065353216, 2143294004], dtype=torch.int32).view(torch.float32).view(2, 2)
    waveforms = [first, torch.empty(0), torch.tensor([[2.0, -1.0, 3.0]]).t()]
    outputs = [waveform.to(device) for waveform in waveforms]

    copied = batch_waveforms_to_cpu(outputs)

    assert [value.shape for value in copied] == [value.shape for value in waveforms]
    for actual, expected in zip(copied, waveforms, strict=True):
        assert actual.device.type == "cpu"
        assert actual.is_contiguous()
        assert torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize(
    "outputs",
    [
        None,
        [],
        [torch.ones(2)],
        (torch.ones(2), torch.ones(3)),
        [torch.ones(2), None],
        [torch.ones(2), torch.ones(3, dtype=torch.float64)],
        [torch.ones(2), torch.ones(3, device="meta")],
    ],
)
def test_unsupported_waveform_payloads_are_unchanged(outputs):
    assert batch_waveforms_to_cpu(outputs) is outputs


def test_empty_waveforms_and_autograd_are_safe():
    outputs = [torch.empty(0, requires_grad=True), torch.empty(1, 0, requires_grad=True)]
    copied = batch_waveforms_to_cpu(outputs)
    assert [value.shape for value in copied] == [value.shape for value in outputs]
    assert all(not value.requires_grad for value in copied)


def test_codec_runner_packs_before_upstream_request_dispatch(monkeypatch):
    runner = object.__new__(EasyMagpieCodecGPUGenerationModelRunner)
    waveforms = [torch.tensor([3.0, 4.0]), torch.tensor([1.0])]
    metadata = torch.tensor(22050)
    state = ExecuteModelState(*([None] * len(ExecuteModelState._fields)))._replace(
        multimodal_outputs={"model_outputs": waveforms, "sample_rate": metadata}
    )
    runner.execute_model_state = state
    runner.kv_connector_output = None
    runner.speculative_config = None
    runner._async_chunk = False
    runner.routed_experts_initialized = False
    runner.supports_mm_inputs = False
    runner.use_async_scheduling = False
    runner.input_batch = SimpleNamespace(
        num_reqs=2, req_ids=["second", "first"], req_id_to_index={"second": 0, "first": 1}
    )
    runner._should_accumulate_full_payload_output = lambda: False
    runner.get_omni_connector_output = lambda: None
    packed = []

    def record_copy(outputs):
        result = batch_waveforms_to_cpu(outputs)
        packed.append(result)
        return result

    monkeypatch.setattr(runner_module, "batch_waveforms_to_cpu", record_copy)

    result = runner.sample_tokens()

    assert len(packed) == 1
    assert packed[0] is not waveforms
    assert state.multimodal_outputs["model_outputs"] is waveforms
    assert runner.execute_model_state is None
    assert result.req_ids == ["second", "first"]
    for payload, expected in zip(result.multimodal_outputs, waveforms, strict=True):
        torch.testing.assert_close(payload["model_outputs"], expected)
        torch.testing.assert_close(payload["sample_rate"], metadata)


@pytest.mark.parametrize("payload", [None, [], {"model_outputs": [torch.ones(2)]}, {"other": torch.ones(2)}])
def test_codec_runner_preserves_unsupported_state_and_passes_grammar(monkeypatch, payload):
    runner = object.__new__(EasyMagpieCodecGPUGenerationModelRunner)
    state = (
        None
        if payload is None
        else ExecuteModelState(*([None] * len(ExecuteModelState._fields)))._replace(multimodal_outputs=payload)
    )
    runner.execute_model_state = state
    grammar = object()

    def sample_tokens(self, grammar_output):
        assert self.execute_model_state is state
        assert grammar_output is grammar
        return "upstream"

    monkeypatch.setattr(GPUGenerationModelRunner, "sample_tokens", sample_tokens)
    assert runner.sample_tokens(grammar) == "upstream"


@pytest.mark.parametrize("fail", [False, True])
def test_codec_worker_restores_upstream_runner_class(monkeypatch, fail):
    original = runner_module.gpu_generation_worker.GPUGenerationModelRunner
    worker = object.__new__(EasyMagpieCodecGPUGenerationWorker)

    def init_device(self):
        assert runner_module.gpu_generation_worker.GPUGenerationModelRunner is EasyMagpieCodecGPUGenerationModelRunner
        if fail:
            raise RuntimeError("init failed")
        return "initialized"

    monkeypatch.setattr(GPUGenerationWorker, "init_device", init_device)
    if fail:
        with pytest.raises(RuntimeError, match="init failed"):
            worker.init_device()
    else:
        assert worker.init_device() == "initialized"
    assert runner_module.gpu_generation_worker.GPUGenerationModelRunner is original


def test_two_stage_deployment_selects_codec_transfer_worker():
    deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / "easymagpie.yaml").read_text())
    codec_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 1)
    assert codec_stage["engine_extras"]["worker_cls"] == (
        "easymagpie_vllm_omni.runner.EasyMagpieCodecGPUGenerationWorker"
    )
