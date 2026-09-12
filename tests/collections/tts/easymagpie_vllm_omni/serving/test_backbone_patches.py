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
"""Tests for narrow vLLM Nemotron-H compatibility patches."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import easymagpie_vllm_omni.backbone_patches as backbone_patches  # noqa: E402
from easymagpie_vllm_omni.backbone_patches import patch_shared_expert_activation  # noqa: E402
from vllm.model_executor.layers.activation import ReLUSquaredActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    GroupedTopKRouter,
    fused_grouped_topk,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402


class NemotronHMoE:
    def __init__(self, activation):
        self.shared_experts = SimpleNamespace(act_fn=activation)


def _backbone(activation: str, current_activation):
    layer = SimpleNamespace(mixer=NemotronHMoE(current_activation))
    return SimpleNamespace(config=SimpleNamespace(mlp_hidden_act=activation), layers=[layer])


def _relu_squared_without_vllm_context():
    activation = ReLUSquaredActivation.__new__(ReLUSquaredActivation)
    torch.nn.Module.__init__(activation)
    return activation


def test_shared_expert_activation_is_read_from_config():
    backbone = _backbone("silu", _relu_squared_without_vllm_context())

    assert patch_shared_expert_activation(backbone) == 1
    torch.testing.assert_close(
        backbone.layers[0].mixer.shared_experts.act_fn(torch.tensor([-1.0, 0.0, 1.0])),
        torch.nn.functional.silu(torch.tensor([-1.0, 0.0, 1.0])),
    )
    assert patch_shared_expert_activation(backbone) == 0


def test_shared_expert_patch_rejects_unknown_upstream_implementation():
    backbone = _backbone("silu", torch.nn.Identity())

    with pytest.raises(RuntimeError, match="implementation changed"):
        patch_shared_expert_activation(backbone)


def _router_backbone(monkeypatch, device="cpu", num_experts=24, top_k=4):
    # Exercise the real GateLinear.forward without a distributed model constructor.
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    generator = torch.Generator().manual_seed(729)
    gate.weight = torch.nn.Parameter(
        torch.randn(num_experts, 1536, generator=generator, dtype=torch.float16).to(device) / 32
    )
    gate.bias = None
    gate.skip_bias_add = False
    gate.return_bias = True
    gate.quant_method = UnquantizedLinearMethod()
    gate.out_dtype = torch.float32
    for name in ("ll_bf16", "dsv3_router", "fp32_router", "bf16x3_router", "cublas_router"):
        setattr(gate, f"allow_{name}_gemm", False)
    router = GroupedTopKRouter(
        top_k=top_k,
        global_num_experts=num_experts,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(num_experts, device=device),
    )
    mixer = NemotronHMoE(None)
    mixer.gate = gate
    mixer.experts = SimpleNamespace(
        router=router, is_monolithic=False, moe_config=SimpleNamespace(router_logits_dtype=torch.float32)
    )
    monkeypatch.setattr(
        backbone_patches,
        "platforms",
        SimpleNamespace(current_platform=SimpleNamespace(is_cuda=lambda: True)),
        raising=False,
    )
    monkeypatch.setattr(backbone_patches, "envs", SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=True), raising=False)
    return SimpleNamespace(layers=[SimpleNamespace(mixer=mixer)]), gate, router


@pytest.mark.parametrize("num_experts,top_k", [(2, 1), (24, 4), (128, 8)])
def test_router_cast_elision_is_idempotent(monkeypatch, num_experts, top_k):
    backbone, gate, _ = _router_backbone(monkeypatch, num_experts=num_experts, top_k=top_k)
    patch = backbone_patches.patch_moe_router_logit_cast

    assert patch(backbone) == 1
    assert gate.out_dtype == torch.float16
    assert backbone.layers[0].mixer.experts.moe_config.router_logits_dtype == torch.float16
    assert patch(backbone) == 0


@pytest.mark.parametrize(
    "unsupported",
    [
        "non_cuda",
        "unfused",
        "bf16",
        "fp32",
        "already_half",
        "no_output_cast",
        "monolithic",
        "missing_config",
        "router_dtype",
        "unknown_gate",
        "unknown_quant",
        "unknown_router",
        "softmax",
        "no_bias",
        "multiple_groups",
        "no_groups",
        "topk_group",
        "generic_expert_count",
        "generic_topk",
        "zero_topk",
        "too_many_topk",
        "ll_bf16",
        "dsv3_router",
        "fp32_router",
        "bf16x3_router",
        "cublas_router",
    ],
)
def test_router_cast_elision_keeps_unsupported_paths(monkeypatch, unsupported):
    backbone, gate, router = _router_backbone(monkeypatch)
    experts = backbone.layers[0].mixer.experts
    config = experts.moe_config
    if unsupported == "non_cuda":
        backbone_patches.platforms.current_platform.is_cuda = lambda: False
    elif unsupported == "unfused":
        backbone_patches.envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK = False
    elif unsupported in ("bf16", "fp32"):
        gate.weight = torch.nn.Parameter(gate.weight.to(torch.bfloat16 if unsupported == "bf16" else torch.float32))
    elif unsupported in ("already_half", "no_output_cast"):
        gate.out_dtype = torch.float16 if unsupported == "already_half" else None
    elif unsupported == "monolithic":
        experts.is_monolithic = True
    elif unsupported == "missing_config":
        del experts.moe_config
    elif unsupported == "router_dtype":
        config.router_logits_dtype = torch.bfloat16
    elif unsupported == "unknown_gate":
        backbone.layers[0].mixer.gate = SimpleNamespace(**vars(gate))
    elif unsupported == "unknown_quant":
        gate.quant_method = SimpleNamespace()
    elif unsupported == "unknown_router":
        experts.router = SimpleNamespace(**vars(router))
    elif unsupported == "softmax":
        router.scoring_func = "softmax"
    elif unsupported == "no_bias":
        router.e_score_correction_bias = None
    elif unsupported in ("multiple_groups", "no_groups"):
        router.num_expert_group = 2 if unsupported == "multiple_groups" else 0
    elif unsupported == "topk_group":
        router.topk_group = 2
    elif unsupported == "generic_expert_count":
        gate.weight = torch.nn.Parameter(torch.zeros(129, 1536, dtype=torch.float16))
    elif unsupported in ("generic_topk", "zero_topk", "too_many_topk"):
        router.top_k = {"generic_topk": 9, "zero_topk": 0, "too_many_topk": 25}[unsupported]
    else:
        setattr(gate, f"allow_{unsupported}_gemm", True)
    original_dtype = gate.out_dtype
    original_router_dtype = config.router_logits_dtype

    patch = backbone_patches.patch_moe_router_logit_cast
    assert patch(backbone) == 0
    assert gate.out_dtype == original_dtype
    assert config.router_logits_dtype == original_router_dtype


@pytest.mark.parametrize("num_rows", [1, 32, 64, 128])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_router_cast_elision_preserves_cpu_gate_logits(monkeypatch, num_rows, input_dtype):
    import vllm.model_executor.layers.linear as linear

    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda: lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias),
    )
    backbone, gate, _ = _router_backbone(monkeypatch)
    x = torch.randn(num_rows, 1536, generator=torch.Generator().manual_seed(31)).to(input_dtype)
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as before:
        reference, reference_bias = gate(x)
    patch = backbone_patches.patch_moe_router_logit_cast
    assert patch(backbone) == 1
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as after:
        actual, actual_bias = gate(x)

    assert reference.dtype == torch.float32 and actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), reference, rtol=0, atol=0)
    assert actual_bias is reference_bias is None
    counts = [
        sum(event.count for event in profile.key_averages() if event.key == "aten::_to_copy")
        for profile in (before, after)
    ]
    assert counts[0] - counts[1] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped router required")
@pytest.mark.parametrize("num_rows", [1, 16, 32, 64, 128, 513])
@pytest.mark.parametrize("pattern", ["random", "ties", "close_scores"])
def test_router_cast_elision_preserves_cuda_expert_ids_and_weights(monkeypatch, num_rows, pattern):
    backbone, gate, router = _router_backbone(monkeypatch, device="cuda")
    x = torch.randn(num_rows, 1536, device="cuda", dtype=torch.float16)
    if pattern == "ties":
        x.zero_()
    elif pattern == "close_scores":
        x.mul_(1e-4)
    if pattern == "random":
        router.e_score_correction_bias.copy_(torch.linspace(-0.125, 0.125, 24, device="cuda"))
    elif pattern == "close_scores":
        router.e_score_correction_bias.copy_(torch.linspace(-1e-6, 1e-6, 24, device="cuda"))

    def route():
        logits, _ = gate(x)
        weights, ids = fused_grouped_topk(
            x,
            logits,
            router.top_k,
            True,
            router.e_score_correction_bias,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
        )
        return logits, weights, ids

    with torch.no_grad():
        reference_logits, reference_weights, reference_ids = route()
        patch = backbone_patches.patch_moe_router_logit_cast
        assert patch(backbone) == 1
        actual_logits, actual_weights, actual_ids = route()
    torch.testing.assert_close(actual_logits.float(), reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, reference_weights, rtol=0, atol=0)
    assert torch.equal(actual_ids, reference_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped router required")
@pytest.mark.parametrize("num_rows", [1, 32, 64, 128])
def test_router_cast_elision_preserves_cuda_graph_replay(monkeypatch, num_rows):
    backbone, gate, router = _router_backbone(monkeypatch, device="cuda")
    reference_gate = deepcopy(gate)
    x = torch.randn(num_rows, 1536, device="cuda", dtype=torch.float16)

    def route(gate):
        logits, _ = gate(x)
        return fused_grouped_topk(
            x,
            logits,
            router.top_k,
            True,
            router.e_score_correction_bias,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
        )

    with torch.no_grad():
        patch = backbone_patches.patch_moe_router_logit_cast
        assert patch(backbone) == 1
        assert reference_gate.out_dtype == torch.float32
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                route(gate)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_weights, actual_ids = route(gate)
        for step in range(3):
            if step < 2:
                x.normal_()
            else:
                x.zero_()
            graph.replay()
            reference_weights, reference_ids = route(reference_gate)
            torch.testing.assert_close(actual_weights, reference_weights, rtol=0, atol=0)
            assert torch.equal(actual_ids, reference_ids)
