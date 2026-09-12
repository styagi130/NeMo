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
"""Tests for local-transformer sampling contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from conftest import build_vllm_config  # noqa: E402
from easymagpie_vllm_omni import local_transformer as local_transformer_module  # noqa: E402
from easymagpie_vllm_omni.config import EasyMagpieOmniArch  # noqa: E402
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor  # noqa: E402
from vllm.config import CUDAGraphMode  # noqa: E402

# Cover identity and linear projection paths.
ARCH_PROFILES = {
    "equal_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=64,
        local_transformer_hidden_dim=64,
        local_transformer_n_heads=4,
    ),
    "mixed_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=48,
        local_transformer_hidden_dim=80,
        local_transformer_n_heads=4,
    ),
}


def _build_predictor(profile_kwargs: dict):
    """Build an initialized code predictor and its derived architecture."""
    cfg = build_vllm_config(**profile_kwargs)
    arch = EasyMagpieOmniArch.from_hf_config(cfg.model_config.hf_config)

    cp = EasyMagpieCodePredictor(vllm_config=cfg, prefix="code_predictor").eval()
    cp.init_forbidden_mask()
    return cp, arch


@pytest.mark.unit
def test_generate_codes_shape_dtype_and_range():
    """``generate_codes`` returns valid (num_tokens, num_codebooks) int64 codes within vocab."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    num_tokens = 5

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(num_tokens, arch.hidden_dim))

    assert codes.shape == (num_tokens, arch.num_stacked_codebooks)
    assert codes.dtype == torch.long
    assert codes.min().item() >= 0
    assert codes.max().item() < arch.num_all_tokens_per_codebook


@pytest.mark.unit
def test_generate_codes_respects_forbidden_mask():
    """With argmax sampling, forbidden special tokens are never emitted (only EOS stays reachable)."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    cp.temperature = 0.0  # argmax over masked logits

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(7, arch.hidden_dim))

    # Allowed = real codebook tokens [0, codebook_size) plus the audio EOS id.
    allowed = (codes < arch.codebook_size) | (codes == arch.audio_eos_id)
    assert allowed.all(), f"sampled forbidden tokens: {sorted(set(codes[~allowed].tolist()))}"


@pytest.mark.unit
def test_generate_codes_deterministic_with_seed():
    """Same seed + same input ⇒ identical sampled codes (sampler is RNG-driven, no host state)."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    dec_hidden = torch.randn(4, arch.hidden_dim)

    torch.manual_seed(7)
    first = cp.generate_codes(dec_hidden)
    torch.manual_seed(7)
    second = cp.generate_codes(dec_hidden)

    assert torch.equal(first, second)


@pytest.mark.unit
@pytest.mark.parametrize("outer_mode", [None, CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
@pytest.mark.parametrize("raise_from_loop", [False, True], ids=["success", "failure"])
def test_generate_codes_uses_full_cudagraph_mode_and_restores_outer_mode(monkeypatch, outer_mode, raise_from_loop):
    """Check context selection/restoration, not numerical equivalence or CUDA replay."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    outer_context = SimpleNamespace(cudagraph_runtime_mode=outer_mode)
    observed_modes = []

    def get_context():
        assert outer_mode is not None
        return outer_context

    monkeypatch.setattr(local_transformer_module, "get_forward_context", get_context, raising=False)
    monkeypatch.setattr(
        local_transformer_module, "is_forward_context_available", lambda: outer_mode is not None, raising=False
    )

    def fake_code_loop(dec_hidden, _noise, _temperature):
        observed_modes.append(outer_context.cudagraph_runtime_mode)
        if raise_from_loop:
            raise RuntimeError("code loop failed")
        return torch.zeros(dec_hidden.shape[0], arch.num_stacked_codebooks, dtype=torch.long)

    monkeypatch.setattr(cp._code_loop, "forward", fake_code_loop)
    dec_hidden = torch.randn(3, arch.hidden_dim)

    if raise_from_loop:
        with pytest.raises(RuntimeError, match="code loop failed"):
            cp.generate_codes(dec_hidden)
    else:
        cp.generate_codes(dec_hidden)

    expected = CUDAGraphMode.FULL if outer_mode == CUDAGraphMode.PIECEWISE else outer_mode
    assert observed_modes == [expected]
    assert outer_context.cudagraph_runtime_mode == outer_mode


def _full_buffer_codes(cp, dec_hidden, gumbel_noise, temperature):
    """Reference the original loop that transforms all codebook positions."""
    num_tokens = dec_hidden.shape[0]
    num_codebooks = cp.num_codebooks
    buf = dec_hidden.new_zeros(num_tokens, num_codebooks, cp.lt_hidden)
    buf[:, 0, :] = cp.local_transformer_in_projection(dec_hidden)

    codes = []
    for k in range(num_codebooks):
        hidden = cp.local_transformer(buf)
        row = cp.local_transformer_audio_out_projection(hidden[:, k, :])
        logits = cp.local_transformer_out_projections[k](row)
        logits = logits.masked_fill(cp.forbidden_mask, float("-inf")) / temperature
        vals, idxs = torch.topk(logits, cp._sample_top_k, dim=-1)
        picked = (vals + gumbel_noise[:, k, :]).argmax(dim=-1, keepdim=True)
        code = idxs.gather(-1, picked).squeeze(-1)
        codes.append(code)
        if k + 1 < num_codebooks:
            emb = cp.audio_in_projection(cp.audio_embeddings[k](code))
            buf[:, k + 1, :] = cp.local_transformer_in_projection(emb)
    return torch.stack(codes, dim=1)


@pytest.mark.unit
@pytest.mark.parametrize("profile", ARCH_PROFILES.values(), ids=ARCH_PROFILES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16], ids=["fp32", "bf16", "fp16"])
def test_code_loop_matches_full_buffer_reference(profile, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(9101)
    cp, arch = _build_predictor(profile)
    cp.to(device=device, dtype=dtype)
    dec_hidden = torch.randn(3, arch.hidden_dim, device=device, dtype=dtype)
    noise = torch.rand(3, arch.num_stacked_codebooks, cp._sample_top_k, device=device).clamp_(1e-20, 1 - 1e-7)
    noise.log_().neg_().log_().neg_()
    temperature = torch.tensor([0.7], device=device)
    logits = []
    handles = [
        layer.register_forward_hook(lambda _m, _args, out: logits.append(out.detach().clone()))
        for layer in cp.local_transformer_out_projections
    ]
    try:
        expected = _full_buffer_codes(cp, dec_hidden, noise, temperature)
        reference_logits = list(logits)
        logits.clear()
        actual = cp._code_loop(dec_hidden, noise, temperature)
    finally:
        for handle in handles:
            handle.remove()
    assert len(logits) == len(reference_logits) == arch.num_stacked_codebooks
    for got, wanted in zip(logits, reference_logits, strict=True):
        torch.testing.assert_close(got, wanted)
    assert torch.equal(actual, expected)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 32, 128])
def test_code_loop_cuda_graph_replay_matches_full_buffer(batch_size):
    """Exercise real dense-loop capture/replay, independently of vLLM graph dispatch."""
    torch.manual_seed(9102)
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    cp.to(device="cuda", dtype=torch.float16)
    hidden = torch.randn(batch_size, arch.hidden_dim, device="cuda", dtype=torch.float16)
    noise = torch.empty(batch_size, arch.num_stacked_codebooks, cp._sample_top_k, device="cuda")
    noise.exponential_().log_().neg_()
    temperature = torch.tensor([0.7], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            cp._code_loop(hidden, noise, temperature)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        actual = cp._code_loop(hidden, noise, temperature)
    for _ in range(3):
        hidden.normal_()
        noise.exponential_().log_().neg_()
        graph.replay()
        with torch.no_grad():
            expected = _full_buffer_codes(cp, hidden, noise, temperature)
        assert torch.equal(actual, expected)
