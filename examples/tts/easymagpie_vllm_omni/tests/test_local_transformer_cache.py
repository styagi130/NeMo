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
"""Correctness tests for the frame-local NeMo-style LT cache."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from conftest import build_vllm_config  # noqa: E402
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor  # noqa: E402


@torch.no_grad()
def _teacher_forced_step_logits(
    predictor: EasyMagpieCodePredictor,
    dec_hidden: torch.Tensor,
    codes: torch.Tensor,
    *,
    use_cache: bool,
) -> tuple[torch.Tensor, list]:
    batch = dec_hidden.shape[0]
    n = predictor.num_codebooks
    buf = dec_hidden.new_zeros(batch, n, predictor.lt_hidden)
    buf[:, 0, :] = predictor.local_transformer_in_projection(dec_hidden)
    caches = [(None, None, None) for _ in predictor.local_transformer.layers]
    logits = []

    for index in range(n):
        if use_cache:
            hidden, caches = predictor.local_transformer.forward_nemo_cached(buf[:, : index + 1, :], caches)
            row = hidden[:, -1, :]
        else:
            hidden = predictor.local_transformer(buf)
            row = hidden[:, index, :]
        row = predictor.local_transformer_audio_out_projection(row)
        logits.append(predictor.local_transformer_out_projections[index](row))

        if index + 1 < n:
            embedding = predictor.audio_in_projection(predictor.audio_embeddings[index](codes[:, index]))
            buf[:, index + 1, :] = predictor.local_transformer_in_projection(embedding)

    return torch.stack(logits, dim=1), caches


@pytest.mark.unit
@pytest.mark.parametrize("batch", [1, 5])
def test_nemo_style_cache_matches_uncached_local_transformer(batch):
    torch.manual_seed(1234)
    config = build_vllm_config()
    predictor = EasyMagpieCodePredictor(vllm_config=config, prefix="code_predictor").eval()
    dec_hidden = torch.randn(batch, predictor.embedding_dim)
    codes = torch.randint(0, predictor.arch.codebook_size, (batch, predictor.num_codebooks))

    expected, _ = _teacher_forced_step_logits(predictor, dec_hidden, codes, use_cache=False)
    actual, caches = _teacher_forced_step_logits(predictor, dec_hidden, codes, use_cache=True)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for key, value, attention in caches:
        assert key.shape[2] == predictor.num_codebooks
        assert value.shape[2] == predictor.num_codebooks
        assert attention.shape[1] == predictor.num_codebooks


@pytest.mark.unit
def test_generate_codes_with_nemo_style_cache():
    torch.manual_seed(7)
    predictor = EasyMagpieCodePredictor(vllm_config=build_vllm_config(), prefix="code_predictor").eval()
    predictor.init_forbidden_mask()
    predictor.local_transformer_use_kv_cache = True

    codes = predictor.generate_codes(torch.randn(4, predictor.embedding_dim))

    assert codes.shape == (4, predictor.num_codebooks)
    assert codes.dtype == torch.long
    allowed = (codes < predictor.arch.codebook_size) | (codes == predictor.arch.audio_eos_id)
    assert allowed.all()
