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

from math import prod

import pytest
import torch
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packed import PackedFiniteScalarDequantizer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("levels", [[2, 2], [4, 4, 4, 4, 4], [3, 5, 7]])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fsq_full_codebook_matches_arithmetic(device, levels, index_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU lookup equivalence")
    config = EasyMagpieCodecConfig(
        input_dim=2 * len(levels), num_codebooks=2, codebook_size=prod(levels), num_levels_per_group=levels
    )
    dequantizer = PackedFiniteScalarDequantizer(config).to(device)
    codes = torch.arange(config.codebook_size, dtype=index_dtype, device=device)
    indices = torch.stack((codes, codes.roll(1)), dim=-1)
    level_tensor = torch.tensor(levels, device=device)
    bases = torch.tensor([1, *levels[:-1]], device=device).cumprod(0)
    scale = torch.div(level_tensor, 2, rounding_mode="floor")
    nonnegative = torch.div(indices.unsqueeze(-1), bases, rounding_mode="floor") % level_tensor
    expected = ((nonnegative - scale) / scale).flatten(start_dim=1)

    torch.testing.assert_close(dequantizer(indices), expected, atol=0, rtol=0)
    assert not dequantizer.state_dict()
