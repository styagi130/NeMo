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

import pytest
import torch
import torch.nn.functional as F

from easymagpie_vllm_omni.codec.kernels import packed_causal_conv1d, packed_causal_conv_transpose1d, packed_half_snake
from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")


def test_packed_half_snake() -> None:
    torch.manual_seed(17)
    inputs = torch.randn(113, 32, device="cuda")
    alpha = torch.rand(1, 16, 1, device="cuda") + 0.25
    actual = packed_half_snake(inputs, alpha)
    snake_in = inputs[:, :16]
    scale = alpha.reshape(1, -1)
    expected = torch.cat(
        (snake_in + torch.sin(scale * snake_in).square() / (scale + 1e-9), F.leaky_relu(inputs[:, 16:])),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(1, 32), (113, 32), (257, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_residual", [False, True])
def test_packed_half_snake_residual_matches_separate_add(shape, dtype, use_residual) -> None:
    torch.manual_seed(18)
    inputs = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn_like(inputs) if use_residual else None
    alpha = torch.rand(1, shape[1] // 2, 1, dtype=dtype, device="cuda") + 0.25
    combined = inputs if residual is None else inputs + residual
    expected = packed_half_snake(combined, alpha)

    actual = packed_half_snake(inputs, alpha, residual)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kernel_size", [1, 3, 7])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_packed_causal_conv1d(kernel_size, use_bias, noncontiguous) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    factor = 2
    lengths = [3, 2]
    channels, output_channels, history = 35, 65, kernel_size - 1
    sequences = [torch.randn(length * factor, channels, device=device) for length in lengths]
    packed = torch.cat(sequences)
    weight = torch.randn(output_channels, channels, kernel_size, device=device) * 0.1
    if noncontiguous:
        weight = weight.transpose(0, 1).contiguous().transpose(0, 1)
        assert not weight.is_contiguous()
    bias = torch.randn(output_channels, device=device) if use_bias else None
    state = torch.randn(3, CODEC_STATE_ELEMENTS, device=device)
    expected_state = state.clone()
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    has_initial = torch.zeros(2, dtype=torch.bool, device=device)

    actual = packed_causal_conv1d(
        packed,
        weight,
        bias,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        time_factor=factor,
    )
    # Triton uses IEEE dot products; compare against a float64 CPU oracle rather than CUDA TF32.
    expected = torch.cat(
        [
            F.conv1d(
                F.pad(sequence.T[None].double().cpu(), (history, 0)),
                weight.double().cpu(),
                None if bias is None else bias.double().cpu(),
            )[0].T
            for sequence in sequences
        ]
    ).to(device=device, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for index, sequence in enumerate(sequences):
        if history:
            expected_state[index, : history * channels] = F.pad(sequence.T, (history, 0)).T[-history:].reshape(-1)
    assert torch.equal(state, expected_state)


@pytest.mark.parametrize("factor", [1, 7])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("output_channels", [7, 33, 65])
def test_packed_causal_conv1d_ragged_time_tiles(factor, use_bias, output_channels) -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    channels, history = 5, 6
    weight = torch.randn(output_channels, channels, history + 1, device=device) * 0.1
    bias = torch.randn(output_channels, device=device) if use_bias else None
    state = torch.randn(5, CODEC_STATE_ELEMENTS, device=device)

    for lengths, pages, initial in [
        ([1, 65, 2, 17], [3, 0, 4, 1], [True, False, True, False]),
        ([17, 2, 65, 1], [1, 4, 0, 3], [True, False, True, True]),
    ]:
        sequences = [torch.randn(length * factor, channels, device=device) for length in lengths]
        packed = torch.cat(sequences)
        starts = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
        indices = torch.tensor(pages, dtype=torch.int32, device=device)
        flags = torch.tensor(initial, dtype=torch.bool, device=device)
        before = state.clone()
        reference_state = before.clone()
        expected = []
        for sequence, page, has_initial in zip(sequences, pages, initial):
            previous = before[page, : history * channels].view(history, channels)
            joined = torch.cat((previous if has_initial else torch.zeros_like(previous), sequence))
            expected.append(
                F.conv1d(
                    joined.T[None].double().cpu(),
                    weight.double().cpu(),
                    None if bias is None else bias.double().cpu(),
                )[0].T
            )
            reference_state[page, : history * channels] = joined[-history:].reshape(-1)

        actual = packed_causal_conv1d(
            packed, weight, bias, state, starts, indices, flags, time_factor=factor, max_query_len=129
        )
        # The larger grid adds wholly invalid tiles; partial tiles must keep their masks.
        compact_state = before.clone()
        compact = packed_causal_conv1d(
            packed, weight, bias, compact_state, starts, indices, flags, time_factor=factor, max_query_len=max(lengths)
        )
        assert torch.equal(actual, compact)
        assert torch.equal(state, compact_state)
        assert torch.equal(state, reference_state)
        torch.testing.assert_close(
            actual, torch.cat(expected).to(device=device, dtype=actual.dtype), atol=2e-5, rtol=2e-5
        )


def test_packed_causal_conv_transpose1d() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    factor = 2
    stride = 2
    lengths = [3, 2]
    sequences = [torch.randn(length * factor, 8, device=device) for length in lengths]
    packed = torch.cat(sequences)
    conv = torch.nn.ConvTranspose1d(8, 4, 2 * stride, stride=stride, groups=4).to(device)
    state = torch.zeros(2, CODEC_STATE_ELEMENTS, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    has_initial = torch.zeros(2, dtype=torch.bool, device=device)

    actual = packed_causal_conv_transpose1d(
        packed,
        conv.weight,
        conv.bias,
        state,
        query_start_loc,
        cache_indices,
        has_initial,
        stride=stride,
        time_factor=factor,
        output_channels=4,
    )
    expected = torch.cat([conv(sequence.T[None])[0, :, :-stride].T for sequence in sequences])
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for index, sequence in enumerate(sequences):
        torch.testing.assert_close(state[index, :8], sequence[-1])
