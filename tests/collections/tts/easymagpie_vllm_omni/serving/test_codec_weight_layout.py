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

from types import SimpleNamespace

import pytest
import torch

from easymagpie_vllm_omni.codec import kernels
from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS


@pytest.mark.parametrize("kernel_size", [1, 3, 7])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_conv_weight_layout_is_materialized_per_call(monkeypatch, kernel_size, noncontiguous) -> None:
    captured = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: captured.append((args, args[1].clone()))

    monkeypatch.setattr(kernels, "_packed_causal_conv1d_kernel", Kernel())
    monkeypatch.setattr(kernels, "update_packed_state", lambda *args, **kwargs: None)
    inputs = SimpleNamespace(is_cuda=True, shape=(3, 35), dtype=torch.float32, device="cpu")
    inputs.contiguous = lambda: inputs
    starts, pages, flags = torch.tensor([0, 1, 3]), torch.tensor([1, 0]), torch.tensor([False, True])
    state = torch.zeros(2, CODEC_STATE_ELEMENTS)

    with torch.inference_mode():
        weight = torch.arange(65 * 35 * kernel_size, dtype=torch.float32).reshape(65, 35, kernel_size)
        if noncontiguous:
            weight = weight.transpose(0, 1).contiguous().transpose(0, 1)
        snapshots = []
        for increment in (0, 100):
            weight.data.copy_(weight + increment)
            snapshots.append(weight.clone())
            output = kernels.packed_causal_conv1d(
                inputs, weight, None, state, starts, pages, flags, time_factor=1, max_query_len=2
            )
            assert output.shape == (3, 65)
            assert torch.equal(weight, snapshots[-1])

    for (arguments, values), original in zip(captured, snapshots):
        converted = arguments[1]
        assert arguments[9:13] == (35, 65, kernel_size, 1)
        assert converted.shape == (kernel_size, 35, 65) and converted.is_contiguous()
        # An already-contiguous KIO view may share storage; inspect values at launch.
        assert torch.equal(values.permute(2, 1, 0), original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_conv_weight_updates_on_current_stream(tmp_path) -> None:
    torch.manual_seed(31)
    conv = torch.nn.Conv1d(35, 65, 3).cuda()
    checkpoint = tmp_path / "conv.pt"
    torch.save(conv.state_dict(), checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    inputs = torch.randn(19, 35, device="cuda")
    starts = torch.tensor([0, 2, 19], dtype=torch.int32, device="cuda")
    pages = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    flags = torch.tensor([False, True], device="cuda")
    initial_state = torch.randn(4, CODEC_STATE_ELEMENTS, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    results = []

    with torch.inference_mode(), torch.cuda.stream(stream):
        for update in ("original", "copy", "replace", "reload"):
            if update == "copy":
                conv.weight.data.copy_(conv.weight + 0.25)
            elif update == "replace":
                conv.weight = torch.nn.Parameter(conv.weight.flip(0))
            elif update == "reload":
                conv.load_state_dict(restored)
            state = initial_state.clone()
            output = kernels.packed_causal_conv1d(
                inputs, conv.weight, conv.bias, state, starts, pages, flags, time_factor=1, max_query_len=17
            )
            results.append((output, state, conv.weight.clone(), conv.bias.clone()))
    stream.synchronize()

    # Reference calls use separate weight objects after the producer stream completes.
    for output, state, weight, bias in results:
        reference_state = initial_state.clone()
        reference = kernels.packed_causal_conv1d(
            inputs, weight, bias, reference_state, starts, pages, flags, time_factor=1, max_query_len=17
        )
        assert torch.equal(output, reference)
        assert torch.equal(state, reference_state)
    assert not torch.equal(results[0][0], results[1][0])
    assert not torch.equal(results[1][0], results[2][0])
    assert torch.equal(results[0][0], results[3][0])
