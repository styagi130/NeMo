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
"""Parity with the pinned Mamba2 builder without reducing its device mask."""

import inspect
from copy import deepcopy
from dataclasses import fields
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni import backbone_patches
from vllm.config import CUDAGraphMode
from vllm.utils import torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import mamba_attn
from vllm.v1.attention.backends import utils as attention_utils
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

_PATCH = "patch_mamba_prefill_initial_states"


@pytest.fixture(autouse=True)
def cpu_allocations_without_driver(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
        monkeypatch.setattr(attention_utils, "PIN_MEMORY", False)


@pytest.fixture
def original_build(monkeypatch):
    original = inspect.unwrap(Mamba2AttentionMetadataBuilder.build)
    monkeypatch.setattr(Mamba2AttentionMetadataBuilder, "build", original)
    # Keep the existing EasyMagpie streaming classifier installed and restore it.
    monkeypatch.setattr(mamba_attn, "split_decodes_and_prefills", mamba_attn.split_decodes_and_prefills)
    backbone_patches.patch_mamba_streaming_decode()
    getattr(backbone_patches, _PATCH, lambda: None)()
    return original


@pytest.mark.parametrize("computed", [[0, 0], [0, 8], [17, 0]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_flag_does_not_reduce_device_mask(computed, device, original_build, monkeypatch):
    _require_device(device)
    builder = _builder(device)
    metadata = _metadata([3, 7], computed, device)
    compute = builder._compute_common_metadata
    device_masks = []

    def common(*args, **kwargs):
        result = compute(*args, **kwargs)
        device_masks.append(result.has_initial_states_p)
        return result

    original_any = torch.any

    def no_device_mask(tensor, *args, **kwargs):
        assert all(tensor is not mask for mask in device_masks), "device-mask reduction synchronizes the worker"
        return original_any(tensor, *args, **kwargs)

    monkeypatch.setattr(builder, "_compute_common_metadata", common)
    monkeypatch.setattr(torch, "any", no_device_mask)
    actual = builder.build(0, metadata)
    assert actual.prep_initial_states is any(computed)
    assert actual.has_initial_states_p is device_masks[0]
    assert len(device_masks) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_mode", ["none", "align", "all"])
@pytest.mark.parametrize(
    "lengths,computed",
    [
        ([3, 7], [0, 0]),
        ([3, 7], [0, 17]),
        ([1, 3, 7], [9, 0, 8]),
        ([1, 1, 3], [9, 8, 0]),
        ([1, 1], [9, 8]),
        ([1, 0, 0], [9, 0, 0]),
    ],
)
def test_real_builder_metadata_parity(device, cache_mode, lengths, computed, original_build, monkeypatch):
    _require_device(device)
    builder = _builder(device, cache_mode, graphs=True)
    metadata = _metadata(lengths, computed, device)
    expected = deepcopy(original_build(builder, 0, metadata, fast_build=True))
    common_values = []
    compute = builder._compute_common_metadata

    def common(*args, **kwargs):
        result = compute(*args, **kwargs)
        common_values.append(result)
        return result

    monkeypatch.setattr(builder, "_compute_common_metadata", common)
    actual = builder.build(0, metadata, fast_build=True)
    _equal_metadata(actual, expected)
    assert len(common_values) == 1
    replaced = {"prep_initial_states", "chunk_size", "seq_idx_p", "cu_chunk_seqlen_p", "last_chunk_indices_p"}
    for field in fields(actual):
        if field.name not in replaced:
            assert getattr(actual, field.name) is getattr(common_values[0], field.name)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_mode", ["none", "align", "all"])
@pytest.mark.parametrize("prefill_computed", [0, 9])
def test_async_decode_upper_bound_cannot_set_prefill_flag(device, cache_mode, prefill_computed, original_build):
    _require_device(device)
    builder = _builder(device, cache_mode, spec_tokens=2)
    metadata = _metadata([3, 7], [12, prefill_computed], device, prefilling=[False, True])
    metadata.seq_lens_cpu_upper_bound[0] += 2
    kwargs = {
        "num_accepted_tokens": torch.tensor([1, 1], dtype=torch.int32, device=device),
        "prev_last_scheduled_idx": torch.tensor([1, -1], dtype=torch.int32, device=device),
    }
    expected = deepcopy(original_build(builder, 0, metadata, **kwargs))
    actual = builder.build(0, metadata, **kwargs)
    _equal_metadata(actual, expected)
    assert actual.num_decodes == actual.num_prefills == 1
    assert actual.prep_initial_states is (prefill_computed > 0)


def test_reordered_reused_slots_do_not_retain_prefill_flag(original_build):
    builder = _builder(cache_mode="all")
    for lengths, computed in [([1, 3, 7], [8, 0, 17]), ([3, 7], [0, 0]), ([1, 7], [8, 9])]:
        metadata = _metadata(lengths, computed)
        metadata.block_table_tensor = metadata.block_table_tensor.flip(0)
        expected = deepcopy(original_build(builder, 0, metadata))
        actual = builder.build(0, metadata)
        _equal_metadata(actual, expected)


@pytest.mark.parametrize("architecture", ["EasyMagpieTTSForConditionalGeneration", "EasyMagpieTTS"])
def test_supported_architecture_and_idempotent_install(architecture, original_build):
    builder = _builder()
    builder.vllm_config.model_config.hf_config.architectures = [architecture]
    installed = Mamba2AttentionMetadataBuilder.build
    getattr(backbone_patches, _PATCH)()
    assert Mamba2AttentionMetadataBuilder.build is installed
    metadata = _metadata([3], [0])
    assert builder.build(0, metadata).prep_initial_states is False


@pytest.mark.parametrize(
    "variant",
    [
        "other-model",
        "missing-architecture",
        "string-architecture",
        "missing-seq",
        "missing-query",
        "absent-query",
        "short-seq",
        "short-query",
        "matrix-query",
        "device-seq",
        "device-query",
    ],
)
def test_unsupported_inputs_delegate_once_with_original_arguments(variant, monkeypatch):
    builder = _builder()
    metadata = _metadata([3], [0])
    config = builder.vllm_config.model_config.hf_config
    if variant == "other-model":
        config.architectures = ["NemotronHForCausalLM"]
    elif variant == "missing-architecture":
        del config.architectures
    elif variant == "string-architecture":
        config.architectures = "EasyMagpieTTS"
    elif variant == "missing-seq":
        metadata.seq_lens_cpu_upper_bound = None
    elif variant == "missing-query":
        metadata.query_start_loc_cpu = None
    elif variant == "absent-query":
        del metadata.query_start_loc_cpu
    elif variant == "short-seq":
        metadata.seq_lens_cpu_upper_bound = torch.empty(0, dtype=torch.int32)
    elif variant == "short-query":
        metadata.query_start_loc_cpu = torch.tensor([0], dtype=torch.int32)
    elif variant == "matrix-query":
        metadata.query_start_loc_cpu = metadata.query_start_loc_cpu.unsqueeze(0)
    elif variant == "device-seq":
        metadata.seq_lens_cpu_upper_bound = torch.empty(1, device="meta", dtype=torch.int32)
    elif variant == "device-query":
        metadata.query_start_loc_cpu = torch.empty(2, device="meta", dtype=torch.int32)
    calls = []
    result = object()

    def original(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return result

    monkeypatch.setattr(Mamba2AttentionMetadataBuilder, "build", original)
    getattr(backbone_patches, _PATCH, lambda: None)()
    kwargs = {"fast_build": True, "num_accepted_tokens": object(), "extra": object()}
    assert builder.build(13, metadata, **kwargs) is result
    assert calls == [(builder, (13, metadata), kwargs)]


@pytest.mark.parametrize("failure", ["common", "chunks"])
def test_helper_exceptions_are_not_retried(failure, original_build, monkeypatch):
    builder = _builder()
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("expected metadata failure")

    helper = "_compute_common_metadata" if failure == "common" else "_build_chunk_metadata_tensors"
    monkeypatch.setattr(builder, helper, fail)
    with pytest.raises(RuntimeError, match="expected metadata failure"):
        builder.build(0, _metadata([3], [0]))
    assert len(calls) == 1


def test_none_mask_keeps_false_and_preserves_chunk_helper(original_build, monkeypatch):
    builder = _builder()
    compute = builder._compute_common_metadata

    def common(*args, **kwargs):
        result = compute(*args, **kwargs)
        result.has_initial_states_p = None
        return result

    monkeypatch.setattr(builder, "_compute_common_metadata", common)
    actual = builder.build(0, _metadata([3], [8]))
    assert actual.prep_initial_states is False
    assert actual.has_initial_states_p is None
    assert actual.cu_chunk_seqlen_p is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cache_mode", ["none", "align", "all"])
def test_full_capture_builder_path_is_unchanged(device, cache_mode, original_build):
    _require_device(device)
    builder = _builder(device, cache_mode, graphs=True)
    metadata = _metadata([1, 1, 0], [7, 12, 0], device)
    expected = deepcopy(original_build(builder, 0, metadata))
    actual = builder.build_for_cudagraph_capture(metadata)
    _equal_metadata(actual, expected)
    assert actual.num_prefills == 0 and actual.prep_initial_states is False


def _require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _builder(device="cpu", cache_mode="none", graphs=False, spec_tokens=0):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="nemotron_h", architectures=["EasyMagpieTTSForConditionalGeneration"]
            ),
            max_model_len=128,
            get_mamba_chunk_size=lambda: 8,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(mamba_cache_mode=cache_mode),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=8,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if graphs else CUDAGraphMode.NONE,
        ),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_tokens, parallel_drafting=False) if spec_tokens else None
        ),
        num_speculative_tokens=spec_tokens,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    spec = MambaSpec(
        block_size=8,
        shapes=((2, 2),),
        dtypes=(torch.float32,),
        mamba_cache_mode=cache_mode,
        num_speculative_blocks=spec_tokens,
    )
    return Mamba2AttentionMetadataBuilder(spec, ["backbone.layers.0.mixer"], config, torch.device(device))


def _metadata(lengths, computed, device="cpu", prefilling=None):
    starts = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    seq_lens = torch.tensor(lengths, dtype=torch.int32) + torch.tensor(computed, dtype=torch.int32)
    count = len(lengths)
    return CommonAttentionMetadata(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=seq_lens.to(device),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_reqs=count,
        num_actual_tokens=sum(lengths),
        max_query_len=max(lengths),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.arange(count * 16, dtype=torch.int32, device=device).reshape(count, 16),
        slot_mapping=torch.arange(sum(lengths), device=device),
        is_prefilling=torch.tensor(prefilling if prefilling is not None else [n > 1 for n in lengths]),
    )


def _equal_metadata(actual, expected):
    def equal(value, reference):
        if isinstance(value, torch.Tensor):
            assert value.dtype == reference.dtype and value.device == reference.device
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        elif isinstance(value, dict):
            assert value.keys() == reference.keys()
            for key in value:
                equal(value[key], reference[key])
        else:
            assert value == reference

    assert type(actual) is type(expected)
    for field in fields(actual):
        equal(getattr(actual, field.name), getattr(expected, field.name))
