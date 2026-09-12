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
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.model import EasyMagpieCodecForConditionalGeneration
from easymagpie_vllm_omni.codec.packed import CODEC_STATE_ELEMENTS, CodecStateLayer, PackedEasyMagpieCodec
from easymagpie_vllm_omni.runner import EasyMagpieCodecGPUGenerationWorker
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker


def _model(device="cpu", extra=None, *, max_num_seqs=4, max_num_batched_tokens=64):
    model = EasyMagpieCodecForConditionalGeneration.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = EasyMagpieCodecConfig(
        input_dim=4,
        input_filters=8,
        hidden_filters=16,
        num_hidden_layers=2,
        pre_upsample_rates=[2],
        pre_upsample_filters=[8],
        resblock_upsample_rates=[2],
        resblock_upsample_filters=[4],
        num_codebooks=2,
        codebook_size=4,
        num_levels_per_group=[2, 2],
        frame_stacking_factor=2,
    )
    model.vllm_config = VllmConfig(device_config=DeviceConfig(device=device))
    model.vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    model.vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    with set_current_vllm_config(model.vllm_config):
        model.codec = PackedEasyMagpieCodec(model.config, dtype=torch.float32).to(device).eval()
    model.vllm_config.model_config = SimpleNamespace(stage_connector_config={"extra": extra or {}})
    layers = [layer for layer in model.codec.modules() if isinstance(layer, CodecStateLayer)]
    for layer in layers:
        state = torch.empty_strided((6, CODEC_STATE_ELEMENTS), (CODEC_STATE_ELEMENTS + 16, 1), device=device)
        layer.kv_cache = [state.fill_(0.375)]
    return model, layers


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_codec_warmup_uses_real_codec_and_restores_owned_state(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model, layers = _model(device, {"codec_chunk_frames": 8, "codec_startup_chunk_frames": [2, 2, 4]})
    original = [layer.kv_cache for layer in layers]
    state_values = [cache[0].clone() for cache in original]
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device == "cuda" else None
    forward = model.codec.forward
    calls = []
    temporary = []

    def record(codes):
        metadata = get_forward_context().attn_metadata[layers[0].prefix]
        for layer, cache in zip(layers, original, strict=True):
            assert layer.kv_cache is not cache
            assert layer.kv_cache[0].data_ptr() != cache[0].data_ptr()
            assert layer.kv_cache[0].stride() == cache[0].stride()
            assert layer.kv_cache[0].shape == (4, cache[0].shape[1])
            assert layer.kv_cache[0].dtype == cache[0].dtype
            assert layer.kv_cache[0].device == cache[0].device
        current = [layer.kv_cache[0].data_ptr() for layer in layers]
        if temporary:
            assert current == temporary
        else:
            temporary.extend(current)
        lengths = torch.diff(metadata.query_start_loc_p).tolist() if metadata.num_prefills else []
        assert codes.shape == (sum(lengths) + metadata.num_decodes, model.config.num_stacked_codebooks)
        if metadata.num_prefills:
            assert metadata.state_indices_tensor_p.data_ptr() % 16 == metadata.num_decodes * 4 % 16
        calls.append((lengths, metadata.num_decodes, metadata.codec_prefill_uniform))
        result = forward(codes)
        assert result.shape == (codes.shape[0] * model.config.samples_per_frame,)
        assert torch.isfinite(result).all()
        return result

    monkeypatch.setattr(model.codec, "forward", record)
    model.warmup_codec()

    assert calls == [
        ([2, 2], 0, True),
        ([2, 2], 1, True),
        ([4, 4], 0, True),
        ([4, 4], 1, True),
        ([8, 8], 0, True),
        ([8, 8], 1, True),
        ([2, 4, 8], 0, False),
        ([2, 4, 8], 0, False),
        ([2, 4, 8], 1, False),
        ([], 4, False),
    ] + [([2] * batch, 0, True) for batch in (1, 3, 4)]
    for layer, cache, values in zip(layers, original, state_values, strict=True):
        assert layer.kv_cache is cache
        assert torch.equal(cache[0], values)
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_codec_warmup_restores_state_after_forward_failure(startup_failure, monkeypatch):
    model, layers = _model(extra={"codec_chunk_frames": 8, "codec_startup_chunk_frames": [2]})
    original = [layer.kv_cache for layer in layers]
    values = [cache[0].clone() for cache in original]

    def fail(codes):
        metadata = get_forward_context().attn_metadata[layers[0].prefix]
        if startup_failure and metadata.num_prefills != 4:
            return codes.new_empty(0)
        layers[0].kv_cache[0].fill_(9)
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(model.codec, "forward", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        model.warmup_codec()
    for layer, cache, value in zip(layers, original, values, strict=True):
        assert layer.kv_cache is cache
        assert torch.equal(cache[0], value)


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, [25]),
        ({"codec_chunk_frames": 8, "codec_startup_chunk_frames": [1, 2]}, [1, 2, 8]),
        (
            {"codec_chunk_frames": 6, "codec_startup_chunk_frames": [2, 2], "codec_busy_startup_chunk_frames": [3]},
            [2, 3, 6],
        ),
    ],
)
def test_codec_warmup_uses_configured_uniform_sizes(extra, expected, monkeypatch):
    model, layers = _model(extra=extra)
    uniform_sizes = []

    def record(codes):
        metadata = get_forward_context().attn_metadata[layers[0].prefix]
        if metadata.codec_prefill_uniform:
            uniform_sizes.append(metadata.codec_max_query_len)
        return codes.new_empty(0)

    monkeypatch.setattr(model.codec, "forward", record)
    model.warmup_codec()
    assert uniform_sizes[: 2 * len(expected)] == [size for size in expected for _ in (0, 1)]
    assert set(uniform_sizes) == set(expected)


@pytest.mark.parametrize(
    "extra,max_seqs,max_tokens,expected",
    [
        ({"codec_startup_chunk_frames": [2, 4]}, 5, 64, [(2, 1), (2, 3), (2, 4), (2, 5)]),
        ({"codec_startup_chunk_frames": [2, 4]}, 8, 7, [(2, 1), (2, 3)]),
        ({"codec_startup_chunk_frames": [2]}, 1, 64, [(2, 1)]),
        ({"codec_startup_chunk_frames": [8]}, 8, 7, []),
        (
            {"codec_startup_chunk_frames": [2, 4], "codec_busy_startup_chunk_frames": [3, 6]},
            8,
            7,
            [(2, 1), (2, 3), (3, 1)],
        ),
        ({"codec_startup_chunk_frames": [2], "codec_busy_startup_chunk_frames": [2, 4]}, 3, 64, [(2, 1), (2, 3)]),
        ({"codec_busy_startup_chunk_frames": [3, 6]}, 3, 64, [(3, 1), (3, 3)]),
        ({"codec_startup_chunk_frames": [], "codec_busy_startup_chunk_frames": [3]}, 3, 64, [(3, 1), (3, 3)]),
        ({}, 128, 1536, []),
        ({"codec_startup_chunk_frames": [], "codec_busy_startup_chunk_frames": []}, 128, 1536, []),
        (
            {"codec_chunk_frames": 8, "codec_startup_chunk_frames": [2, 2, 2, 4]},
            128,
            1536,
            [(2, batch) for batch in range(1, 129) if batch != 2],
        ),
    ],
)
def test_codec_warmup_bounds_startup_batch_sweep(extra, max_seqs, max_tokens, expected, monkeypatch):
    model, layers = _model(extra=extra, max_num_seqs=max_seqs, max_num_batched_tokens=max_tokens)
    sizes = {int(extra.get("codec_chunk_frames", 25))}
    for key in ("codec_startup_chunk_frames", "codec_busy_startup_chunk_frames"):
        sizes.update(extra.get(key, []))
    legacy_calls = 2 * len(sizes) + 4
    calls = []
    pages = max(4, max((batch for _, batch in expected), default=0))

    def record(codes):
        metadata = get_forward_context().attn_metadata[layers[0].prefix]
        assert all(layer.kv_cache[0].shape[0] == pages for layer in layers)
        if len(calls) >= legacy_calls:
            assert metadata.num_decodes == 0
            assert metadata.codec_uniform and metadata.codec_prefill_uniform
            assert not metadata.has_initial_states_p.any()
            assert metadata.num_prefills <= max_seqs
            assert codes.shape[0] <= max_tokens
        calls.append((metadata.codec_max_query_len, metadata.num_prefills))
        return codes.new_empty(0)

    monkeypatch.setattr(model.codec, "forward", record)
    model.warmup_codec()
    assert calls[legacy_calls:] == expected


def test_codec_warmup_restores_state_after_partial_allocation_failure(monkeypatch):
    model, layers = _model(extra={"codec_startup_chunk_frames": [2]})
    original = [layer.kv_cache for layer in layers]
    allocate = torch.empty_strided
    allocations = 0

    def fail(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        if allocations == 3:
            raise RuntimeError("allocation failed")
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_strided", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        model.warmup_codec()
    assert all(layer.kv_cache is cache for layer, cache in zip(layers, original, strict=True))
    assert all(torch.all(cache[0] == 0.375) for cache in original)


@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.parametrize(
    "key", ["codec_chunk_frames", "codec_startup_chunk_frames", "codec_busy_startup_chunk_frames"]
)
def test_codec_warmup_rejects_invalid_chunk_sizes(size, key):
    model, _ = _model(extra={key: size if key == "codec_chunk_frames" else [2, size]})
    with pytest.raises(ValueError, match="positive"):
        model.warmup_codec()


@pytest.mark.parametrize("connector", ["absent", None, {"extra": None}])
def test_codec_warmup_allows_missing_connector_config(connector, monkeypatch):
    model, layers = _model()
    if connector == "absent":
        del model.vllm_config.model_config.stage_connector_config
    else:
        model.vllm_config.model_config.stage_connector_config = connector
    sizes = []

    def record(codes):
        metadata = get_forward_context().attn_metadata[layers[0].prefix]
        if metadata.codec_prefill_uniform:
            sizes.append(metadata.codec_max_query_len)
        return codes.new_empty(0)

    monkeypatch.setattr(model.codec, "forward", record)
    model.warmup_codec()
    assert sizes == [25, 25]


def test_codec_worker_warms_before_upstream_readiness(monkeypatch):
    calls = []
    result = object()
    model = SimpleNamespace(warmup_codec=lambda: calls.append("codec"))
    worker = EasyMagpieCodecGPUGenerationWorker.__new__(EasyMagpieCodecGPUGenerationWorker)
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    def upstream(self):
        calls.append("upstream")
        return result

    monkeypatch.setattr(GPUGenerationWorker, "compile_or_warm_up_model", upstream)
    assert worker.compile_or_warm_up_model() is result
    assert calls == ["codec", "upstream"]


def test_codec_worker_does_not_report_readiness_after_warmup_failure(monkeypatch):
    def fail():
        raise RuntimeError("codec warmup failed")

    worker = EasyMagpieCodecGPUGenerationWorker.__new__(EasyMagpieCodecGPUGenerationWorker)
    worker.model_runner = SimpleNamespace(get_model=lambda: SimpleNamespace(warmup_codec=fail))
    monkeypatch.setattr(GPUGenerationWorker, "compile_or_warm_up_model", lambda self: pytest.fail("reached readiness"))
    with pytest.raises(RuntimeError, match="codec warmup failed"):
        worker.compile_or_warm_up_model()
