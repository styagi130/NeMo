# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import types
from unittest import mock

import pytest
import torch
from easymagpie_vllm_omni import watermark
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.watermark import AudioChunk, DisabledAudioWatermarker, TaperedPerthWatermarker


class _FakeAudioProcessor:
    def signal_to_magphase(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return signal, torch.zeros_like(signal)

    def magphase_to_signal(self, magnitude: torch.Tensor, _phase: torch.Tensor) -> torch.Tensor:
        return magnitude


class _FakeEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, magnitude: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.batch_sizes.append(int(magnitude.shape[0]))
        return magnitude + 0.125, None


class _FakePerthNet:
    def __init__(self) -> None:
        self.hp = types.SimpleNamespace(sample_rate=22_050, n_fft=8)
        self.device = torch.device("cpu")
        self.ap = _FakeAudioProcessor()
        self.encoder = _FakeEncoder()


def _chunks(*waveforms: torch.Tensor) -> list[AudioChunk]:
    return [
        AudioChunk(request_id=f"req-{index}", is_final=False, audio=waveform)
        for index, waveform in enumerate(waveforms)
    ]


def test_unindexed_cuda_device_is_normalized() -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=2),
    ):
        assert watermark.normalized_device("cuda") == "cuda:2"


def test_equal_length_waveforms_are_batched_and_length_is_preserved() -> None:
    perth_net = _FakePerthNet()
    watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
    result = watermarker.apply(_chunks(torch.zeros(16), torch.full((16,), 0.5), torch.zeros(12)))

    assert perth_net.encoder.batch_sizes == [2, 1]
    assert [tensor.numel() for tensor in result] == [16, 16, 12]
    ramp = torch.sin(torch.linspace(0, torch.pi / 2, 4)).square()
    envelope = torch.ones(16)
    envelope[:4] = ramp
    envelope[-4:] = ramp.flip(0)
    torch.testing.assert_close(result[0], 0.125 * envelope)
    torch.testing.assert_close(result[1], 0.5 + 0.125 * envelope)


def test_perth_resamplers_are_reused_with_wideband_float32_kernels() -> None:
    perth_net = _FakePerthNet()
    perth_net.hp.sample_rate = 32_000
    with mock.patch(
        "torchaudio.transforms.Resample", side_effect=lambda *_args, **_kwargs: torch.nn.Identity()
    ) as fake:
        watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
        for _ in range(2):
            watermarker.apply(_chunks(torch.zeros(16)))

    assert fake.call_count == 2
    assert [call.args for call in fake.call_args_list] == [(22_050, 32_000), (32_000, 22_050)]
    for call in fake.call_args_list:
        assert call.kwargs["resampling_method"] == "sinc_interp_kaiser"
        assert call.kwargs["lowpass_filter_width"] == 64
        assert call.kwargs["rolloff"] == 0.999
        assert call.kwargs["dtype"] == torch.float32


def test_cached_resampling_matches_functional_after_half_precision_construction() -> None:
    from torchaudio.functional import resample

    perth_net = _FakePerthNet()
    perth_net.hp.sample_rate = 32_000
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
    finally:
        torch.set_default_dtype(previous_dtype)
    waveform = torch.linspace(-0.5, 0.5, 3528, dtype=torch.float32)
    watermarker.apply(_chunks(waveform))
    for cached in watermarker._resamplers:
        expected = resample(
            waveform,
            cached.orig_freq,
            cached.new_freq,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.999,
        )
        torch.testing.assert_close(cached(waveform), expected, rtol=0, atol=0)


def test_watermark_delta_taper_preserves_chunk_edge_samples() -> None:
    original = torch.linspace(-0.5, 0.5, 16).repeat(2, 1)
    marked = original + 0.125

    result = watermark.taper_watermark_delta(marked, original, edge_samples=4)

    torch.testing.assert_close(result[:, 0], original[:, 0])
    torch.testing.assert_close(result[:, -1], original[:, -1])
    torch.testing.assert_close(result[:, 4:-4], marked[:, 4:-4])


def test_short_waveform_is_left_unchanged() -> None:
    perth_net = _FakePerthNet()
    watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
    original = torch.tensor([0.1, -0.2, 0.3, -0.4])

    result = watermarker.apply(_chunks(original.clone()))

    assert perth_net.encoder.batch_sizes == []
    torch.testing.assert_close(result[0], original)


def test_disabled_watermarker_is_identity() -> None:
    waveforms = [torch.zeros(16), torch.ones(8)]
    chunks = _chunks(*waveforms)
    result = DisabledAudioWatermarker().apply(chunks)
    assert result[0] is waveforms[0]
    assert result[1] is waveforms[1]


@pytest.mark.parametrize("environment", [None, "1", "true", "off"])
def test_explicit_disable_bypasses_perth_load(environment, monkeypatch) -> None:
    if environment is None:
        monkeypatch.delenv("NEMOTRON_TTS_PERTH_WATERMARK", raising=False)
    else:
        monkeypatch.setenv("NEMOTRON_TTS_PERTH_WATERMARK", environment)
    with mock.patch.object(watermark, "_load_perth_net") as load_perth:
        created = watermark.create_audio_watermarker("cpu", sample_rate=22_050, enabled=False)

    load_perth.assert_not_called()
    assert isinstance(created, DisabledAudioWatermarker)


@pytest.mark.parametrize("environment", [None, "1", "off"])
def test_default_enable_ignores_environment(environment, monkeypatch) -> None:
    if environment is None:
        monkeypatch.delenv("NEMOTRON_TTS_PERTH_WATERMARK", raising=False)
    else:
        monkeypatch.setenv("NEMOTRON_TTS_PERTH_WATERMARK", environment)
    with mock.patch.object(watermark, "_load_perth_net", return_value=_FakePerthNet()) as load_perth:
        created = watermark.create_audio_watermarker("cpu", sample_rate=22_050)
    assert isinstance(created, TaperedPerthWatermarker)
    load_perth.assert_called_once_with("cpu", models_dir=None)


@pytest.mark.parametrize("enabled", [False, True])
def test_codec_config_watermark_round_trip(enabled) -> None:
    config = EasyMagpieCodecConfig(watermark_enabled=enabled)
    assert EasyMagpieCodecConfig.from_dict(config.to_dict()).watermark_enabled is enabled
    assert EasyMagpieCodecConfig().watermark_enabled is True


@pytest.mark.parametrize("enabled", ["true", "false", 0, 1, None])
def test_watermark_config_requires_boolean(enabled) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        EasyMagpieCodecConfig(watermark_enabled=enabled)
    with (
        mock.patch.object(watermark, "_load_perth_net") as load_perth,
        pytest.raises(ValueError, match="must be a boolean"),
    ):
        watermark.create_audio_watermarker("cpu", sample_rate=22_050, enabled=enabled)
    load_perth.assert_not_called()


@pytest.mark.parametrize("environment", ["1", "off"])
def test_enabled_initialization_failure_is_fatal(environment, monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_TTS_PERTH_WATERMARK", environment)
    with (
        mock.patch.object(watermark, "_load_perth_net", side_effect=ValueError("bad checkpoint")),
        pytest.raises(RuntimeError, match="could not be initialized"),
    ):
        watermark.create_audio_watermarker("cpu", sample_rate=22_050, enabled=True)


def test_encoder_is_compiled_on_cuda() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cuda:0",
        ap=_FakeAudioProcessor(),
    )
    compiled = mock.Mock(name="compiled_encoder", side_effect=lambda mag: (mag, None))

    with (
        mock.patch.object(watermark, "normalized_device", return_value="cuda:0"),
        mock.patch.object(torch, "compile", return_value=compiled) as compile_fn,
        mock.patch.object(torch.cuda, "is_available", return_value=False),
    ):
        result = TaperedPerthWatermarker("cuda:0", sample_rate=22_050, perth_net=perth_net)

    compile_fn.assert_called_once_with(encoder, mode="default", dynamic=True)
    assert result.perth_net.encoder is compiled


def test_encoder_is_not_compiled_on_cpu() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cpu",
    )
    with (mock.patch.object(torch, "compile") as compile_fn,):
        result = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)

    compile_fn.assert_not_called()
    assert result.perth_net.encoder is encoder


def test_load_perth_net_imports_perthnet_without_sys_modules_shim() -> None:
    source = inspect.getsource(watermark._load_perth_net)
    assert "_install_perth_net_namespace" not in source
    assert "sys.modules" not in source
    assert "perth.perth_net.perth_net_implicit.model.perth_net" in source


@pytest.mark.parametrize("fail", [False, True])
def test_perth_load_uses_float32_and_restores_default_dtype(fail, tmp_path) -> None:
    (tmp_path / "implicit").mkdir()
    (tmp_path / "implicit" / "hparams.yaml").write_text("{}")

    def load(*_args, **_kwargs):
        assert torch.get_default_dtype() == torch.float32
        if fail:
            raise ValueError("bad checkpoint")
        return torch.nn.Linear(1, 1)

    module = types.SimpleNamespace(PerthNet=types.SimpleNamespace(load=load))
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with mock.patch.dict("sys.modules", {"perth.perth_net.perth_net_implicit.model.perth_net": module}):
            if fail:
                with pytest.raises(ValueError, match="bad checkpoint"):
                    watermark._load_perth_net("cpu", models_dir=tmp_path)
            else:
                result = watermark._load_perth_net("cpu", models_dir=tmp_path)
                assert next(result.parameters()).dtype == torch.float32
        assert torch.get_default_dtype() == torch.float16
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("environment", ["1", "off"])
def test_config_enable_initializes_and_forwards_models_dir(environment, monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_TTS_PERTH_WATERMARK", environment)
    with mock.patch.object(watermark, "_load_perth_net", return_value=_FakePerthNet()) as load_perth:
        config = EasyMagpieCodecConfig(watermark_enabled=True)
        created = watermark.create_audio_watermarker(
            "cpu", sample_rate=22_050, enabled=config.watermark_enabled, models_dir="/ckpt/watermark/perth"
        )

    assert isinstance(created, TaperedPerthWatermarker)
    load_perth.assert_called_once_with("cpu", models_dir="/ckpt/watermark/perth")
