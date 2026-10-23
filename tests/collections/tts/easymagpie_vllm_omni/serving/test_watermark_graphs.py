# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torchaudio

from easymagpie_vllm_omni import watermark_graphs as graph_module
from easymagpie_vllm_omni.watermark_graphs import PerthCudaGraphs, _PreparedInverse


@pytest.mark.parametrize('n_fft,win,hop', [(2048, 2048, 320), (2049, 2049, 320), (2048, 1024, 320)])
@pytest.mark.parametrize('frames', [2, 3, 17])
def test_prepared_inverse_preserves_native_reconstruction(n_fft, win, hop, frames):
    inverse = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win, hop_length=hop)
    prepared = _PreparedInverse(inverse, frames)
    spectrum = torch.randn(2, n_fft // 2 + 1, frames, dtype=torch.complex64)
    assert prepared.supported
    actual = prepared(spectrum)
    assert actual.shape[-1] == n_fft + hop * (frames - 1) - 2 * (n_fft // 2)
    assert torch.equal(actual, inverse(spectrum))
    assert not prepared.used_fallback


@pytest.mark.parametrize('flags', [{'normalized': True}, {'center': False}, {'pad': 3}, {'onesided': False}])
def test_unsupported_inverse_is_rejected_before_capture(flags):
    inverse = torchaudio.transforms.InverseSpectrogram(n_fft=32, hop_length=8, window_fn=torch.hamming_window, **flags)
    prepared = _PreparedInverse(inverse, 5)
    spectrum = torch.randn(2, 32 if flags.get('onesided') is False else 17, 5, dtype=torch.complex64)
    assert not prepared.supported
    assert torch.equal(prepared(spectrum), inverse(spectrum))
    assert prepared.used_fallback


def test_nola_failure_is_preserved():
    inverse = torchaudio.transforms.InverseSpectrogram(n_fft=32, hop_length=8)
    inverse.window.zero_()
    prepared = _PreparedInverse(inverse, 5)
    assert not prepared.supported
    with pytest.raises(RuntimeError, match='window overlap add min'):
        prepared(torch.randn(2, 17, 5, dtype=torch.complex64))


@pytest.mark.parametrize('change', ['dtype', 'frames', 'length'])
def test_unexpected_spectrum_metadata_marks_warmup_as_uncapturable(change):
    inverse = torchaudio.transforms.InverseSpectrogram(n_fft=32, hop_length=8)
    prepared = _PreparedInverse(inverse, 5)
    spectrum = torch.randn(
        2, 17, 4 if change == 'frames' else 5, dtype=torch.complex128 if change == 'dtype' else torch.complex64
    )
    length = 37 if change == 'length' else None
    assert torch.equal(prepared(spectrum, length=length), inverse(spectrum, length=length))
    assert prepared.used_fallback


def test_warm_inverse_contains_no_scalar_read(monkeypatch):
    inverse = torchaudio.transforms.InverseSpectrogram(n_fft=32, hop_length=8)
    spectrum = torch.randn(2, 17, 5, dtype=torch.complex64)
    expected = inverse(spectrum)
    prepared = _PreparedInverse(inverse, 5)

    def forbidden(*args, **kwargs):
        raise AssertionError('scalar read during reconstruction')

    monkeypatch.setattr(torch.Tensor, 'item', forbidden)
    monkeypatch.setattr(torch.Tensor, '__bool__', forbidden)
    assert torch.equal(prepared(spectrum), expected)


def native_net():
    hp = SimpleNamespace(
        n_fft=32, window_size=32, hop_size=8, window_fn='hann', stft_magnitude_min=1e-9, sample_rate=32000
    )
    module = pytest.importorskip("perth.perth_net.perth_net_implicit.audio_processor")
    return SimpleNamespace(hp=hp, ap=module.AudioProcessor(hp))


@pytest.mark.parametrize('case', ['cpu', 'custom', 'dtype', 'inverse_flags', 'limit'])
def test_uncapturable_or_excess_shapes_never_start_cuda_capture(case, monkeypatch):
    net = native_net()
    # Metadata-only stand-in reaches eligibility checks without touching CUDA.
    originals = SimpleNamespace(is_cuda=True, dtype=torch.float32, device=torch.device('cpu'), shape=(2, 128))
    cache = PerthCudaGraphs(net, 32000)
    if case == 'cpu':
        originals = torch.zeros(2, 128)
    elif case == 'custom':
        net.ap = SimpleNamespace()
    elif case == 'dtype':
        originals.dtype = torch.float64
    elif case == 'inverse_flags':
        net.ap.inv_spectrogram.normalized = True
    else:
        cache.graphs = {index: object() for index in range(64)}

    def forbidden(*args, **kwargs):
        raise AssertionError('ineligible input attempted capture or DSP')

    monkeypatch.setattr(torch.cuda, 'Stream', forbidden)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', forbidden)
    if case == 'limit':
        monkeypatch.setattr(graph_module, '_PreparedInverse', forbidden)
    assert cache(originals, forbidden) is None
    assert len(cache.graphs) == (64 if case == 'limit' else 0)
