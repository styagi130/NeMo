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
"""Batched decode preparation preserves the single-request state machine."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stacking", [0, 1, 2])
def test_batched_preparation_matches_individual_requests(device, stacking):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    reference = _model(device, stacking)
    batched = _model(device, stacking)
    device = reference._dec_audio_codes.device
    ids = torch.zeros(32, device=device, dtype=torch.long)
    infos = [
        {"decode_offset": 0},
        {"decode_offset": 3, "phoneme_ended": True},
        {"decode_offset": 5},
        {"decode_offset": 6, "phoneme_ended": torch.tensor(False, device=device)},
        {"decode_offset": 7, "phoneme_ended": torch.tensor(True, device=device)},
        {"decode_offset": 8},
        # Partial feedback retains the original single-row behavior.
        {"decode_offset": 9, "last_audio_codes": torch.tensor([12], device=device)},
    ]
    for index, info in enumerate(infos):
        info.update(text_tokens=[10, 11, 12], text_token=[13, 14, 15, 16, 17, 18, 19], text_token_start=3)
        info.setdefault("last_audio_codes", torch.arange(4, device=device) + index * 7)
        if stacking and index >= 3:
            tokens = torch.full((1, stacking), 9, device=device)
            if index == 3:
                tokens[0, -1] = reference.phoneme_eos_id
            info["last_phoneme_token"] = tokens
    retained_ended = []
    for epoch, offsets in enumerate(([11, 2, 8, 1, 15, 4, 19], [4, 15, 1, 8, 2, 11, 19])):
        rows = list(zip(offsets, infos, strict=True))
        batched.prepare_decode_batch(rows, device)
        for start, info in rows:
            expected = reference._preprocess_decode(ids[start : start + 1], start, device, deepcopy(info))
            actual = batched._preprocess_decode(ids[start : start + 1], start, device, deepcopy(info))
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            _assert_updates(actual[2], expected[2])
            if stacking:
                retained_ended.append((actual[2]["phoneme_ended"], expected[2]["phoneme_ended"].clone()))
            info.update(actual[2])
        assert not batched._prepared_decode_updates
        _assert_buffers(reference, batched, stacking)
        # Completed slots are admitted again with fresh BOS/startup metadata.
        if epoch == 0:
            infos[0] = {"decode_offset": 3, "text_tokens": [21, 22, 23, 24]}
    for actual, expected in retained_ended:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_failed_or_cancelled_preparation_does_not_leak_metadata(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _model(device, 2)
    device = model._dec_audio_codes.device
    ids = torch.zeros(32, device=device, dtype=torch.long)
    model.prepare_decode_batch([(1, {"decode_offset": 5}), (2, {"decode_offset": 6})], device)
    model.clear_decode_batch()
    _, _, fresh = model._preprocess_decode(ids[1:2], 1, device, {"decode_offset": 3})
    assert fresh["decode_offset"] == 4
    assert not fresh["phoneme_ended"]
    assert model._dec_audio_valid[1] == 0
    assert torch.equal(model._dec_phoneme_tokens[1], torch.tensor([10, 10], device=device))


def test_runner_prepares_only_single_token_rows_after_reordering():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    observed = []
    runner.model = SimpleNamespace(prepare_decode_batch=lambda rows, device: observed.append((rows, device)))
    runner.query_start_loc = SimpleNamespace(cpu=np.array([0, 4, 5, 6]))
    runner._omni_num_scheduled_tokens_np = np.array([4, 1, 1])
    runner.model_intermediate_buffer = {key: {"decode_offset": i} for i, key in enumerate(("c", "a", "b"))}
    runner._maybe_run_batch_preprocess(["c", "a", "b"], torch.device("cpu"))
    assert observed == [([(4, {"decode_offset": 1}), (5, {"decode_offset": 2})], torch.device("cpu"))]


@pytest.mark.parametrize("failed", [False, True])
def test_runner_clears_prepared_rows_on_success_and_failure(monkeypatch, failed):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    cleared = []
    runner.model = SimpleNamespace(clear_decode_batch=lambda: cleared.append(True))

    def preprocess(*args):
        if failed:
            raise RuntimeError("preprocessing failed")
        return "prepared"

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    if failed:
        with pytest.raises(RuntimeError, match="preprocessing failed"):
            runner._preprocess(None, 2)
    else:
        assert runner._preprocess(None, 2) == "prepared"
    assert cleared == [True]


def _model(device, stacking):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(phoneme_stacking_factor=stacking, audio_bos_id=1025)
    model.has_phoneme = stacking > 0
    model.phonemes_delay = 3
    model.phoneme_bos_id = 10
    model.phoneme_eos_id = 11
    model.speech_delay = 5
    model.embedding_dim = 3
    model.num_codebooks = 4
    model._combined_embeddings = torch.zeros(32, 3, device=device)
    model._dec_text_tokens = torch.arange(32, dtype=torch.long, device=device)
    model._dec_text_mask = torch.zeros(32, dtype=torch.long, device=device)
    model._dec_audio_codes = torch.arange(128, dtype=torch.long, device=device).view(32, 4)
    model._dec_audio_valid = torch.zeros(32, dtype=torch.long, device=device)
    if stacking:
        model._dec_phoneme_tokens = torch.arange(32 * stacking, dtype=torch.long, device=device).view(32, stacking)
        model._dec_phoneme_valid = torch.zeros(32, dtype=torch.long, device=device)
    return model


def _assert_updates(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


def _assert_buffers(reference, actual, stacking):
    names = ["_dec_text_tokens", "_dec_text_mask", "_dec_audio_codes", "_dec_audio_valid"]
    if stacking:
        names += ["_dec_phoneme_tokens", "_dec_phoneme_valid"]
    for name in names:
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
