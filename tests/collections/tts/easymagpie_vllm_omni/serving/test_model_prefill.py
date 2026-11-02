# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoderOutput
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.easymagpie import _append_rolling_tail as append_tail
from easymagpie_vllm_omni.easymagpie import _build_rolling_context as build_context
from easymagpie_vllm_omni.easymagpie import _context_codes as context_codes
from safetensors.torch import save_file
from torch import nn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stacking", [1, 2])
def test_phoneme_eos_is_fed_once_then_masked(device, stacking):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(phoneme_stacking_factor=stacking, audio_bos_id=1025)
    model.has_phoneme = True
    model.phonemes_delay = 1
    model.phoneme_bos_id = 10
    model.phoneme_eos_id = 11
    model.speech_delay = 99
    model.embedding_dim = 3
    model.num_codebooks = 2
    model._combined_embeddings = torch.zeros(2, 3, device=device)
    model._dec_text_tokens = torch.zeros(2, dtype=torch.long, device=device)
    model._dec_text_mask = torch.zeros(2, dtype=torch.long, device=device)
    model._dec_phoneme_tokens = torch.zeros(2, stacking, dtype=torch.long, device=device)
    model._dec_phoneme_valid = torch.zeros(2, dtype=torch.long, device=device)
    model._dec_audio_codes = torch.zeros(2, 2, dtype=torch.long, device=device)
    model._dec_audio_valid = torch.zeros(2, dtype=torch.long, device=device)
    input_ids = torch.zeros(1, dtype=torch.long, device=device)

    _, _, update = model._preprocess_decode(
        input_ids,
        0,
        input_ids.device,
        {
            "decode_offset": 2,
            "last_phoneme_token": torch.tensor([[3] * (stacking - 1) + [model.phoneme_eos_id]], device=device),
        },
    )

    assert model._dec_phoneme_valid[0].item() == 1
    assert update["phoneme_ended"].item() is True

    model._preprocess_decode(
        input_ids,
        1,
        input_ids.device,
        {
            "decode_offset": 3,
            "last_phoneme_token": torch.full((1, stacking), 3, device=device),
            "phoneme_ended": update["phoneme_ended"],
        },
    )

    assert model._dec_phoneme_valid[1].item() == 0
    assert update["phoneme_ended"].device == input_ids.device

    # A new request reuses the EOS request's physical slot with independent state.
    _, _, fresh = model._preprocess_decode(input_ids, 0, input_ids.device, {"decode_offset": 1})
    assert model._dec_phoneme_valid[0].item() == 1
    assert model._dec_phoneme_tokens[0].tolist() == [model.phoneme_bos_id] * stacking
    assert fresh["phoneme_ended"].item() is False
    assert update["phoneme_ended"].item() is True
    assert "phoneme_ended" in model.gpu_resident_buffer_keys


def test_two_stage_output_copies_codes_once_and_uses_async_output():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._single_stage_audio = False
    model._out_codes = torch.tensor([[1, 2], [3, 4]])
    hidden = torch.zeros(2, 3)

    output = model.make_omni_output(hidden)

    assert set(output.multimodal_outputs) == {"codes"}
    torch.testing.assert_close(output.multimodal_outputs["codes"]["audio"], model._out_codes)
    assert model.use_async_omni_output
    assert model.eager_omni_postprocess_before_async_output

    model._single_stage_audio = True
    output = model.make_omni_output(hidden)
    assert set(output.multimodal_outputs) == {"model_outputs"}
    torch.testing.assert_close(output.multimodal_outputs["model_outputs"], model._out_codes)


def test_text_prefill_embeddings_add_phoneme_bos_at_position_three():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=1)
    model.embedding_dim = 3
    model.has_phoneme = True
    model.phonemes_delay = 3
    model.phoneme_bos_id = 7
    model.text_embedding = nn.Embedding(32, 3)
    model.phoneme_embeddings = nn.ModuleList([nn.Embedding(16, 3)])

    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.phoneme_embeddings[0].weight.zero_()
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
        model.phoneme_embeddings[0].weight[7] = torch.tensor([0, 0, 10])

    rows = model._build_text_prefill_embeds(
        torch.device("cpu"),
        torch.float32,
        {"text_prefill_num": 4, "prefill_text_tokens": [10, 11, 12, 13]},
    )

    torch.testing.assert_close(
        rows,
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 10]], dtype=torch.float32),
    )


class _FakeReferenceCodecEncoder(nn.Module):
    def forward(self, audio, audio_lens, *, audio_frame_embedder=None):
        del audio_frame_embedder
        self.audio_data_ptr = audio.data_ptr()
        self.audio = audio.detach().clone()
        self.audio_lens = audio_lens.detach().clone()
        count = int(audio.shape[0])
        return EasyMagpieCodecEncoderOutput(
            acoustic_codes=torch.zeros(audio.shape[0], 4, 2, dtype=torch.long),
            acoustic_lens=torch.full((audio.shape[0],), 2, dtype=torch.long),
            reference_speaker_embeddings=torch.full((count, 4, 2), 9.0),
            reference_speaker_embedding_lens=torch.full((count,), 4, dtype=torch.long),
        )


def test_embed_multimodal_returns_reference_embeddings_when_codec_encoder_is_present():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: None,
        codec_samples_per_row=16,
        audio_user_speaking_id=17,
        require_user_audio_prefill=lambda: None,
    )
    model.model_path = "unused"
    model.speech_delay = 0
    model.num_codebooks = 2
    model.codec_encoder = _FakeReferenceCodecEncoder()
    model.code_predictor = SimpleNamespace(embed_audio_frame=lambda codes: codes.float())
    model._combined_embeddings = torch.zeros(1, 2)

    waveform = torch.tensor([1.0, 2.0, 3.0, 4.0])
    outputs = model.embed_multimodal(
        audio_values=waveform,
        audio_lens=torch.tensor([4]),
    )

    assert [output.shape for output in outputs] == [(4, 2)]
    torch.testing.assert_close(outputs[0], torch.full((4, 2), 9.0))
    torch.testing.assert_close(model.codec_encoder.audio[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert model.codec_encoder.audio_data_ptr == waveform.data_ptr()
    torch.testing.assert_close(model.codec_encoder.audio_lens, torch.tensor([4]))


def test_embed_multimodal_rejects_reference_audio_when_codec_encoder_is_absent():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: (_ for _ in ()).throw(RuntimeError("no reference-audio encoder"))
    )
    model.codec_encoder = None

    with pytest.raises(RuntimeError, match="no reference-audio encoder"):
        model.embed_multimodal(
            audio_values=[torch.ones(4)],
            audio_lens=torch.tensor([4]),
        )


def _raw_reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._maybe_set_lt_sampling_params = lambda info: None
    model._build_reference_audio_prefill_chunk = lambda **kwargs: (
        torch.full((kwargs["span_len"], 2), 7.0),
        2,
        False,
        2,
    )
    model._build_prefill_embeds = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("known-speaker path must not run")
    )
    model.speech_delay = 2
    return model


def test_prefill_infers_raw_reference_when_speaker_id_is_absent():
    model = _raw_reference_prefill_model()

    _, rows, _ = model._preprocess_prefill(
        input_ids=torch.tensor([1, 1]),
        input_embeds=torch.ones(2, 2),
        span_len=2,
        device=torch.device("cpu"),
        info_dict={
            "_omni_num_computed_tokens": 0,
            "_omni_prompt_len": 2,
        },
    )

    torch.testing.assert_close(rows, torch.full((2, 2), 7.0))


def test_reference_prefill_rejects_non_default_task_mode():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="Only task_mode_id=0"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"task_mode_id": 1},
            text_tokens=[],
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def _reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        audio_input_token_id=1,
        require_user_audio_prefill=lambda: None,
        text_prefill_num=4,
    )
    model.model_path = "unused"
    model.embedding_dim = 3
    model.has_phoneme = False
    model.task_embedding = None
    model.num_task_embeddings = 0
    model.text_embedding = nn.Embedding(32, 3)
    model._encode_context_text = lambda text, device: torch.tensor([20], device=device)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[20] = torch.tensor([0, 8, 0])
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
    return model


def test_reference_and_user_prefill_overlays_task_before_reference_boundary_is_known():
    model = _reference_prefill_model()
    model.arch.audio_input_token_id = 1
    model.task_embedding = nn.Embedding(1, 3)
    model.num_task_embeddings = 1
    with torch.no_grad():
        model.task_embedding.weight[0] = torch.tensor([9, 0, 0])

    rows, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=torch.ones(2, 3),
        info_dict={"context_text": "[EN]", "text_prefill_num": 4},
        text_tokens=[],
        input_ids=torch.tensor([0, 1]),
        offset=0,
        span_len=2,
        prompt_len=8,
    )

    assert conditioning_len == 0
    assert has_user_audio is False
    assert reference_rows == 0
    torch.testing.assert_close(rows[0], torch.tensor([9, 0, 0], dtype=torch.float32))
    torch.testing.assert_close(rows[1], torch.ones(3))


def test_reference_audio_prefill_rejects_non_default_task_mode():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="Only task_mode_id=0"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"task_mode_id": 1},
            text_tokens=[],
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def test_reference_prefill_rejects_initial_user_only_layout_without_speaker_id():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="must place reference audio before context rows"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"context_text": "[EN]", "text_prefill_num": 0},
            text_tokens=[],
            input_ids=torch.tensor([1, 1]),
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def test_reference_audio_prefill_preserves_speaker_rows_and_is_chunk_invariant():
    model = _reference_prefill_model()
    info = {
        "context_text": "[EN]",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
        "reference_audio_num_rows": 2,
    }
    input_embeds = torch.ones(7, 3)
    input_ids = torch.tensor([1, 1, 0, 0, 0, 0, 0])

    whole, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=input_embeds,
        info_dict=info,
        text_tokens=[],
        input_ids=input_ids,
        offset=0,
        span_len=7,
        prompt_len=7,
    )
    chunks = torch.cat(
        [
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[:4],
                info_dict=info,
                text_tokens=[],
                input_ids=input_ids[:4],
                offset=0,
                span_len=4,
                prompt_len=7,
            )[0],
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[4:],
                info_dict=info,
                text_tokens=[],
                input_ids=input_ids[4:],
                offset=4,
                span_len=3,
                prompt_len=7,
            )[0],
        ]
    )

    assert conditioning_len == 3
    assert has_user_audio is False
    assert reference_rows == 2
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], torch.ones(2, 3))
    torch.testing.assert_close(whole[2], torch.tensor([0, 8, 0], dtype=torch.float32))
    torch.testing.assert_close(
        whole[3:],
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=torch.float32),
    )


def test_reference_only_prefill_state_remains_reference_layout_across_scheduler_chunks():
    model = _reference_prefill_model()
    model.speech_delay = 2
    model._combined_embeddings = torch.zeros(1, 3)
    model._maybe_set_lt_sampling_params = lambda info: None
    base_info = {
        "_omni_prompt_len": 7,
        "context_text": "[EN]",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
    }

    _, first_rows, update = model._preprocess_prefill(
        input_ids=torch.tensor([1, 1, 0, 0]),
        input_embeds=torch.ones(4, 3),
        span_len=4,
        device=torch.device("cpu"),
        info_dict={**base_info, "_omni_num_computed_tokens": 0},
    )
    _, second_rows, _ = model._preprocess_prefill(
        input_ids=torch.tensor([0, 0, 0]),
        input_embeds=torch.ones(3, 3),
        span_len=3,
        device=torch.device("cpu"),
        info_dict={
            **base_info,
            **update,
            "_omni_num_computed_tokens": 4,
        },
    )

    torch.testing.assert_close(
        torch.cat((first_rows, second_rows)),
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
            ],
            dtype=torch.float32,
        ),
    )


def test_user_audio_prefill_composition_is_chunk_invariant():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.speech_delay = 2
    model.embedding_dim = 3
    model.has_phoneme = False
    model.text_embedding = nn.Embedding(32, 3)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[10] = torch.tensor([1, 0, 0])
        model.text_embedding.weight[11] = torch.tensor([2, 0, 0])

    conditioning = torch.tensor([[9, 0, 0], [8, 0, 0]], dtype=torch.float32)
    input_embeds = torch.ones(6, 3)
    kwargs = {
        "conditioning": conditioning,
        "text_tokens": [10, 11],
        "prompt_len": 6,
    }

    whole = model._build_user_audio_prefill_chunk(
        input_embeds=input_embeds,
        offset=0,
        span_len=6,
        **kwargs,
    )
    chunks = torch.cat(
        [
            model._build_user_audio_prefill_chunk(
                input_embeds=input_embeds[:3],
                offset=0,
                span_len=3,
                **kwargs,
            ),
            model._build_user_audio_prefill_chunk(
                input_embeds=input_embeds[3:],
                offset=3,
                span_len=3,
                **kwargs,
            ),
        ]
    )

    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], conditioning)
    torch.testing.assert_close(whole[2:4], torch.ones(2, 3))
    torch.testing.assert_close(
        whole[4:],
        torch.tensor([[2, 1, 1], [3, 1, 1]], dtype=torch.float32),
    )


class _FakeCombinedCodecEncoder(nn.Module):
    def forward(self, audio, audio_lens, *, audio_frame_embedder=None):
        del audio_frame_embedder
        self.audio = audio.detach().clone()
        self.audio_lens = audio_lens.detach().clone()
        code_lens = torch.div(audio_lens + 1279, 1280, rounding_mode="floor")
        codes = torch.zeros(audio.shape[0], 2, int(code_lens.max()), dtype=torch.long)
        codes[:, 0] = 1
        codes[:, 1] = 2
        count = int(audio.shape[0])
        return EasyMagpieCodecEncoderOutput(
            acoustic_codes=codes,
            acoustic_lens=code_lens,
            reference_speaker_embeddings=torch.full((count, 4, 2), 9.0),
            reference_speaker_embedding_lens=torch.full((count,), 4, dtype=torch.long),
        )


def test_embed_multimodal_batches_reference_and_variable_user_audio():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        audio_user_speaking_id=37,
        codec_samples_per_row=1280,
        ensure_reference_audio_available=lambda: None,
        require_user_audio_prefill=lambda: None,
    )
    model.model_path = "unused"
    model.speech_delay = 5
    model.num_codebooks = 2
    model.codec_encoder = _FakeCombinedCodecEncoder()
    model.code_predictor = SimpleNamespace(embed_audio_frame=lambda codes: codes.float())
    model._combined_embeddings = torch.zeros(1, 2)

    reference = torch.tensor([1.0, 2.0, 3.0, 4.0])
    user = torch.arange(100, dtype=torch.float32)
    outputs = model.embed_multimodal(
        audio_values=[reference, user],
        audio_lens=torch.tensor([4, 100]),
        audio_output_kinds=torch.tensor([1, 0]),
    )

    assert [output.shape for output in outputs] == [(4, 2), (6, 2)]
    assert model.codec_encoder.audio.shape == (2, 6400)
    assert model.codec_encoder.audio_lens.tolist() == [4, 6400]
    torch.testing.assert_close(model.codec_encoder.audio[0, :4], reference)
    assert torch.count_nonzero(model.codec_encoder.audio[0, 4:]) == 0
    torch.testing.assert_close(model.codec_encoder.audio[1, -100:], user)
    torch.testing.assert_close(outputs[0], torch.full((4, 2), 9.0))
    torch.testing.assert_close(outputs[1][0], torch.tensor([37.0, 37.0]))
    torch.testing.assert_close(outputs[1][1], torch.tensor([38.0, 39.0]))


def test_reference_and_user_audio_prefill_is_chunk_invariant():
    model = _reference_prefill_model()
    model.speech_delay = 2
    info = {
        "context_text": "[EN]",
        "text_prefill_num": 1,
        "reference_audio_num_rows": 2,
    }
    input_embeds = torch.ones(7, 3)
    kwargs = {
        "info_dict": info,
        "text_tokens": [10, 11],
        "prompt_len": 7,
    }

    whole, conditioning_len, has_user_audio, reference_rows = model._build_reference_audio_prefill_chunk(
        input_embeds=input_embeds,
        offset=0,
        span_len=7,
        **kwargs,
    )
    chunks = torch.cat(
        [
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[:4],
                offset=0,
                span_len=4,
                **kwargs,
            )[0],
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[4:],
                offset=4,
                span_len=3,
                **kwargs,
            )[0],
        ]
    )

    assert conditioning_len == 3
    assert has_user_audio is True
    assert reference_rows == 2
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], torch.ones(2, 3))
    torch.testing.assert_close(whole[2], torch.tensor([0, 8, 0], dtype=torch.float32))
    torch.testing.assert_close(whole[3:5], torch.ones(2, 3))
    torch.testing.assert_close(
        whole[5:],
        torch.tensor([[2, 1, 1], [3, 1, 1]], dtype=torch.float32),
    )


def test_user_audio_prefill_detection_handles_resumed_audio_only():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(condition_on_user_speech=True, audio_input_token_id=1)

    common = {"conditioning_len": 2, "prompt_len": 8, "legacy_prompt_len": 6}
    assert model._is_user_audio_prefill_chunk(input_ids=torch.tensor([0, 0]), offset=0, **common)
    assert model._is_user_audio_prefill_chunk(input_ids=torch.tensor([1, 1]), offset=6, **common)
    assert not model._is_user_audio_prefill_chunk(input_ids=torch.tensor([0]), offset=8, **common)


def test_user_audio_prefill_detection_keeps_legacy_text_prefill_unchanged():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(condition_on_user_speech=True, audio_input_token_id=1)

    assert not model._is_user_audio_prefill_chunk(
        input_ids=torch.tensor([0, 0]),
        offset=0,
        conditioning_len=2,
        prompt_len=6,
        legacy_prompt_len=6,
    )


def test_first_agent_decode_uses_user_speaking_end_token():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        phoneme_stacking_factor=0,
        audio_bos_id=32,
        audio_user_speaking_end_id=38,
        use_user_speaking_end_token=True,
    )
    model.embedding_dim = 3
    model.num_codebooks = 2
    model.speech_delay = 5
    model.has_phoneme = False
    model._combined_embeddings = torch.zeros(1, 3)
    model._dec_text_tokens = torch.zeros(1, dtype=torch.long)


def test_load_weights_maps_hf_backbone_names_with_auto_loader():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.backbone = nn.Module()
    model.backbone.embed_tokens = nn.Embedding(4, 3)
    model.backbone.layers = nn.ModuleList([nn.Module()])
    model.backbone.layers[0].mixer = nn.Module()
    model.backbone.layers[0].mixer.A = nn.Parameter(torch.zeros(3))
    model.text_embedding = nn.Embedding(4, 3)
    model.code_predictor = SimpleNamespace(init_forbidden_mask=lambda: None)
    embeddings = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    mamba_a = torch.tensor([1.0, 2.0, 3.0])

    loaded = model.load_weights(
        [
            ("decoder.embeddings.weight", embeddings),
            ("decoder.layers.0.mixer.A_log", mamba_a),
            ("text_embedding.weight", embeddings + 1),
        ]
    )

    assert loaded == {"backbone.embed_tokens.weight", "backbone.layers.0.mixer.A", "text_embedding.weight"}
    torch.testing.assert_close(model.backbone.embed_tokens.weight, embeddings)
    torch.testing.assert_close(model.backbone.layers[0].mixer.A, mamba_a)
    torch.testing.assert_close(model.text_embedding.weight, embeddings + 1)


class TestRollingContext:

    def test_existing_reference_rows_and_direct_generated_embeddings_match_native_layout(self):
        saved = torch.arange(384).reshape(96, 4).float()

        def embed(codes):
            return codes.float()

        for count in (1, 2, 62, 63, 64, 75, 124, 125):
            tail = torch.arange(count * 2).reshape(2, count).remainder(16)
            actual = build_context(
                saved, tail, embed_audio=embed, codebook_size=16, stacking_factor=2, total_frames=250, has_audio=True
            )
            rows = 1 + min((250 - count + 1) // 2, 94)
            expected = context_codes(tail, bos_id=18, eos_id=19, stacking_factor=2).T.float()[1:]
            torch.testing.assert_close(actual[:rows], saved[:rows], atol=0, rtol=0)
            torch.testing.assert_close(actual[rows:], expected, atol=0, rtol=0)
            assert not bool((actual[rows:] == 18).any())
        assert actual.shape == (127, 4)

    def test_generation_specials_are_rejected_in_model_namespace(self):
        for bad in (-1, 16, 17, 18, 19):
            with pytest.raises(ValueError, match='invalid generated'):
                build_context(
                    torch.randn(96, 4),
                    torch.tensor([[bad], [1]]),
                    embed_audio=None,
                    codebook_size=16,
                    stacking_factor=2,
                    total_frames=250,
                    has_audio=True,
                )

    def select(self, rows, previous=None, n=3):
        return append_tail(previous, torch.tensor(rows), codebooks=2, stacking_factor=2, eos_id=99, maximum=n)

    def test_eos_in_second_stacked_frame_keeps_first_frame(self):
        (tail, ended) = self.select([[1, 2, 11, 12], [3, 99, 13, 14]])
        assert ended
        assert tail.tolist() == [[1, 2, 3], [11, 12, 13]]

    def test_eos_in_any_codebook_discards_that_frame_and_every_later_frame(self):
        (tail, ended) = self.select([[1, 2, 11, 12], [3, 4, 99, 14], [5, 6, 15, 16]])
        assert ended
        assert tail.tolist() == [[1, 2], [11, 12]]

    def test_eos_first_frame_preserves_previous_tail_without_eos(self):
        previous = torch.tensor([[1, 2, 3], [11, 12, 13]])
        (tail, ended) = self.select([[99, 4, 14, 15]], previous)
        assert ended
        assert tail.tolist() == previous.tolist()

    def test_tail_is_bounded_across_decode_steps(self):
        (tail, ended) = self.select([[1, 2, 11, 12]])
        assert not ended
        (tail, ended) = self.select([[3, 4, 13, 14]], tail)
        assert not ended
        assert tail.tolist() == [[2, 3, 4], [12, 13, 14]]

    def test_context_special_tokens_and_stacking_match_nemo_layout(self):
        actual = context_codes(torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]]), bos_id=98, eos_id=99, stacking_factor=2)
        assert actual.tolist() == [[98, 1, 3, 99], [98, 2, 4, 99], [98, 11, 13, 99], [98, 12, 14, 99]]

    def test_factor_one_and_odd_length_have_exact_context_boundaries(self):
        for factor in (1, 2, 3):
            actual = context_codes(torch.arange(10).reshape(2, 5), bos_id=98, eos_id=99, stacking_factor=factor)
            assert (actual[:, 0] == 98).all()
            assert actual.shape[1] == 1 + (6 + factor - 1) // factor


class _RollingEmbedder:
    def embed_audio_frame(self, codes):
        self.last_codes = codes.clone()
        return codes.float()


class TestRollingModel:

    def setup_method(self):
        self.model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
        torch.nn.Module.__init__(self.model)
        self.model.has_phoneme = False
        self.model.arch = SimpleNamespace(
            num_audio_codebooks=2, frame_stacking_factor=2, audio_eos_id=17, codebook_size=16
        )
        self.model.code_predictor = _RollingEmbedder()
        self.model.rolling_enabled = True
        self.model.rolling_total_frames = 250
        self.model.rolling_tail_limit = 125
        self.model.rolling_original_has_audio = {'voice': True}
        self.model.speech_delay = 5
        self.model.embedding_dim = 4
        self.model.register_buffer('original', torch.arange(384).reshape(96, 4).float())
        self.model._speaker_embedding_buffers = {'voice': 'original'}

    def test_single_row_rolling_prefill_uses_scheduler_phase(self):
        self.model._preprocess_decode = lambda *args: 'decode'
        self.model._preprocess_prefill = lambda *args: 'prefill'
        inputs = torch.zeros(1, dtype=torch.long)
        assert self.model.preprocess(inputs, None, _omni_is_prefill=False) == 'decode'
        assert (
            self.model.preprocess(inputs, None, context_tail_tokens=3, rolling_voice='voice', _omni_is_prefill=True)
            == 'prefill'
        )
        assert (
            self.model.preprocess(
                inputs,
                None,
                additional_information=dict(context_tail_tokens=3, rolling_voice='voice'),
                _omni_is_prefill=True,
            )
            == 'prefill'
        )
        assert self.model.postprocess_uses_req_infos
        assert 'rolling_tail' in self.model.gpu_resident_buffer_keys

    def test_interleaved_requests_only_emit_their_own_context(self):
        hidden = torch.zeros(2, 4)
        mm = {'codes': {'audio': torch.tensor([[1, 2, 3, 4], [5, 17, 6, 7]])}}
        a = self.model.postprocess(
            hidden[:1], mm, context_tail_tokens=3, rolling_voice='voice', rolling_request_key='a'
        )
        b = self.model.postprocess(
            hidden[1:],
            mm,
            context_tail_tokens=3,
            rolling_voice='voice',
            rolling_request_key='b',
            request_id='internal-b',
        )
        assert set(mm['rolling_contexts']) == {'internal-b'}
        assert a['rolling_tail'].tolist() == [[1, 2], [3, 4]]
        assert b['rolling_tail'].tolist() == [[5], [6]]
        later = {'codes': {'audio': torch.tensor([[8, 17, 9, 10]])}}
        self.model.postprocess(
            hidden[:1],
            later,
            context_tail_tokens=3,
            rolling_tail=a['rolling_tail'],
            rolling_voice='voice',
            rolling_request_key='a',
        )
        assert set(later['rolling_contexts']) == {'a'}
        assert not torch.equal(later['rolling_contexts']['a'], mm['rolling_contexts']['internal-b'])

    def test_prefill_and_pre_speech_predictions_do_not_enter_the_tail(self):
        for flags in [dict(_omni_is_prefill=True), dict(decode_offset=5)]:
            mm = {'codes': {'audio': torch.full((1, 4), 17)}}
            result = self.model.postprocess(torch.zeros(1, 4), mm, context_tail_tokens=3, **flags)
            assert 'rolling_tail' not in result
            assert 'rolling_contexts' not in mm

    def test_short_generated_payload_keeps_saved_prefix_and_native_context_eos(self):
        mm = {'codes': {'audio': torch.tensor([[2, 17, 3, 4]])}}
        result = self.model.postprocess(
            torch.zeros(1, 4), mm, context_tail_tokens=125, rolling_voice='voice', rolling_request_key='a'
        )
        assert result['rolling_tail'].tolist() == [[2], [3]]
        context = mm['rolling_contexts']['a']
        torch.testing.assert_close(context[:95], self.model.original[:95], rtol=0, atol=0)
        assert context.shape == (96, 4)
        assert not bool((self.model.code_predictor.last_codes == 17).any())
        assert bool((self.model.code_predictor.last_codes == 19).any())
        assert result['rolling_ended']
        other = {'codes': {'audio': torch.tensor([[2, 17, 3, 4]])}}
        self.model.postprocess(torch.zeros(1, 4), other, context_tail_tokens=125, rolling_ended=True)
        assert 'rolling_contexts' not in other

    def test_empty_generated_payload_falls_back_to_complete_saved_tensor(self):
        mm = {'codes': {'audio': torch.tensor([[17, 2, 3, 4]])}}
        self.model.postprocess(
            torch.zeros(1, 4), mm, context_tail_tokens=125, rolling_voice='voice', rolling_request_key='a'
        )
        assert mm['rolling_contexts']['a'] is self.model.original

    def test_text_only_continuation_is_direct_generated_audio(self):
        self.model.rolling_original_has_audio['voice'] = False
        self.model.original = self.model.original[:2]
        mm = {'codes': {'audio': torch.tensor([[2, 17, 3, 4]])}}
        self.model.postprocess(
            torch.zeros(1, 4),
            mm,
            context_tail_tokens=125,
            rolling_tail=torch.ones(2, 124, dtype=torch.long),
            rolling_voice='voice',
            rolling_request_key='a',
        )
        tail = torch.cat((torch.ones(2, 124, dtype=torch.long), torch.tensor([[2], [3]])), dim=-1)
        expected = context_codes(tail, bos_id=18, eos_id=19, stacking_factor=2).T.float()
        torch.testing.assert_close(mm['rolling_contexts']['a'], expected)
        assert expected.shape == (64, 4)

    def test_continuation_context_is_placed_before_unchanged_style_text(self):
        model = self.model
        model._combined_embeddings = torch.zeros(1, 4)
        model._prefill_cache = {}
        model.task_embedding = None
        model.text_embedding = torch.nn.Embedding(3, 4)
        model.text_embedding.weight.data.fill_(3.0)
        model._encode_context_text = lambda text, device: torch.tensor([1, 2])
        model._build_text_prefill_embeds = lambda *args: None
        generated = torch.full((11, 4), 5.0)
        output = model._build_prefill_embeds(
            torch.device('cpu'), dict(rolling_voice='voice', context_text='style', speaker_embedding=generated)
        )
        torch.testing.assert_close(output, torch.cat((generated, torch.full((2, 4), 3.0))))
        model._maybe_set_lt_sampling_params = lambda info: None
        model.arch.condition_on_user_speech = False
        model._encode_text_stream = lambda text: [1, 2]
        inputs = torch.zeros(13, dtype=torch.long)
        (_, prefill, update) = model.preprocess(
            inputs,
            None,
            speaker_embedding=generated,
            context_text='style',
            text='next',
            context_tail_tokens=125,
            rolling_voice='voice',
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=13,
        )
        torch.testing.assert_close(prefill, output)
        assert update['text_tokens'] == [1, 2]
        assert model._prefill_cache == {}

    def test_metadata_is_validated_without_loading_additional_weights(self, tmp_path):
        self.model.model_path = str(tmp_path)
        self.model._initialize_rolling_context()
        assert not self.model.rolling_enabled
        root = tmp_path / 'rolling_context'
        root.mkdir()
        self.model.original.add_(0.0001)
        config = dict(
            schema_version=5,
            embedding_dim=4,
            original_has_audio={'voice': True},
            initial_contexts={
                'voice': dict(rows=96, sha256=hashlib.sha256(self.model.original.numpy().tobytes()).hexdigest())
            },
            max_generated_frames=125,
            total_frames=250,
            frames_per_second=25.0,
            total_duration_seconds=10.0,
        )
        (root / 'config.json').write_text(json.dumps(config))
        self.model._initialize_rolling_context()
        assert self.model.rolling_enabled
        assert not hasattr(self.model, 'rolling_encoder')
        original = self.model.original
        for dtype in (torch.float16, torch.bfloat16):
            self.model.original = original.to(dtype)
            self.model._initialize_rolling_context()
            assert self.model.rolling_enabled
            assert self.model.original.dtype == dtype
        self.model.original = original
        config['initial_contexts']['voice']['sha256'] = 'changed'
        (root / 'config.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='complete saved context'):
            self.model._initialize_rolling_context()
        config['schema_version'] = 3
        (root / 'config.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='schema 5'):
            self.model._initialize_rolling_context()


def _speaker_model(model_path, embedding_dim=3):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_path = str(model_path)
    model.embedding_dim = embedding_dim
    model._combined_embeddings = torch.zeros(1, embedding_dim)
    model._speaker_embedding_buffers = {}
    return model


def test_context_bundle_is_preloaded_as_nonpersistent_buffers(tmp_path):
    speaker_dir = tmp_path / "speaker_embeddings"
    speaker_dir.mkdir()
    registry = {
        "schema_version": 1,
        "contexts": {
            "eng": {"tensor_key": "speaker.eng", "context_text": "[EN]", "prompt_len": 67},
            "deu": {"tensor_key": "speaker.deu", "context_text": "[DE]", "prompt_len": 69},
        },
    }
    (speaker_dir / "contexts.json").write_text(json.dumps(registry))
    tensors = {"speaker.eng": torch.ones(2, 3), "speaker.deu": torch.full((4, 3), 2.0)}
    save_file(tensors, speaker_dir / "contexts.safetensors")
    model = _speaker_model(tmp_path)

    model._preload_known_speaker_embeddings()

    assert sorted(model._speaker_embedding_buffers) == ["deu", "eng"]
    torch.testing.assert_close(
        model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float32), tensors["speaker.eng"]
    )
    assert not any(name.startswith("_speaker_context_") for name in model.state_dict())

    assert EasyMagpieTTSForConditionalGeneration.get_prompt_len("eng", str(tmp_path), tokenize=list) == 67
    with pytest.raises(FileNotFoundError, match="speaker_id 'missing'"):
        EasyMagpieTTSForConditionalGeneration.get_prompt_len("missing", str(tmp_path), tokenize=list)

    assert EasyMagpieTTSForConditionalGeneration.get_prompt_len(
        "eng", str(tmp_path), tokenize=list, context_text="longer style"
    ) == 67 - len("[EN]") + len("longer style")
