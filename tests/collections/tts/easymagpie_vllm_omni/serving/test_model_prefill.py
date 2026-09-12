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
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
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
