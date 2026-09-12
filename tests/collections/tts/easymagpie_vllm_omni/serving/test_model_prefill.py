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
from unittest.mock import patch

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from torch import nn


@pytest.mark.parametrize("offset,previous_codes", [(0, None), (2, None), (3, None), (3, (8, 9))])
def test_decode_scalar_updates_do_not_copy_host_tensors(offset, previous_codes):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(audio_bos_id=1025)
    model.has_phoneme = False
    model.speech_delay = 2
    model.embedding_dim = 3
    model.num_codebooks = 2
    model._combined_embeddings = torch.zeros(3, 3)
    model._dec_text_tokens = torch.full((3,), -1, dtype=torch.long)
    model._dec_text_mask = torch.full((3,), -1, dtype=torch.long)
    model._dec_audio_valid = torch.full((3,), -1, dtype=torch.long)
    model._dec_audio_codes = torch.full((3, 2), -1, dtype=torch.long)
    input_ids = torch.zeros(1, dtype=torch.long)
    info = {"decode_offset": offset, "text_tokens": [7, 11, 13]}
    if previous_codes is not None:
        info["last_audio_codes"] = torch.tensor(previous_codes)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        ids, embeds, update = model._preprocess_decode(input_ids, 1, input_ids.device, info)

    assert ids is input_ids
    torch.testing.assert_close(embeds, torch.zeros(1, 3))
    assert update == {"decode_offset": offset + 1}
    assert model._dec_text_tokens.tolist() == [-1, info["text_tokens"][offset] if offset < 3 else -1, -1]
    assert model._dec_text_mask.tolist() == [-1, int(offset < 3), -1]
    assert model._dec_audio_valid.tolist() == [-1, int(offset >= model.speech_delay), -1]
    expected_codes = list(previous_codes) if previous_codes is not None else [1025, 1025]
    assert model._dec_audio_codes.tolist() == [[-1, -1], expected_codes if offset >= 2 else [-1, -1], [-1, -1]]
    copies = sum(event.count for event in profile.key_averages() if event.key == "aten::copy_")
    assert copies == int(previous_codes is not None)  # Only existing tensor feedback needs a copy.


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


def test_batch_target_prefill_public_hook_matches_scalar_and_consumes_once():
    model = _target_prefill_model()
    infos = {
        key: {"speaker_id": "eng", "text_prefill_num": 4, "prefill_text_tokens": ids}
        for key, ids in (("a", [8, 9, 10, 11]), ("b", [12, 13]))
    }
    inputs = torch.zeros(7, dtype=torch.long)
    expected = {key: model.preprocess(inputs, None, request_id=key, **info)[1] for key, info in infos.items()}

    model.preprocess_batch(["b", "a"], infos, inputs.device)

    assert set(model._batch_text_prefill) == {"a", "b"}
    for key in ("a", "b"):
        _, actual, update = model.preprocess(inputs, None, request_id=key, **infos[key])
        assert torch.equal(actual, expected[key])
        assert update == {"prefill_offset": 7, "decode_offset": 4}
        assert key not in model._batch_text_prefill
    assert set(infos["a"]) == {"speaker_id", "text_prefill_num", "prefill_text_tokens"}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stacking", [0, 1, 2])
@pytest.mark.parametrize("delay", [0, 3])
@pytest.mark.parametrize(
    "dtypes",
    [
        (a, b)
        for a in (torch.float32, torch.float16, torch.bfloat16)
        for b in (torch.float32, torch.float16, torch.bfloat16)
    ],
)
def test_batch_target_prefill_exact_values_and_one_embedding_lookup(device, stacking, delay, dtypes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _target_prefill_model(device, *dtypes, stacking, delay)
    infos = {
        str(i): {"text_prefill_num": 4, "prefill_text_tokens": list(range(8, 8 + length)), "speaker_id": "eng"}
        for i, length in enumerate((4, 0, 2, 4, 1, 3))
    }
    inputs = torch.zeros(7, device=device, dtype=torch.long)
    expected = {key: model.preprocess(inputs, None, **info)[1] for key, info in infos.items()}
    with (
        patch.object(model.text_embedding, "forward", wraps=model.text_embedding.forward) as embed,
        patch.object(model, "_embed_phoneme", wraps=model._embed_phoneme) as phoneme,
    ):
        model.preprocess_batch(["3", "0", "5", "2", "1", "4"], infos, inputs.device)
        for key, info in infos.items():
            _, actual, _ = model.preprocess(inputs, None, request_id=key, **info)
            assert torch.equal(actual, expected[key])
            assert torch.equal(actual.contiguous().view(torch.uint8), expected[key].contiguous().view(torch.uint8))
        assert embed.call_count == 1
        assert phoneme.call_count == int(bool(stacking))
    assert model._batch_text_prefill == {}


def test_batch_target_prefill_owns_rows_and_drops_previous_ids():
    model = _target_prefill_model()
    info = {"text_prefill_num": 4, "prefill_text_tokens": [8, 9]}
    model.preprocess_batch(["a", "b"], {"a": info, "b": info}, torch.device("cpu"))
    old = model._batch_text_prefill["a"][2]
    other = model._batch_text_prefill["b"][2].clone()
    old.zero_()
    assert torch.equal(model._batch_text_prefill["b"][2], other)
    model.preprocess_batch(["a"], {"a": dict(info, prefill_text_tokens=[10])}, torch.device("cpu"))
    assert set(model._batch_text_prefill) == {"a"}
    assert model._batch_text_prefill["a"][2].untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
    assert not old.any()
    model.preprocess_batch([], {}, torch.device("cpu"))
    assert model._batch_text_prefill == {}


def test_batch_target_prefill_nested_metadata_uses_actual_id_and_top_level_values():
    model = _target_prefill_model()
    info = {
        "request_id": "forged",
        "text_prefill_num": 4,
        "prefill_text_tokens": [8, 9],
        "additional_information": {"text_prefill_num": 3, "prefill_text_tokens": [31], "speaker_id": "eng"},
        "text_tokens": [30] * 50,
        "_batch_text_prefill": torch.full((4, 7), 999.0),
    }
    inputs = torch.zeros(7, dtype=torch.long)
    expected = model.preprocess(inputs, None, **info)[1]
    model.preprocess_batch(["actual"], {"actual": info}, inputs.device)
    assert set(model._batch_text_prefill) == {"actual"}
    info["request_id"] = "actual"  # The upstream per-request loop installs the authoritative ID.
    assert torch.equal(model.preprocess(inputs, None, **info)[1], expected)
    assert model._batch_text_prefill == {}
    assert set(info) == {
        "request_id",
        "text_prefill_num",
        "prefill_text_tokens",
        "additional_information",
        "text_tokens",
        "_batch_text_prefill",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"prefill_offset": 1},
        {"prefill_offset": -1},
        {"prefill_offset": "0"},
        {"prefill_offset": torch.tensor(0)},
        {"text_prefill_num": 0},
        {"text_prefill_num": "4"},
        {"text_prefill_num": 3},
        {"prefill_text_tokens": torch.tensor([8])},
        {"prefill_text_tokens": [8.0]},
        {"prefill_text_tokens": [-1]},
        {"prefill_text_tokens": [32]},
        {"prefill_text_tokens": [8] * 5},
    ],
)
def test_batch_target_prefill_skips_unsupported_metadata_without_changing_decode(change):
    model = _target_prefill_model(stacking=0)
    info = {"request_id": "a", "text_prefill_num": 4, "prefill_text_tokens": [8], **change}
    model.preprocess_batch(["a", "missing"], {"a": info}, torch.device("cpu"))
    assert model._batch_text_prefill == {}
    _, _, update = model.preprocess(torch.zeros(1, dtype=torch.long), None, **info)
    assert update == {"decode_offset": 1}


@pytest.mark.parametrize("change", ["prefix", "count", "offset", "dtype", "device", "request_id"])
def test_batch_target_prefill_rechecks_before_consuming(change):
    model = _target_prefill_model()
    info = {"request_id": "a", "speaker_id": "eng", "text_prefill_num": 4, "prefill_text_tokens": [8, 9]}
    model.preprocess_batch(["a"], {"a": info}, torch.device("cpu"))
    count, prefix, target = model._batch_text_prefill["a"]
    target.fill_(9999)
    if change == "prefix":
        info["prefill_text_tokens"][0] = 12
    elif change == "count":
        info["text_prefill_num"] = "4"  # Valid scalar input, intentionally ineligible for batching.
    elif change == "offset":
        info["prefill_offset"] = 2
    elif change == "request_id":
        info["request_id"] = "other"
    else:
        target = target.to(torch.float16) if change == "dtype" else target.to("meta")
        model._batch_text_prefill["a"] = (count, prefix, target)
    inputs = torch.zeros(2 if change == "offset" else 7, dtype=torch.long)
    _, actual, _ = model.preprocess(inputs, None, **info)
    model._batch_text_prefill = {}
    _, expected, _ = model.preprocess(inputs, None, **info)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("custom_voice", [False, True])
def test_batch_target_prefill_partial_chunks_keep_conditioning_and_offset(custom_voice):
    model = _target_prefill_model()
    info = {
        "request_id": "a",
        "text_prefill_num": 4,
        "prefill_text_tokens": [8, 9],
        "text_token": [8],
        "text_token_start": 0,
    }
    info.update(speaker_embedding=torch.full((2, 7), 2.0)) if custom_voice else info.update(speaker_id="eng")
    full = model._build_prefill_embeds(torch.device("cpu"), info)
    model.preprocess_batch(["a"], {"a": info}, torch.device("cpu"))
    pieces = []
    for size in (2, 2, 3):
        _, part, update = model.preprocess(torch.zeros(size, dtype=torch.long), None, **info)
        pieces.append(part)
        info.update(update)
        model.preprocess_batch(["a"], {"a": info}, torch.device("cpu"))
        assert model._batch_text_prefill == {}
    assert torch.equal(torch.cat(pieces), full)
    assert info["prefill_offset"] == 7 and info["text_tokens"] == [8]
    assert len(model._prefill_cache) == int(not custom_voice)


def test_batch_target_prefill_is_unused_by_single_token_streaming_decode():
    model = _target_prefill_model(stacking=0)
    info = {
        "request_id": "a",
        "text_prefill_num": 4,
        "prefill_text_tokens": [8],
        "text_tokens": [8],
        "text_token": [9],
        "text_token_start": 1,
        "_omni_is_prefill": True,
    }
    model.preprocess_batch(["a"], {"a": info}, torch.device("cpu"))
    target = model._batch_text_prefill["a"][2]
    target.fill_(9999)
    inputs = torch.zeros(8, dtype=torch.long)
    _, embeds, update = model.preprocess(inputs[5:6], None, start=0, **info)
    assert torch.equal(embeds, torch.zeros(1, 7))
    assert update == {"decode_offset": 1, "text_tokens": [8, 9]}
    assert model._dec_text_tokens[5] == 8
    assert model._batch_text_prefill["a"][2] is target
    info.update(update, prefill_offset=7)
    model.preprocess_batch(["a"], {"a": info}, inputs.device)
    assert model._batch_text_prefill == {}


@pytest.mark.parametrize("prefix", [None, []])
def test_batch_target_prefill_all_empty_prefixes_and_absent_counts(prefix):
    model = _target_prefill_model()
    info = {"text_prefill_num": 4, "prefill_text_tokens": prefix, "prefill_offset": None}
    expected = model._build_text_prefill_embeds(torch.device("cpu"), torch.float32, info)
    model.preprocess_batch(["a", "empty"], {"a": info, "empty": {}}, torch.device("cpu"))
    actual = model._build_text_prefill_embeds(torch.device("cpu"), torch.float32, dict(info, request_id="a"))
    assert torch.equal(actual, expected)
    assert model._batch_text_prefill == {}


@pytest.mark.parametrize("change", [{"text_prefill_num": 3}, {"prefill_text_tokens": [8] * 5}])
def test_batch_target_prefill_preserves_scalar_validation_errors(change):
    model = _target_prefill_model()
    info = {"text_prefill_num": 4, "prefill_text_tokens": [8], **change}
    with pytest.raises(AssertionError) as scalar:
        model._build_text_prefill_embeds(torch.device("cpu"), torch.float32, info)
    model.preprocess_batch(["a"], {"a": info}, torch.device("cpu"))
    with pytest.raises(AssertionError) as batched:
        model._build_text_prefill_embeds(torch.device("cpu"), torch.float32, dict(info, request_id="a"))
    assert str(scalar.value) == str(batched.value)


def test_batch_decode_prepares_phoneme_rhs_and_zeros_once():
    model = _target_prefill_model(stacking=2)
    infos = {
        key: {
            "decode_offset": 12,
            "text_tokens": [8] * 15,
            "last_phoneme_token": torch.tensor([[3, model.phoneme_eos_id]]),
            "phoneme_ended": torch.tensor(ended),
            "last_audio_codes": torch.tensor([[4, 5]]),
        }
        for key, ended in (("a", False), ("b", True))
    }
    inputs = torch.zeros(9, dtype=torch.long)
    model.preprocess_batch(["b", "a"], infos, inputs.device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        outputs = [
            model.preprocess(inputs[start : start + 1], None, request_id=key, **infos[key])
            for key, start in (("a", 7), ("b", 2))
        ]
    names = {event.key for event in profile.key_averages()}
    assert "aten::eq" not in names
    assert "aten::any" not in names
    assert "aten::zeros" not in names
    assert model._dec_phoneme_valid[[7, 2]].tolist() == [1, 0]
    assert all(result[2]["phoneme_ended"].item() for result in outputs)
    outputs[0][1].fill_(99)
    assert not outputs[1][1].any()
    assert model._batch_decode == {}


def test_batch_decode_preparation_is_fresh_and_consumed_once():
    model = _target_prefill_model()
    info = {"decode_offset": 8, "last_phoneme_token": torch.tensor([3]), "phoneme_ended": torch.tensor(False)}
    inputs = torch.zeros(1, dtype=torch.long)
    model.preprocess_batch(["a", "unused"], {"a": info, "unused": info}, inputs.device)
    _, old, _ = model.preprocess(inputs, None, request_id="a", **info)
    old.fill_(9)
    _, fresh, _ = model.preprocess(inputs, None, request_id="a", **info)
    assert not fresh.any()
    model.preprocess_batch(["a"], {"a": info}, inputs.device)
    assert set(model._batch_decode) == {"a"}
    _, new, _ = model.preprocess(inputs, None, request_id="a", **info)
    assert new.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
    assert not new.any()
    model.preprocess_batch([], {}, inputs.device)
    assert model._batch_decode == {}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stacking", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batch_decode_matches_scalar_across_delays_eos_partial_feedback_and_reordering(device, stacking, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _target_prefill_model(device=device, output_dtype=dtype, stacking=stacking)
    scalar = _target_prefill_model(device=device, output_dtype=dtype, stacking=stacking)
    fields = [name for name in vars(model) if name.startswith("_dec_")]
    for candidate in (model, scalar):
        for name in fields:
            getattr(candidate, name).fill_(-17)
    infos = {}
    for i, offset in enumerate((0, 3, 4, 10, 12, 12, 12, 12)):
        phoneme = torch.full((1, stacking), 3, device=device, dtype=torch.long)
        if stacking and i == 5:
            phoneme[0, -1] = model.phoneme_eos_id
        if i == 4:
            phoneme = phoneme[:, : max(0, stacking - 1)]
        infos[str(i)] = {
            "decode_offset": offset,
            "text_tokens": [8] * 15,
            "text_token": [9],
            "text_token_start": 15,
            "last_phoneme_token": phoneme,
            "phoneme_ended": torch.tensor(i == 6, device=device),
            "last_audio_codes": torch.tensor([[4, 5]], device=device),
        }
    infos["6"]["last_audio_codes"] = torch.tensor([6], device=device)
    infos["7"]["last_audio_codes"] = None
    infos["7"]["phoneme_ended"] = False  # Preserve the ordinary scalar conversion path.
    inputs = torch.zeros(32, device=device, dtype=torch.long)
    slots = (17, 1, 9, 3, 14, 0, 7, 5)
    for order in (list(infos), list(reversed(infos))):
        model.preprocess_batch(order, infos, inputs.device)
        for key in reversed(order):
            view = inputs[slots[int(key)] : slots[int(key)] + 1]
            actual = model.preprocess(view, None, request_id=key, **infos[key])
            expected = scalar.preprocess(view, None, request_id=key, **infos[key])
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))
            _assert_decode_updates_equal(actual[2], expected[2])
            infos[key].update(actual[2])
        for name in fields:
            assert torch.equal(getattr(model, name), getattr(scalar, name)), name
        assert model._batch_decode == {}


@pytest.mark.parametrize("change", ["offset", "phoneme", "ended", "phoneme_dtype", "ended_dtype", "eos", "id"])
def test_batch_decode_rechecks_inputs_and_falls_back(change):
    model = _target_prefill_model(stacking=2)
    info = {
        "decode_offset": 12,
        "last_phoneme_token": torch.tensor([[3, model.phoneme_eos_id]]),
        "phoneme_ended": torch.tensor(False),
    }
    inputs = torch.zeros(1, dtype=torch.long)
    model.preprocess_batch(["a"], {"a": info}, inputs.device)
    request_id = "a"
    if change == "offset":
        info["decode_offset"] += 1
    elif change == "phoneme":
        info["last_phoneme_token"] = torch.tensor([[3, 3]])
    elif change == "ended":
        info["phoneme_ended"] = torch.tensor(True)
    elif change == "phoneme_dtype":
        info["last_phoneme_token"] = info["last_phoneme_token"].float()
    elif change == "ended_dtype":
        info["phoneme_ended"] = info["phoneme_ended"].long()
    elif change == "eos":
        model.phoneme_eos_id = 13
    else:
        request_id = "other"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = model.preprocess(inputs, None, request_id=request_id, **info)
    assert any(event.key == "aten::eq" for event in profile.key_averages())
    model._batch_decode = {}
    expected = model.preprocess(inputs, None, **info)
    _assert_decode_updates_equal(actual[2], expected[2])
    assert torch.equal(actual[1], expected[1])


def test_batch_decode_nested_metadata_and_actual_id_preserve_streaming_merge():
    model = _target_prefill_model(stacking=2)
    info = {
        "request_id": "forged",
        "decode_offset": 4,
        "last_phoneme_token": torch.tensor([[3, model.phoneme_eos_id]]),
        "phoneme_ended": torch.tensor(False),
        "text_tokens": [8, 9, 10, 11],
        "text_token": [12, 13],
        "text_token_start": 4,
        "additional_information": {"decode_offset": 0, "phoneme_ended": True, "text_token": [31]},
        "_batch_decode": torch.ones(3),
    }
    inputs = torch.zeros(8, dtype=torch.long)
    model.preprocess_batch(["actual"], {"actual": info}, inputs.device)
    assert set(model._batch_decode) == {"actual"}
    info["request_id"] = "actual"  # Pinned runner supplies this after the public hook.
    _, _, first = model.preprocess(inputs[6:7], None, **info)
    assert first["text_tokens"] == [8, 9, 10, 11, 12, 13]
    assert model._dec_text_tokens[6] == 12
    assert model._dec_phoneme_valid[6] == 1 and first["phoneme_ended"]
    info.update(first)
    model.preprocess_batch(["actual"], {"actual": info}, inputs.device)
    _, _, second = model.preprocess(inputs[2:3], None, **info)
    assert "text_tokens" not in second  # The repeated absolute-position chunk is not appended twice.
    assert model._dec_text_tokens[2] == 13 and model._dec_phoneme_valid[2] == 0
    info.update(second, text_token=[31], text_token_start=4)
    model.preprocess_batch(["actual"], {"actual": info}, inputs.device)
    with pytest.raises(ValueError, match="Conflicting streaming text chunk"):
        model.preprocess(inputs[4:5], None, **info)


def test_batch_decode_preparation_does_not_choose_phase_or_consume_prefill():
    model = _target_prefill_model()
    info = {"speaker_id": "eng", "text_prefill_num": 4, "prefill_text_tokens": [8], "decode_offset": 12}
    inputs = torch.zeros(7, dtype=torch.long)
    expected = model.preprocess(inputs, None, **info)
    model.preprocess_batch(["a"], {"a": info}, inputs.device)
    zero, state = model._batch_decode["a"]
    zero.fill_(999)
    actual = model.preprocess(inputs, None, request_id="a", **info)
    assert torch.equal(actual[1], expected[1])
    assert "a" in model._batch_decode
    model.preprocess(inputs[:0], inputs[:0].reshape(0, 1), request_id="a", **info)
    assert "a" in model._batch_decode


@pytest.mark.parametrize("change", ["dtype", "device", "shape"])
def test_batch_decode_zero_rows_recheck_output_contract(change):
    model = _target_prefill_model(stacking=0)
    inputs = torch.zeros(1, dtype=torch.long)
    model.preprocess_batch(["a"], {"a": {}}, inputs.device)
    zero, state = model._batch_decode["a"]
    zero.fill_(999)
    zero = zero.double() if change == "dtype" else zero.to("meta") if change == "device" else zero[:, :2]
    model._batch_decode["a"] = (zero, state)
    _, output, _ = model.preprocess(inputs, None, request_id="a")
    assert output.shape == (1, 7) and output.dtype == torch.float32 and output.device == inputs.device
    assert not output.any()


def test_batch_decode_uses_pinned_runner_hook_with_mixed_rows_and_owning_updates(monkeypatch):
    import numpy as np
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm_omni.worker import gpu_model_runner

    model = _target_prefill_model()
    runner = EasyMagpieGPUARModelRunner.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    runner.supports_mm_inputs = runner.enable_prompt_embeds = runner.uses_mrope = runner.has_talker_mtp = False
    runner.uses_xdrope_dim = 0
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(8, dtype=torch.long))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(8, 7))
    runner.positions = torch.arange(8)
    runner._init_model_kwargs = lambda: {}
    runner.requests = {}
    runner.model_intermediate_buffer = {
        "decode": {
            "request_id": "forged",
            "decode_offset": 12,
            "last_phoneme_token": torch.tensor([[model.phoneme_eos_id]]),
            "phoneme_ended": torch.tensor(False),
        }
    }
    monkeypatch.setattr(gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    prepared = []
    original = model.preprocess_batch

    def record_preparation(**kwargs):
        original(**kwargs)
        prepared.append(model._batch_decode["decode"])

    monkeypatch.setattr(model, "preprocess_batch", record_preparation)
    for prefill_id, order, counts, offsets, computed in (
        ("first", ["first", "decode"], [7, 1], [0, 7, 8], [0, 7]),
        ("fresh", ["decode", "fresh"], [1, 7], [0, 1, 8], [8, 0]),
    ):
        runner.model_intermediate_buffer[prefill_id] = {
            "speaker_id": "eng",
            "text_prefill_num": 4,
            "prefill_text_tokens": [8],
        }
        runner.input_batch = SimpleNamespace(req_ids=order, num_computed_tokens_cpu=np.array(computed))
        runner.query_start_loc = SimpleNamespace(cpu=np.array(offsets))
        runner.requests.update({key: SimpleNamespace(prompt_token_ids=[0] * 7) for key in order})
        scheduler = SimpleNamespace(
            total_num_scheduled_tokens=8,
            num_scheduled_tokens=dict(zip(order, counts)),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(additional_information={}),
        )
        _, embeddings, *_ = runner._preprocess(scheduler, 8)
        slot = offsets[order.index("decode")]
        assert not embeddings[slot].any()
        assert model._dec_phoneme_valid[slot] == int(prefill_id == "first")
        state = runner.model_intermediate_buffer["decode"]
        assert state["request_id"] == "decode"
        assert state["phoneme_ended"]
        saved_ended = prepared[-1][1][2]
        assert state["phoneme_ended"].untyped_storage().data_ptr() != saved_ended.untyped_storage().data_ptr()
        saved_ended.fill_(False)
        assert state["phoneme_ended"]  # Immediate runner storage still owns its update.
        assert "decode" not in model._batch_decode
        runner.model_intermediate_buffer["decode"]["last_phoneme_token"] = torch.tensor([[3]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("count", [1, 32, 64])
def test_pinned_preprocess_batches_owning_ended_copies(monkeypatch, count, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _target_prefill_model(device=device, stacking=2)
    runner, scheduler = _decode_preprocess_runner(monkeypatch, model, count, device)
    original = model.preprocess_batch
    prepared = []

    def record_preparation(**kwargs):
        original(**kwargs)
        prepared.extend(entry[1][2] for entry in model._batch_decode.values())

    monkeypatch.setattr(model, "preprocess_batch", record_preparation)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        _, embeddings, *_ = runner._preprocess(scheduler, count)
    clones = sum(event.count for event in profile.key_averages() if event.key == "aten::clone")
    assert clones == 0, f"Per-request ownership still launches {clones} clones"
    assert not embeddings.any()
    for index, source in enumerate(prepared):
        cached = runner.model_intermediate_buffer[str(index)]
        assert cached["decode_offset"] == 13
        assert cached["phoneme_ended"].shape == () and cached["phoneme_ended"].dtype == torch.bool
        assert cached["phoneme_ended"].untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        source.fill_(False)
        assert cached["phoneme_ended"]
        assert runner.requests[str(index)].additional_information_cpu is cached
    runner.model_intermediate_buffer["0"]["phoneme_ended"].fill_(False)
    assert all(runner.model_intermediate_buffer[str(i)]["phoneme_ended"] for i in range(1, count))


@pytest.mark.parametrize("failure", [None, "hook", "scalar", "packing"])
def test_pinned_preprocess_copy_scope_preserves_metadata_order_and_failures(monkeypatch, failure):
    model = _target_prefill_model(stacking=2)
    runner, scheduler = _decode_preprocess_runner(monkeypatch, model, 3, "cpu")
    incoming = torch.tensor(True)
    scheduler.scheduled_new_reqs = [
        SimpleNamespace(req_id="0", model_intermediate_buffer={"phoneme_ended": incoming}, additional_information=None)
    ]
    original_hook, original_scalar = model.preprocess_batch, model.preprocess
    calls, prepared = [], []

    def hook(**kwargs):
        flag = runner.model_intermediate_buffer["0"]["phoneme_ended"]
        assert flag and flag.data_ptr() != incoming.data_ptr()
        incoming.fill_(False)
        assert flag  # Incoming metadata was copied before the hook.
        original_hook(**kwargs)
        prepared.extend(entry[1][2] for entry in model._batch_decode.values())
        if failure == "hook":
            raise ValueError("hook failed")

    def scalar(*args, **kwargs):
        key = kwargs["request_id"]
        if calls:
            assert runner.model_intermediate_buffer[calls[-1]]["decode_offset"] == 13
        if failure == "scalar" and key == "1":
            raise ValueError("scalar failed")
        result = original_scalar(*args, **kwargs)
        calls.append(key)
        return result

    concatenate = torch.cat

    def pack(*args, **kwargs):
        if failure == "packing" and len(calls) == 3:
            raise ValueError("packing failed")
        return concatenate(*args, **kwargs)

    monkeypatch.setattr(model, "preprocess_batch", hook)
    monkeypatch.setattr(model, "preprocess", scalar)
    monkeypatch.setattr(torch, "cat", pack)
    if failure is None:
        runner._preprocess(scheduler, 3)
    else:
        with pytest.raises(ValueError, match=f"{failure} failed"):
            runner._preprocess(scheduler, 3)
    assert runner._feedback_copy_context.pending is None
    assert runner._feedback_copy_context.phase is None
    for flag in prepared:
        flag.fill_(False)
    expected_done = {"0", "1", "2"} if failure is None else ({"0"} if failure == "scalar" else set())
    for key, info in runner.model_intermediate_buffer.items():
        assert bool(info["phoneme_ended"]) == (key == "0" or key in expected_done)
        assert info["decode_offset"] == (13 if key in calls else 12)
    # A direct hook after the scope must not enable deferred ordinary stores.
    monkeypatch.setattr(model, "preprocess_batch", lambda **kwargs: None)
    runner._maybe_run_batch_preprocess([], torch.device("cpu"))
    value = torch.tensor(True)
    runner._update_intermediate_buffer("2", {"phoneme_ended": value})
    value.fill_(False)
    assert runner.model_intermediate_buffer["2"]["phoneme_ended"]


def _decode_preprocess_runner(monkeypatch, model, count, device):
    import numpy as np
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm_omni.worker import gpu_model_runner

    runner = EasyMagpieGPUARModelRunner.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=True))
    runner.supports_mm_inputs = runner.enable_prompt_embeds = runner.uses_mrope = runner.has_talker_mtp = False
    runner.uses_xdrope_dim = 0
    runner.input_ids = SimpleNamespace(gpu=torch.zeros(count, dtype=torch.long, device=device))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(count, 7, device=device))
    runner.positions = torch.arange(count, device=device)
    runner._init_model_kwargs = lambda: {}
    ids = [str(index) for index in range(count)]
    runner.requests = {key: SimpleNamespace(prompt_token_ids=[0]) for key in ids}
    runner.input_batch = SimpleNamespace(req_ids=ids, num_computed_tokens_cpu=np.full(count, 13))
    runner.query_start_loc = SimpleNamespace(cpu=np.arange(count + 1))
    runner.model_intermediate_buffer = {
        key: {
            "decode_offset": 12,
            "phoneme_ended": torch.tensor(False, device=device),
            "last_phoneme_token": torch.tensor([[3, model.phoneme_eos_id]], device=device),
        }
        for key in ids
    }
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=count,
        num_scheduled_tokens=dict.fromkeys(ids, 1),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(additional_information={}),
    )
    monkeypatch.setattr(gpu_model_runner, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    return runner, scheduler


def _assert_decode_updates_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def _target_prefill_model(device="cpu", source_dtype=torch.float32, output_dtype=torch.float32, stacking=1, delay=3):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=stacking, audio_bos_id=15)
    model.embedding_dim = 7
    model.num_codebooks = 2
    model.has_phoneme = bool(stacking)
    model.phonemes_delay = delay
    model.phoneme_bos_id = 7
    model.phoneme_eos_id = 14
    model.speech_delay = 10
    model.text_embedding = nn.Embedding(32, 7, device=device, dtype=source_dtype)
    model.phoneme_embeddings = nn.ModuleList(
        nn.Embedding(16, 7, device=device, dtype=source_dtype) for _ in range(stacking)
    )
    model.task_embedding = None
    model._prefill_cache = {}
    model._combined_embeddings = torch.zeros(128, 7, device=device, dtype=output_dtype)
    model._dec_text_tokens = torch.zeros(128, device=device, dtype=torch.long)
    model._dec_text_mask = torch.zeros(128, device=device, dtype=torch.long)
    model._dec_audio_codes = torch.zeros(128, 2, device=device, dtype=torch.long)
    model._dec_audio_valid = torch.zeros(128, device=device, dtype=torch.long)
    model._dec_phoneme_tokens = torch.zeros(128, stacking, device=device, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(128, device=device, dtype=torch.long)
    model._load_known_speaker_embedding = lambda speaker, device, dtype: torch.ones(2, 7, device=device, dtype=dtype)
    model._encode_context_text = lambda text, device: torch.tensor([2], device=device)
    return model
