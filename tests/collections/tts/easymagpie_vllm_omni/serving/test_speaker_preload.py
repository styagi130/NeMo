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
"""Known .pt voices are optional startup state, never a new format contract."""

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from easymagpie_vllm_omni import easymagpie
from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls
from torch import nn


def _fixture(tmp_path, value=None):
    (tmp_path / "speaker_embeddings").mkdir(exist_ok=True)
    config = dict(
        hidden_dim=4,
        embedding_dim=4,
        audio_embedding_dim=4,
        text_vocab_size=32,
        phoneme_vocab_size=0,
        phoneme_stacking_factor=0,
        num_audio_codebooks=2,
        codebook_size=8,
        frame_stacking_factor=1,
        local_transformer_hidden_dim=4,
        local_transformer_n_heads=1,
        local_transformer_n_layers=1,
        num_task_embeddings=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(
        torch.arange(8, dtype=torch.float32).view(2, 4) if value is None else value,
        tmp_path / "speaker_embeddings/eng.pt",
    )
    return config


def _model(tmp_path, monkeypatch, dtype=torch.float16):
    for name in (
        "NemotronHModel",
        "EasyMagpieCodePredictor",
        "patch_shared_expert_activation",
        "patch_moe_routed_scale",
        "patch_mamba_streaming_decode",
    ):
        monkeypatch.setattr(easymagpie, name, lambda *args, **kwargs: nn.Module())
    monkeypatch.setattr(easymagpie, "set_model_tag", lambda *_args: nullcontext())
    config = json.loads((tmp_path / "config.json").read_text())
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**config), model=str(tmp_path), dtype=dtype),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    model = easymagpie.EasyMagpieTTSForConditionalGeneration(vllm_config=vllm_config)
    model._text_tokenizer = SimpleNamespace(encode_context=lambda text: [1, 2, 3] if text == "[EN]" else [4])
    return model


def _adapter(tmp_path):
    context = SimpleNamespace(engine_client=SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path))))
    adapter = _build_adapter_cls()(context)
    adapter._tokenizer = SimpleNamespace(encode_context=lambda text: [1, 2, 3] if text == "[EN]" else [4])
    return adapter


def test_model_startup_preloads_voice_before_any_context_miss(tmp_path, monkeypatch):
    _fixture(tmp_path)
    with patch.object(torch, "load", wraps=torch.load) as load:
        model = _model(tmp_path, monkeypatch)
        assert load.call_count == 1
        assert load.call_args.kwargs["weights_only"] is True
        assert load.call_args.kwargs["map_location"] == "cpu"
    with patch.object(torch, "load", side_effect=AssertionError("request-time voice read")):
        for context in ("[EN]", "different"):
            for mode in (0, 1):
                result = model._build_prefill_embeds(
                    torch.device("cpu"), {"speaker_id": "eng", "context_text": context, "task_mode_id": mode}
                )
                assert result.shape == (1 + 2 + (3 if context == "[EN]" else 1), 4)
    assert len(model._prefill_cache) == 4


def test_adapter_startup_keeps_frames_without_request_time_tensor_load(tmp_path):
    _fixture(tmp_path)
    with patch.object(torch, "load", wraps=torch.load) as load:
        adapter = _adapter(tmp_path)
        assert load.call_count == 1
        assert load.call_args.kwargs["weights_only"] is True
    with patch.object(torch, "load", side_effect=AssertionError("request-time voice read")):
        assert adapter._prompt_len("eng") == 6
        assert adapter._prompt_len("eng", "different") == 4
    assert all(not isinstance(value, torch.Tensor) for entry in adapter._speaker_lengths.values() for value in entry)


@pytest.mark.parametrize("payload", [torch.ones(2, 4), {"speaker_encoding": torch.ones(2, 4)}])
def test_supported_pt_formats_are_identical(tmp_path, payload):
    from easymagpie_vllm_omni.speakers import load_speaker_embedding

    _fixture(tmp_path, payload)
    result = load_speaker_embedding(tmp_path / "speaker_embeddings/eng.pt", width=4, dtype=torch.float16)
    assert torch.equal(result, torch.ones(2, 4, dtype=torch.float16))
    assert result.device.type == "cpu" and result.is_contiguous()


@pytest.mark.parametrize(
    "payload",
    [
        torch.ones(4),
        torch.ones(2, 3),
        torch.empty(0, 4),
        torch.ones(2, 4, 1),
        torch.full((2, 4), float("nan")),
        torch.full((2, 4), float("inf")),
        torch.full((2, 4), 70000.0),
        torch.ones(2, 4, dtype=torch.complex64),
        torch.ones(2, 4).to_sparse(),
        nn.Parameter(torch.ones(2, 4)),
        {"other": torch.ones(2, 4)},
        torch.empty(2, 4, device="meta"),
    ],
)
def test_invalid_known_voice_is_rejected_before_device_transfer(tmp_path, payload):
    from easymagpie_vllm_omni.speakers import load_speaker_embedding

    _fixture(tmp_path, payload)
    with pytest.raises(ValueError, match="speaker"):
        load_speaker_embedding(tmp_path / "speaker_embeddings/eng.pt", width=4, dtype=torch.float16)


def test_unrelated_corrupt_voice_does_not_break_valid_model_or_adapter(tmp_path, monkeypatch):
    _fixture(tmp_path)
    (tmp_path / "speaker_embeddings/bad.pt").write_bytes(b"not a checkpoint")
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    assert model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float16).shape == (2, 4)
    assert adapter._prompt_len("eng") == 6
    with pytest.raises(ValueError, match="speaker"):
        model._load_known_speaker_embedding("bad", torch.device("cpu"), torch.float16)
    torch.save(torch.ones(3, 4), tmp_path / "speaker_embeddings/bad.pt")
    assert model._load_known_speaker_embedding("bad", torch.device("cpu"), torch.float16).shape == (3, 4)
    assert adapter._prompt_len("bad") == 7


def test_late_and_replaced_voices_keep_existing_combo_cache_semantics(tmp_path, monkeypatch):
    _fixture(tmp_path)
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    info = {"speaker_id": "eng", "context_text": "[EN]"}
    previous = model._build_prefill_embeds(torch.device("cpu"), info).clone()
    assert adapter._prompt_len("eng") == 6
    replacement = tmp_path / "speaker_embeddings/replacement.pt"
    torch.save(torch.full((5, 4), 9.0), replacement)
    replacement.replace(tmp_path / "speaker_embeddings/eng.pt")
    assert torch.equal(model._build_prefill_embeds(torch.device("cpu"), info), previous)
    assert adapter._prompt_len("eng") == 6
    changed = model._build_prefill_embeds(torch.device("cpu"), dict(info, context_text="new"))
    assert changed.shape == (7, 4) and torch.equal(changed[1:6], torch.full((5, 4), 9.0).half())
    assert adapter._prompt_len("eng", "new") == 7
    torch.save(torch.ones(3, 4), tmp_path / "speaker_embeddings/late.pt")
    assert model._load_known_speaker_embedding("late", torch.device("cpu"), torch.float16).shape == (3, 4)
    assert adapter._prompt_len("late") == 7


def test_removed_voice_still_errors_on_new_context_only(tmp_path, monkeypatch):
    _fixture(tmp_path)
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    info = {"speaker_id": "eng"}
    cached = model._build_prefill_embeds(torch.device("cpu"), info)
    assert adapter._prompt_len("eng") == 6
    (tmp_path / "speaker_embeddings/eng.pt").unlink()
    assert model._build_prefill_embeds(torch.device("cpu"), info) is cached
    assert adapter._prompt_len("eng") == 6
    with pytest.raises(AssertionError, match="unknown speaker_id"):
        model._build_prefill_embeds(torch.device("cpu"), dict(info, context_text="new"))
    with pytest.raises(FileNotFoundError):
        adapter._prompt_len("eng", "new")


def test_registered_buffers_follow_module_dtype_without_stale_tensor_references(tmp_path, monkeypatch):
    _fixture(tmp_path)
    model = _model(tmp_path, monkeypatch)
    before = model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float16)
    assert any(buffer is before for buffer in model.buffers())
    assert not any("speaker_embedding" in key for key in model.state_dict())
    model.to(dtype=torch.float64)
    after = model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float64)
    assert after is not before and any(buffer is after for buffer in model.buffers())
    assert torch.equal(after, before.double())


def test_raw_custom_voice_and_empty_directory_are_unchanged(tmp_path, monkeypatch):
    _fixture(tmp_path)
    (tmp_path / "speaker_embeddings/eng.pt").unlink()
    model = _model(tmp_path, monkeypatch)
    custom = torch.ones(2, 4)
    with patch.object(torch, "load", side_effect=AssertionError("no known voice requested")):
        result = model._build_prefill_embeds(torch.device("cpu"), {"speaker_embedding": custom})
    assert result.shape == (6, 4) and model._prefill_cache == {}
    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None))
    assert adapter._speaker_lengths == {}


def test_preload_count_file_and_converted_storage_budgets_are_bounded(tmp_path, monkeypatch):
    from easymagpie_vllm_omni import speakers

    _fixture(tmp_path)
    for name in ("a", "b", "c"):
        torch.save(torch.ones(2, 4), tmp_path / "speaker_embeddings" / f"{name}.pt")
    monkeypatch.setattr(speakers, "MAX_PRELOAD_ENTRIES", 2)
    assert [name for name, _, _ in speakers.preload_speakers(tmp_path, width=4, dtype=torch.float16)] == ["a", "b"]
    monkeypatch.setattr(speakers, "MAX_PRELOAD_ENTRIES", 64)
    monkeypatch.setattr(speakers, "MAX_PRELOAD_BYTES", 16)
    assert [name for name, _, _ in speakers.preload_speakers(tmp_path, width=4, dtype=torch.float16)] == ["a"]
    monkeypatch.setattr(speakers, "MAX_PRELOAD_FILE_BYTES", 1)
    with patch.object(torch, "load", side_effect=AssertionError("oversized speculative file")):
        assert list(speakers.preload_speakers(tmp_path, width=4, dtype=torch.float16)) == []


def test_compacts_views_and_rejects_storage_over_budget(tmp_path):
    from easymagpie_vllm_omni.speakers import load_speaker_embedding

    _fixture(tmp_path, torch.arange(400, dtype=torch.float32).view(100, 4)[:2])
    path = tmp_path / "speaker_embeddings/eng.pt"
    result = load_speaker_embedding(path, width=4, dtype=torch.float32)
    assert result.untyped_storage().nbytes() == 32
    with pytest.raises(ValueError, match="speaker"):
        load_speaker_embedding(path, width=4, dtype=torch.float16, max_bytes=8)


@pytest.mark.parametrize(
    "corrupt",
    [
        b"",
        b"\x80",
        b"\x80\x02",
        b"\x80\x02}",
        b"\xff",
        b"PK",
        b"\x80\x02h\x00",
        b"\x80\x02J",
        b"\x80\x02X\x01\x00\x00\x00\xff",
    ],
)
def test_corrupt_speculative_files_never_prevent_valid_voice_loading(tmp_path, corrupt):
    from easymagpie_vllm_omni.speakers import preload_speakers

    _fixture(tmp_path)
    (tmp_path / "speaker_embeddings/bad.pt").write_bytes(corrupt)
    assert [name for name, _, _ in preload_speakers(tmp_path, width=4)] == ["eng"]


def test_cache_hits_do_not_stat_and_unavailable_metadata_uses_lazy_path(tmp_path, monkeypatch):
    from easymagpie_vllm_omni import speakers

    _fixture(tmp_path)
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    info = {"speaker_id": "eng"}
    expected = model._build_prefill_embeds(torch.device("cpu"), info)
    assert adapter._prompt_len("eng") == 6
    with (
        patch.object(easymagpie, "speaker_fingerprint", side_effect=AssertionError("cached context stat")),
        patch.object(speakers, "speaker_fingerprint", side_effect=AssertionError("cached length stat")),
    ):
        assert model._build_prefill_embeds(torch.device("cpu"), info) is expected
        assert adapter._prompt_len("eng") == 6
    with (
        patch.object(easymagpie, "speaker_fingerprint", return_value=None),
        patch.object(speakers, "speaker_fingerprint", return_value=None),
        patch.object(torch, "load", wraps=torch.load) as load,
    ):
        assert model._build_prefill_embeds(torch.device("cpu"), dict(info, context_text="new")).shape == (4, 4)
        assert adapter._prompt_len("eng", "new") == 4
        assert load.call_count == 2


def test_voices_beyond_startup_count_and_byte_budgets_remain_lazy(tmp_path, monkeypatch):
    from easymagpie_vllm_omni import speakers

    _fixture(tmp_path)
    torch.save(torch.ones(3, 4), tmp_path / "speaker_embeddings/a.pt")
    monkeypatch.setattr(speakers, "MAX_PRELOAD_ENTRIES", 1)
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    assert set(model._speaker_buffers) == set(adapter._speaker_lengths) == {"a"}
    with patch.object(torch, "load", wraps=torch.load) as load:
        assert model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float16).shape == (2, 4)
        assert adapter._prompt_len("eng") == 6
        assert load.call_count == 2
    monkeypatch.setattr(speakers, "MAX_PRELOAD_ENTRIES", 64)
    monkeypatch.setattr(speakers, "MAX_PRELOAD_BYTES", 16)
    # A skipped oversized tensor does not consume the accepted-storage budget.
    assert [name for name, _, _ in speakers.preload_speakers(tmp_path, width=4, dtype=torch.float16)] == ["eng"]


def test_preload_skips_unreadable_or_concurrently_replaced_voice_without_negative_cache(tmp_path, monkeypatch):
    from easymagpie_vllm_omni import speakers

    _fixture(tmp_path)
    with patch.object(torch, "load", side_effect=PermissionError("unreadable optional voice")):
        assert list(speakers.preload_speakers(tmp_path, width=4)) == []
    with patch.object(speakers, "speaker_fingerprint", side_effect=[(1, 2, 3, 4), (1, 2, 3, 5)]):
        assert list(speakers.preload_speakers(tmp_path, width=4)) == []
    assert [name for name, _, _ in speakers.preload_speakers(tmp_path, width=4)] == ["eng"]


@pytest.mark.parametrize("task", [-1, 0, 1, 99])
@pytest.mark.parametrize("context", ["[EN]", "different"])
def test_preloaded_and_lazy_contexts_match_exactly_including_task_clamp(tmp_path, monkeypatch, task, context):
    _fixture(tmp_path)
    model = _model(tmp_path, monkeypatch)
    info = {"speaker_id": "eng", "task_mode_id": task, "context_text": context}
    prepared = model._build_prefill_embeds(torch.device("cpu"), info).clone()
    model._speaker_buffers.clear()
    model._prefill_cache.clear()
    with patch.object(torch, "load", wraps=torch.load) as load:
        lazy = model._build_prefill_embeds(torch.device("cpu"), info)
        assert load.call_count == 1
    torch.testing.assert_close(prepared, lazy, rtol=0, atol=0)


def test_internal_buffer_names_do_not_restrict_existing_voice_names(tmp_path, monkeypatch):
    _fixture(tmp_path)
    (tmp_path / "speaker_embeddings/eng.pt").rename(tmp_path / "speaker_embeddings/a.b voice.pt")
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    assert set(model._speaker_buffers) == {"a.b voice"}
    assert model._load_known_speaker_embedding("a.b voice", torch.device("cpu"), torch.float16).shape == (2, 4)
    assert adapter._prompt_len("a.b voice") == 6


@pytest.mark.parametrize("num_tasks", [0, 2])
@pytest.mark.parametrize("context", ["[EN]", "long context", None, ""])
def test_preloaded_http_and_ws_match_model_with_target_prefix_and_chunks(tmp_path, monkeypatch, num_tasks, context):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    config = _fixture(tmp_path)
    config.update(
        num_task_embeddings=num_tasks,
        phoneme_vocab_size=8,
        phoneme_stacking_factor=1,
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    model, adapter = _model(tmp_path, monkeypatch), _adapter(tmp_path)
    tokenizer = SimpleNamespace(
        encode_context=lambda text: [1, 2, 3] if text == "[EN]" else [4] * 25,
        encode=lambda _text, **_kwargs: [5, 6, 7, 8, 9],
    )
    model._text_tokenizer = adapter._tokenizer = tokenizer
    request = OpenAICreateSpeechRequest(input="hello", voice="eng", extra_params={"context_text": context})
    with patch.object(torch, "load", side_effect=AssertionError("preloaded request deserialized voice")):
        prepared = asyncio.run(adapter.build(request, [], False))
        info = prepared.prompt["additional_information"]
        embeds = model._build_prefill_embeds(torch.device("cpu"), info)
        assert (
            len(embeds)
            == len(prepared.prompt["prompt_token_ids"])
            == 2 + bool(num_tasks) + 4 + (25 if context == "long context" else 3)
        )
        chunks = []
        for start, stop in ((0, 2), (2, len(embeds) - 1), (len(embeds) - 1, len(embeds))):
            ids = torch.zeros(stop - start, dtype=torch.long)
            _, chunk, update = model._preprocess_prefill(ids, len(ids), ids.device, dict(info, prefill_offset=start))
            chunks.append(chunk)
            assert update == {"prefill_offset": stop, "decode_offset": 4}
        torch.testing.assert_close(torch.cat(chunks), embeds, rtol=0, atol=0)
        other = model._build_prefill_embeds(torch.device("cpu"), dict(info, prefill_text_tokens=[9, 8, 7, 6]))
        torch.testing.assert_close(other[:-4], embeds[:-4], rtol=0, atol=0)
        assert not torch.equal(other[-4:], embeds[-4:])
        websocket = adapter.build_streaming_spec(request)
        ws_info = websocket.prefill_prompt["additional_information"]
        assert ws_info["context_text"] == "[EN]"
        assert (
            len(websocket.prefill_prompt["prompt_token_ids"])
            == len(model._build_prefill_embeds(torch.device("cpu"), ws_info))
            == 2 + bool(num_tasks) + 3 + 4
        )
