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
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.serving_adapter import MODEL_TYPE, _build_adapter_cls
from easymagpie_vllm_omni.serving_stream import EasyMagpieStreamingSpeechHandler
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest, StreamingSpeechSessionConfig
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

_LONG_CONTEXT = " ".join(f"word{i}" for i in range(25))


def _context_fixture(tmp_path, num_tasks=0):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[": 1, "EN": 2, "]": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = EasyMagpieTextTokenizer(PreTrainedTokenizerFast(tokenizer_object=tokenizer), text_vocab_size=128)
    assert len(tokenizer.encode_context("[EN]")) == 3
    assert len(tokenizer.encode_context(_LONG_CONTEXT)) == 25
    (tmp_path / "speaker_embeddings").mkdir()
    torch.save(
        {"speaker_encoding": torch.arange(12, dtype=torch.float32).view(4, 3)}, tmp_path / "speaker_embeddings/eng.pt"
    )
    (tmp_path / "config.json").write_text(json.dumps({"num_task_embeddings": num_tasks}))
    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None))
    adapter._model_path_cache = str(tmp_path)
    adapter._tokenizer = tokenizer
    adapter._text_stream_metadata = lambda: (99, 4)

    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_path = str(tmp_path)
    model.embedding_dim = 3
    model._combined_embeddings = torch.zeros(1, 3)
    model._prefill_cache = {}
    model._text_tokenizer = tokenizer
    model.task_embedding = torch.nn.Embedding(num_tasks, 3) if num_tasks else None
    model.num_task_embeddings = num_tasks
    model.text_embedding = torch.nn.Embedding(128, 3)
    model.arch = SimpleNamespace(text_prefill_num=4)
    model.has_phoneme = False
    model._maybe_set_lt_sampling_params = lambda _info: None
    return adapter, model


@pytest.mark.parametrize("num_tasks", [0, 2])
@pytest.mark.parametrize("context_text", ["[EN]", _LONG_CONTEXT, None, ""])
def test_http_context_length_matches_exact_model_prefill(tmp_path, num_tasks, context_text):
    adapter, model = _context_fixture(tmp_path, num_tasks)
    request = OpenAICreateSpeechRequest(
        input="one two three four five", voice="eng", extra_params={"context_text": context_text}
    )
    prepared = asyncio.run(adapter.build(request, [], False))
    info = prepared.prompt["additional_information"]
    embeds = model._build_prefill_embeds(torch.device("cpu"), info)
    expected = 4 + (25 if context_text == _LONG_CONTEXT else 3) + 4 + bool(num_tasks)

    assert len(prepared.prompt["prompt_token_ids"]) == len(embeds) == expected
    assert info["context_text"] == (context_text or "[EN]")
    assert info["prefill_text_tokens"] == info["text_tokens"][:4]
    chunks = []
    for start, stop in ((0, 3), (3, expected - 2), (expected - 2, expected)):
        ids = torch.zeros(stop - start, dtype=torch.long)
        _, chunk, update = model._preprocess_prefill(ids, len(ids), ids.device, dict(info, prefill_offset=start))
        chunks.append(chunk)
        assert update["prefill_offset"] == stop
        assert update["decode_offset"] == 4
    torch.testing.assert_close(torch.cat(chunks), embeds, rtol=0, atol=0)


def test_same_speaker_context_switch_uses_separate_cached_lengths(tmp_path, monkeypatch):
    adapter, model = _context_fixture(tmp_path)
    original_load = torch.load
    reads = []

    def load(*args, **kwargs):
        reads.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    lengths = []
    for context in ("[EN]", _LONG_CONTEXT, "[EN]", _LONG_CONTEXT, None, ""):
        request = OpenAICreateSpeechRequest(
            input="one two three four", voice="eng", extra_params={"context_text": context}
        )
        lengths.append(len(asyncio.run(adapter.build(request, [], False)).prompt["prompt_token_ids"]))

    assert lengths == [11, 33, 11, 33, 11, 11]
    assert len(reads) == 2
    assert adapter._prompt_len("eng") == 7
    assert len(reads) == 2


@pytest.mark.parametrize("context_text", [False, 0, 1, [], {}, ["[EN]"]])
def test_http_rejects_non_text_context_before_build(tmp_path, context_text):
    adapter, _ = _context_fixture(tmp_path)
    request = OpenAICreateSpeechRequest(input="hello", voice="eng", extra_params={"context_text": context_text})
    assert adapter.validate(request) == "context_text must be a string or null"


def test_default_get_prompt_len_call_remains_backward_compatible(tmp_path):
    adapter, _ = _context_fixture(tmp_path, num_tasks=2)
    assert (
        EasyMagpieTTSForConditionalGeneration.get_prompt_len("eng", str(tmp_path), tokenize=adapter._tokenize()) == 8
    )


def test_websocket_default_context_matches_prefill_size(tmp_path):
    adapter, model = _context_fixture(tmp_path, num_tasks=2)
    request = OpenAICreateSpeechRequest(input="", voice="eng", response_format="pcm", stream=True)
    spec = adapter.build_streaming_spec(request)
    info = spec.prefill_prompt["additional_information"]
    assert info["context_text"] == "[EN]"
    assert (
        len(spec.prefill_prompt["prompt_token_ids"])
        == len(model._build_prefill_embeds(torch.device("cpu"), info))
        == 12
    )


@pytest.mark.parametrize("context_text", [_LONG_CONTEXT, None, [], False])
def test_upstream_http_preparation_preserves_context_contract(tmp_path, context_text):
    adapter, model = _context_fixture(tmp_path)
    sent = []
    engine = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=128), SamplingParams(max_tokens=128)],
        generate=lambda **kwargs: sent.append(kwargs),
    )
    service = SimpleNamespace(
        engine_client=engine,
        model_config=SimpleNamespace(async_chunk=True),
        _tts_model_type=MODEL_TYPE,
        _get_tts_adapter=lambda: adapter,
        _track_ref_audio_artifact_warmup=lambda *_args, **_kwargs: None,
        _tts_x_vector_only=lambda _params: False,
    )
    request = OpenAICreateSpeechRequest(
        input="one two three four five",
        voice="eng",
        response_format="pcm",
        stream=True,
        extra_params={"context_text": context_text},
    )
    if context_text is not None and not isinstance(context_text, str):
        with pytest.raises(ValueError, match="context_text must be a string or null"):
            asyncio.run(OmniOpenAIServingSpeech._prepare_speech_generation(service, request))
        assert not sent
        return

    asyncio.run(OmniOpenAIServingSpeech._prepare_speech_generation(service, request))
    assert len(sent) == 1
    prompt = sent[0]["prompt"]
    info = prompt["additional_information"]
    assert info["context_text"] == (context_text or "[EN]")
    assert len(prompt["prompt_token_ids"]) == len(model._build_prefill_embeds(torch.device("cpu"), info))


def test_upstream_websocket_config_does_not_expose_context_override():
    handler = EasyMagpieStreamingSpeechHandler.__new__(EasyMagpieStreamingSpeechHandler)
    handler._speech_service = SimpleNamespace()
    config = asyncio.run(
        handler._build_config(
            None,
            {
                "type": "session.config",
                "voice": "eng",
                "response_format": "pcm",
                "stream_audio": True,
                "extra_params": {"context_text": _LONG_CONTEXT},
            },
        )
    )
    assert isinstance(config, StreamingSpeechSessionConfig)
    assert "extra_params" not in config.model_dump()
    assert "context_text" not in config.model_dump()
