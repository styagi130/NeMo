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
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.serving_adapter import MODEL_TYPE, _build_adapter_cls, _patch_detection
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm_omni.entrypoints.openai import serving_speech
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@pytest.mark.parametrize("stream", [False, True])
def test_http_cap_reaches_upstream_generation_without_mutating_defaults(monkeypatch, stream):
    monkeypatch.setattr(
        serving_speech,
        "_SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES",
        serving_speech._SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES - {MODEL_TYPE},
    )
    monkeypatch.setattr(
        serving_speech.OmniOpenAIServingSpeech,
        "_detect_tts_model_type",
        serving_speech.OmniOpenAIServingSpeech._detect_tts_model_type,
    )
    _patch_detection()
    defaults = [
        SamplingParams(max_tokens=2048, output_kind=RequestOutputKind.DELTA),
        SamplingParams(max_tokens=512, output_kind=RequestOutputKind.DELTA),
    ]
    sent = []
    engine = SimpleNamespace(
        errored=False,
        default_sampling_params_list=defaults,
        generate=lambda **kwargs: sent.append(kwargs["sampling_params_list"]),
    )
    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=engine))
    adapter._prompt_len = lambda _speaker, _context_text="[EN]": 2
    adapter._text_stream_metadata = lambda: (99, 4)
    adapter._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11, 12, 13])
    service = SimpleNamespace(
        engine_client=engine,
        model_config=SimpleNamespace(async_chunk=True),
        _tts_model_type=MODEL_TYPE,
        _get_tts_adapter=lambda: adapter,
        _track_ref_audio_artifact_warmup=lambda *_args, **_kwargs: None,
        _tts_x_vector_only=lambda _params: False,
    )

    for cap in (1, 128, None):
        request = OpenAICreateSpeechRequest(input="hello", response_format="pcm", stream=stream, max_new_tokens=cap)
        asyncio.run(serving_speech.OmniOpenAIServingSpeech._prepare_speech_generation(service, request))

    assert [params[0].max_tokens for params in sent] == [1, 128, 2048]
    assert [params[1].max_tokens for params in sent] == [512, 512, 512]
    assert [params.max_tokens for params in defaults] == [2048, 512]
    assert sent[0][0] is not defaults[0]
    assert sent[1][0] is not defaults[0]
    assert sent[0][0] is not sent[1][0]
