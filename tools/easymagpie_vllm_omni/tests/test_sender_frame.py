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
import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni import stage_processors as candidate

spec = importlib.util.spec_from_file_location('fallback_stage_processors', Path(candidate.__file__))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference._SENDER_FRAME_FASTPATH = False


@pytest.fixture(autouse=True)
def enable_selected_path(monkeypatch):
    monkeypatch.setattr(candidate, '_SENDER_FRAME_FASTPATH', True)


@pytest.mark.parametrize('shape', [(16,), (1, 16), (3, 16), (0,), (0, 16), (2, 0), (1, 65)])
@pytest.mark.parametrize('dtype', [torch.long, torch.int32, torch.float32, torch.bool])
@pytest.mark.parametrize('zeros', [False, True])
def test_actual_extractor_matches_disabled_path(shape, dtype, zeros):
    values = (
        torch.zeros(shape, dtype=dtype) if zeros else torch.arange(torch.Size(shape).numel()).reshape(shape).to(dtype)
    )
    before = values.clone()
    got = candidate._extract_last_frame({'audio_codes': values})
    expected = reference._extract_last_frame({'audio_codes': values})
    assert (got is None) == (expected is None)
    if got is not None:
        assert got.dtype == expected.dtype and torch.equal(got, expected)
    assert torch.equal(values, before)


@pytest.mark.parametrize(
    'values',
    [
        torch.arange(64).reshape(4, 16)[:, ::2],
        -torch.arange(16).reshape(1, 16),
        torch.full((1, 16), 0.1),
        torch.ones((1, 16), requires_grad=True),
    ],
)
def test_fallbacks_and_negative_codes(values):
    got = candidate._extract_last_frame({'audio_codes': values})
    expected = reference._extract_last_frame({'audio_codes': values})
    assert torch.equal(got, expected) and got.dtype == expected.dtype


def test_invalid_rank_preserved():
    for module in (candidate, reference):
        with pytest.raises(ValueError, match='Invalid audio_codes shape'):
            module._extract_last_frame({'audio_codes': torch.ones((1, 2, 3), dtype=torch.long)})


def test_disabled_uses_parent_path(monkeypatch):
    monkeypatch.setattr(candidate, '_SENDER_FRAME_FASTPATH', False)
    previous = candidate.sender_frame_status()
    value = torch.arange(16).reshape(1, 16)
    assert torch.equal(
        candidate._extract_last_frame({'audio_codes': value}), reference._extract_last_frame({'audio_codes': value})
    )
    assert candidate.sender_frame_status() == previous


def manager():
    return SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=1)),
        connector=SimpleNamespace(config={'extra': {'codec_chunk_frames': 8, 'codec_startup_chunk_frames': [2, 6]}}),
        code_prompt_token_ids=defaultdict(list),
    )


def request():
    item = SimpleNamespace(external_req_id='same', resumable=True, output_token_ids=[], finished=False)
    item.is_finished = lambda: item.finished
    return item


def test_bulk_sender_flushes_and_resets_each_dialogue_turn():
    from easymagpie_vllm_omni.bulk_sender import BulkSender

    adapter, req, outputs = manager(), request(), []
    req.additional_information = {'reset_codec_on_segment': True}
    adapter.put_req_chunk = defaultdict(int)

    def send(task):
        adapter._easymagpie_send_finish = (task['is_finished'], task['is_segment_finished'])
        payload = candidate.talker2code2wav_async_chunk(adapter, task['multimodal_output'], req)
        if payload is not None:
            outputs.append(payload)

    adapter._send_single_request = send
    sender = BulkSender(adapter, capacity=1)
    sender.publish_ready = lambda ready: outputs.extend(record['payload'] for record in ready)
    for turn in range(2):
        for step in range(2):
            sender.send_envelope(
                dict(
                    request=req,
                    multimodal_output={'audio_codes': torch.full((1, 16), turn * 2 + step + 1)},
                    is_finished=False,
                    is_segment_finished=step == 1,
                )
            )
    assert len(outputs) == 2
    for turn, payload in enumerate(outputs):
        assert payload.meta.codec_streaming is False
        assert torch.equal(payload.codes.audio[0], torch.full((16,), turn * 2 + 2))
    assert req.external_req_id not in adapter._emp_seen_frames


def test_long_stream_exact_payloads_retained_outputs_and_cleanup():
    parents = [manager(), manager()]
    requests = [request(), request()]
    held = []
    for step in range(130):
        codes = torch.arange(16).reshape(1, 16) + step
        outputs = []
        for module, manager_, request_ in zip((reference, candidate), parents, requests):
            request_.finished = step == 129
            request_.resumable = not request_.finished
            outputs.append(module.talker2code2wav_async_chunk(manager_, {'audio_codes': codes}, request_))
        assert (outputs[0] is None) == (outputs[1] is None)
        if outputs[0] is not None:
            assert torch.equal(outputs[0].codes.audio, outputs[1].codes.audio)
            assert torch.equal(outputs[0].meta.finished, outputs[1].meta.finished)
            held.append((outputs[1].codes.audio, outputs[1].codes.audio.clone()))
        for actual, copy in held:
            assert torch.equal(actual, copy)
    for name in ('_emp_requests', '_emp_frame_buffer', '_emp_emitted_chunks', '_emp_seen_frames'):
        assert 'same' not in getattr(parents[1], name)
    status = candidate.sender_frame_status()
    assert status['enabled'] and status['extracted_rows'] >= 130 and status['reused_rows'] >= 128
    assert not torch.cuda.is_initialized()
