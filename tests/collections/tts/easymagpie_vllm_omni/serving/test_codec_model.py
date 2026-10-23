# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.codec.model import EasyMagpieCodecForConditionalGeneration
from easymagpie_vllm_omni.watermark import DisabledAudioWatermarker, TaperedPerthWatermarker


def _payload(frames: int, codebooks: int = 16) -> dict:
    return {"codes": {"audio": torch.arange(frames * codebooks).view(frames, codebooks)}}


def _model(**overrides):
    config = dict(
        num_stacked_codebooks=16,
        frame_stacking_factor=2,
        codebook_size=1024,
        samples_per_codec_frame=3,
        samples_per_frame=6,
    )
    config.update(overrides)
    model = EasyMagpieCodecForConditionalGeneration.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(**config)
    return model


def test_payload_codes_skips_active_but_unscheduled_requests():
    model = _model()
    infos = [_payload(2), _payload(4), _payload(1)]
    spans = [(0, 2), (2, 2), (2, 3)]

    codes, frame_counts, valid_sample_counts, request_ids = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, infos, torch.device("cpu"), spans
    )

    assert codes.shape == (3, 16)
    assert frame_counts == [2, 1]
    assert request_ids == ["codec-0", "codec-2"]
    assert valid_sample_counts == [12, 6]
    assert torch.equal(codes[:2], infos[0]["codes"]["audio"])
    assert torch.equal(codes[2:], infos[2]["codes"]["audio"])


def test_codec_watermark_models_dir_is_relative_to_codec_model_path():
    config = SimpleNamespace(watermark_checkpoint="watermark/perth")
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="/converted/codec_native"))

    assert EasyMagpieCodecForConditionalGeneration._watermark_models_dir(config, vllm_config) == Path(
        "/converted/codec_native/watermark/perth"
    )


def test_payload_codes_rejects_scheduled_frame_mismatch():
    model = _model()

    with pytest.raises(ValueError, match="scheduled 1 placeholders.*2 frames"):
        EasyMagpieCodecForConditionalGeneration._payload_codes(model, [_payload(2)], torch.device("cpu"), [(0, 1)])


def test_payload_codes_computes_valid_lengths_before_device_transfer():
    model = _model(num_stacked_codebooks=4)
    infos = [
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]])}},
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [1025, 3, 1025, 103]])}},
    ]

    codes, frame_counts, valid_sample_counts, _ = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, infos, torch.device("meta")
    )

    assert codes.device.type == "meta"
    assert frame_counts == [2, 2]
    assert valid_sample_counts == [9, 6]


def test_payload_codes_trim_multiple_padded_terminal_frames():
    model = _model(num_stacked_codebooks=4)
    info = {
        "codes": {
            "audio": torch.tensor(
                [
                    [1, 2, 101, 102],
                    [3, 1025, 103, 1025],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            )
        }
    }

    _, frame_counts, valid_sample_counts, _ = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, [info], torch.device("cpu"), [(0, 4)]
    )

    assert frame_counts == [4]
    assert valid_sample_counts == [9]


def test_payload_codes_only_trims_control_from_the_terminal_row():
    model = _model(num_stacked_codebooks=4)
    info = {
        "codes": {
            "audio": torch.tensor(
                [
                    [1, 2, 101, 102],
                    [3, 1025, 103, 1025],
                    [4, 5, 104, 105],
                ]
            )
        }
    }

    _, _, valid_sample_counts, _ = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, [info], torch.device("cpu"), [(0, 3)]
    )

    assert valid_sample_counts == [18]


class _FakeCodec(torch.nn.Module):
    dtype = torch.float32

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.arange(codes.shape[0] * 6, dtype=self.dtype, device=codes.device)


@pytest.mark.parametrize(
    ("terminal_row", "expected_samples"),
    [
        ([3, 1025, 103, 1025], 9),
        ([1025, 3, 1025, 103], 6),
    ],
)
def test_forward_trims_terminal_control_subframes(terminal_row, expected_samples):
    model = EasyMagpieCodecForConditionalGeneration.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_stacked_codebooks=4,
        frame_stacking_factor=2,
        codebook_size=1024,
        samples_per_codec_frame=3,
        samples_per_frame=6,
        output_sample_rate=22050,
    )
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model.codec = _FakeCodec()
    info = {"codes": {"audio": torch.tensor([[1, 2, 101, 102], terminal_row])}}

    output = model.forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        runtime_additional_information=[info],
        request_token_spans=[(0, 2)],
    )

    audio = output.multimodal_outputs["model_outputs"][0]
    torch.testing.assert_close(audio, torch.arange(expected_samples, dtype=torch.float32))


def test_forward_keeps_request_offsets_when_terminal_audio_is_trimmed():
    model = _model(num_stacked_codebooks=4)
    model.config.output_sample_rate = 22050
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model.codec = _FakeCodec()
    infos = [
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]])}},
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [1025, 3, 1025, 103]])}},
    ]

    output = model.forward(
        input_ids=torch.zeros(4, dtype=torch.long),
        runtime_additional_information=infos,
        request_token_spans=[(0, 2), (2, 4)],
    )

    first, second = output.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(first, torch.arange(9, dtype=torch.float32))
    torch.testing.assert_close(second, torch.arange(12, 18, dtype=torch.float32))


class _RecordingWatermarker:
    def __init__(self):
        self.calls = []

    def apply(self, chunks):
        self.calls.append(chunks)
        return [chunk.audio + 100 for chunk in chunks]


def _watermark_model(watermarker):
    value = _model(num_stacked_codebooks=4, output_sample_rate=22050)
    value.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    value.codec = _FakeCodec()
    value.codec.dtype = torch.float16
    value.watermarker = watermarker
    return value


def _watermark_payload(rows, **metadata):
    return {"codes": {"audio": torch.tensor(rows, dtype=torch.long)}, **metadata}


@pytest.mark.parametrize("dispatch", ["native", "graphs", "groups"])
def test_marking_follows_decode_and_trim_with_scheduled_request_identity(dispatch):
    marker = _RecordingWatermarker()
    value = _watermark_model(marker)
    calls = []
    if dispatch == "graphs":

        def run(codes):
            calls.append("graphs")
            return value.codec(codes)

        value._codec_graphs = SimpleNamespace(run=run)
    elif dispatch == "groups":

        def run(codes, frame_counts):
            calls.append(frame_counts)
            return value.codec(codes)

        value._codec_groups = SimpleNamespace(run=run)
    normal = [1, 2, 101, 102]
    padding = [-1, -1, -1, -1]
    infos = [
        _watermark_payload([normal, normal], request_id="alpha"),
        {"codes": "unscheduled payload must not be parsed"},
        _watermark_payload([normal, [3, 1025, 103, 1025], padding], req_id="beta"),
        _watermark_payload([normal, padding]),
        _watermark_payload([padding, padding], external_req_id="empty"),
        _watermark_payload([normal, [1025, 3, 1025, 103]], request_id="delta"),
    ]
    result = value.forward(
        input_ids=torch.zeros(11, dtype=torch.long),
        runtime_additional_information=infos,
        request_token_spans=[(0, 2), (2, 2), (2, 5), (5, 7), (7, 9), (9, 11)],
    ).multimodal_outputs
    assert len(marker.calls) == 1
    chunks = marker.calls[0]
    assert [chunk.request_id for chunk in chunks] == ["alpha", "beta", "codec-3", "empty", "delta"]
    assert [chunk.is_final for chunk in chunks] == [False, True, False, False, True]
    assert [chunk.audio.numel() for chunk in chunks] == [12, 9, 6, 0, 6]
    expected = [(0, 12), (12, 21), (30, 36), (42, 42), (54, 60)]
    for chunk, output, (start, end) in zip(chunks, result["model_outputs"], expected, strict=True):
        assert chunk.audio.dtype == output.dtype == torch.float32
        torch.testing.assert_close(chunk.audio, torch.arange(start, end, dtype=torch.float32))
        torch.testing.assert_close(output, chunk.audio + 100)
    assert [rate.item() for rate in result["sr"]] == [22050] * 5
    assert calls == ({"native": [], "graphs": ["graphs"], "groups": [[2, 3, 2, 2, 2]]}[dispatch])


@pytest.mark.parametrize("marker", [None, DisabledAudioWatermarker()])
def test_disabled_marking_keeps_original_output_and_skips_terminal_read(marker):
    value = _watermark_model(marker)

    def unexpected(*args):
        raise AssertionError("disabled watermark read terminal codes")

    value._chunk_terminal_flags = unexpected
    result = value.forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        codec_codes=torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]]),
    ).multimodal_outputs
    torch.testing.assert_close(result["model_outputs"][0], torch.arange(9, dtype=torch.float32))


@pytest.mark.parametrize("source", ["direct", "scheduled"])
def test_independent_chunk_marking_skips_unused_terminal_read(source):
    marker = TaperedPerthWatermarker.__new__(TaperedPerthWatermarker)
    recording = _RecordingWatermarker()
    marker.apply = recording.apply
    value = _watermark_model(marker)

    def unexpected(*args):
        raise AssertionError("independent-chunk watermark read terminal codes")

    value._chunk_terminal_flags = unexpected
    codes = torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]])
    kwargs = (
        {"codec_codes": codes}
        if source == "direct"
        else {"runtime_additional_information": [{"codes": {"audio": codes}, "request_id": "request"}]}
    )
    result = value.forward(input_ids=torch.zeros(2, dtype=torch.long), **kwargs).multimodal_outputs

    assert marker.needs_terminal_flags is False
    assert len(recording.calls) == 1
    (chunk,) = recording.calls[0]
    assert chunk.request_id == ("codec-codes" if source == "direct" else "request")
    assert chunk.is_final is False
    torch.testing.assert_close(chunk.audio, torch.arange(9, dtype=torch.float32))
    torch.testing.assert_close(result["model_outputs"][0], chunk.audio + 100)


def test_profile_bypasses_watermark_and_terminal_reads():
    marker = _RecordingWatermarker()
    value = _watermark_model(marker)

    def unexpected(*args):
        raise AssertionError("profile read terminal codes")

    value._chunk_terminal_flags = unexpected
    result = value.forward(input_ids=torch.zeros(2, dtype=torch.long)).multimodal_outputs
    assert marker.calls == []
    torch.testing.assert_close(result["model_outputs"][0], torch.arange(12, dtype=torch.float32))


def test_direct_codes_keep_terminal_identity_and_apply_marker():
    marker = _RecordingWatermarker()
    value = _watermark_model(marker)
    result = value.forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        codec_codes=torch.tensor([[1, 2, 101, 102], [1025, 3, 1025, 103]]),
    ).multimodal_outputs
    assert len(marker.calls) == 1
    (chunk,) = marker.calls[0]
    assert chunk.request_id == "codec-codes" and chunk.is_final
    torch.testing.assert_close(result["model_outputs"][0], torch.arange(6, dtype=torch.float32) + 100)


def test_empty_scheduled_batch_does_not_create_phantom_chunks():
    marker = _RecordingWatermarker()
    value = _watermark_model(marker)
    result = value.forward(
        input_ids=torch.empty(0, dtype=torch.long),
        runtime_additional_information=[{"codes": "ignored"}],
        request_token_spans=[(0, 0)],
    ).multimodal_outputs
    assert marker.calls == [[]]
    assert result == {"model_outputs": [], "sr": []}
