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


import pytest

pytest.importorskip("numpy")
pytest.importorskip("requests")

import benchmark_server as benchmark  # noqa: E402


class _StreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        assert chunk_size is None
        yield b"\x00"
        yield b"\x40\xff"
        yield b"\x7f"


@pytest.mark.parametrize("max_new_tokens", [1, 128, 1024])
def test_request_uses_openai_speech_endpoint_and_decodes_streaming_pcm(monkeypatch, max_new_tokens):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    sent = {}

    def post(url, **kwargs):
        sent["url"] = url
        sent.update(kwargs)
        return _StreamingResponse()

    monkeypatch.setattr(benchmark.requests, "post", post)
    result = benchmark._do_request(
        {
            "url": "http://localhost:8091",
            "uttid": "test",
            "text": "Hello",
            "speaker_id": "eng",
            "max_new_tokens": max_new_tokens,
            "sample_rate": 22050,
            "timeout": 10,
            "output_dir": None,
        }
    )

    assert sent["url"] == "http://localhost:8091/v1/audio/speech"
    assert sent["json"] == {
        "input": "Hello",
        "voice": "eng",
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "max_new_tokens": max_new_tokens,
    }
    parsed = OpenAICreateSpeechRequest.model_validate(sent["json"])
    assert parsed.stream is True
    assert parsed.stream_format == "audio"
    assert parsed.max_new_tokens == max_new_tokens
    assert sent["stream"] is True
    assert result.error is None
    assert result.num_samples == 2
    assert result.sr == 22050


def test_make_tasks_avoids_output_filename_collisions_when_corpus_is_large_enough():
    tasks = benchmark._make_tasks(
        [("utt-1", "one"), ("utt-2", "two"), ("utt-3", "three")],
        3,
        url="http://localhost:8091",
        speaker_id="eng",
        max_new_tokens=128,
        sample_rate=22050,
        timeout=10,
        output_dir="wavs",
    )

    assert {task["uttid"] for task in tasks} == {"utt-1", "utt-2", "utt-3"}


def test_make_tasks_still_supports_more_requests_than_corpus_entries():
    tasks = benchmark._make_tasks([("utt-1", "one")], 2, "url", None, 128, 22050, 10, None)

    assert len(tasks) == 2


def test_playback_metrics_compare_gap_to_preceding_chunk_duration():
    metrics = benchmark._playback_metrics(
        arrivals=[0.0, 0.10, 0.25],
        durations=[0.08, 0.20, 0.20],
    )

    assert metrics["deadline_misses"] == 1
    assert metrics["underruns"] == 1
    assert metrics["chunk_rtfs"] == pytest.approx([0.8, 4.0 / 3.0])
    assert metrics["headrooms"] == pytest.approx([-0.02, 0.05])
    assert metrics["transitions"] == [
        pytest.approx((0.08, 0.10, True, True)),
        pytest.approx((0.20, 0.15, False, False)),
    ]


def test_load_items_accepts_tab_and_pipe_manifests(tmp_path):
    manifest = tmp_path / "texts.txt"
    manifest.write_text("tab\tText with | punctuation\n\npipe|Second text\n")

    assert benchmark._load_items(str(manifest)) == [
        ("tab", "Text with | punctuation"),
        ("pipe", "Second text"),
    ]


def test_save_wav_creates_parent_directory(tmp_path):
    import wave

    import numpy as np

    path = tmp_path / "nested" / "test.wav"
    benchmark._save_wav(path, np.array([0.0, 0.5]), 22050)

    with wave.open(str(path)) as wav:
        assert wav.getnframes() == 2
        assert wav.getframerate() == 22050


def test_http_success_does_not_imply_eos_or_known_cap_hits(capsys):
    summary = benchmark._summarize(
        [benchmark.RequestResult("audio", num_samples=22050), benchmark.RequestResult("error", error="timeout")],
        wall_s=2.0,
        concurrency=2,
    )

    assert summary["completion_unknown"] == 1
    assert summary["cap_hits"] is None
    assert summary["rtf"] == 0.5
    benchmark._print_detailed(summary)
    output = capsys.readouterr().out
    assert "1 HTTP audio responses / 1 failed" in output
    assert "completion unknown 1" in output
    assert "cap hits unknown" in output
    assert "complete-utterance throughput unverified" in output


@pytest.mark.parametrize("version", ["0.26.0", None])
def test_server_version_reports_only_available_metadata(monkeypatch, version):
    class Response:
        def raise_for_status(self):
            if version is None:
                raise benchmark.requests.HTTPError("404")

        def json(self):
            return {"version": version}

    monkeypatch.setattr(benchmark.requests, "get", lambda *args, **kwargs: Response())

    assert benchmark._server_version("http://localhost:8091", 10) == (version or "unknown")


def test_main_seed_and_measurement_metadata(monkeypatch, tmp_path, capsys):
    import sys

    manifest = tmp_path / "texts.txt"
    manifest.write_text("".join(f"utt-{i}\ttext {i}\n" for i in range(20)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_server.py", "--text-file", str(manifest), "-n", "4", "-c", "2", "--seed", "9101"],
    )
    monkeypatch.setattr(benchmark, "_server_version", lambda *args: "0.26.0")
    selections = []

    def run_level(tasks, concurrency, output_dir):
        selections.append([task["uttid"] for task in tasks])
        return [benchmark.RequestResult(task["uttid"], num_samples=22050) for task in tasks], 1.0

    monkeypatch.setattr(benchmark, "_run_level", run_level)
    benchmark.main()
    benchmark.main()

    assert selections[:2] == selections[2:]
    output = capsys.readouterr().out
    assert "vLLM server version 0.26.0" in output
    assert "sample rate 22050 Hz; selection seed 9101" in output
    assert "max_new_tokens requested 1024; effective unknown" in output
    assert "warmup 2; measured 4" in output


@pytest.mark.parametrize("cap", ["0", "-1"])
def test_main_rejects_nonpositive_generation_cap(monkeypatch, cap, capsys):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["benchmark_server.py", "--text-file", "unused", "-n", "1", "--max-new-tokens", cap]
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark.main()

    assert "must be positive" in capsys.readouterr().err
