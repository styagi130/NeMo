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
"""Codec completion uses the actual pinned upstream request/output lifecycle."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm_plugin_easymagpie_omni as plugin
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreRequest
from vllm_omni.engine.stage_runtime import StagePool
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor


@pytest.fixture
def registered(monkeypatch):
    # Restore the constructor after each test; exercise the already-imported alias.
    monkeypatch.setattr(StagePool, "__init__", StagePool.__init__)
    plugin.register()


@pytest.mark.parametrize("queued_updates", [1, 8])
@pytest.mark.parametrize("log_stats", [False, True])
@pytest.mark.parametrize("audio_size", [0, 7])
def test_final_codec_output_finishes_with_queued_input(registered, queued_updates, log_stats, audio_size):
    processor = _processor(log_stats=log_stats)
    _queue_final(processor, "request", queued_updates)
    state = processor.request_states["request"]
    assert [update.final for update in state.input_chunk_queue] == [False] * (queued_updates - 1) + [True]

    audio = torch.arange(audio_size, dtype=torch.float32)
    result = processor.process_outputs([_raw("request", audio=audio)], None, None)

    assert [output.finished for output in result.request_outputs] == [True]
    assert "request" not in processor.request_states
    assert "request" not in processor.external_req_ids
    assert result.reqs_to_abort == []
    torch.testing.assert_close(_audio(result.request_outputs[0]), audio, rtol=0, atol=0)


@pytest.mark.parametrize("queued_updates", [0, 1, 3])
@pytest.mark.parametrize("audio_size", [None, 0, 7])
@pytest.mark.parametrize("log_stats", [False, True])
@pytest.mark.parametrize("use_collector", [False, True])
def test_raw_final_does_not_wait_for_frontend_final(registered, queued_updates, audio_size, log_stats, use_collector):
    processor = _processor(log_stats=log_stats)
    collector = RequestOutputCollector(RequestOutputKind.DELTA, "request") if use_collector else None
    for _ in range(queued_updates + 1):
        processor.add_request(_request("request"), prompt=None, queue=collector)
    state = processor.request_states["request"]
    assert state.streaming_input and not any(update.final for update in state.input_chunk_queue)

    # The codec's final connector update can outrun the frontend's final input.
    audio = torch.arange(audio_size or 0, dtype=torch.float32)
    raw = _raw("request", audio=audio)
    if audio_size is None:
        raw.multimodal_output = None
    result = processor.process_outputs([raw], None, None)
    outputs = [collector.get_nowait()] if use_collector else result.request_outputs

    assert len(outputs) == 1 and outputs[0].finished
    assert not processor.request_states and not processor.external_req_ids
    assert result.reqs_to_abort == []
    torch.testing.assert_close(_audio(outputs[0]), audio, rtol=0, atol=0)


def test_segments_and_multiple_requests_keep_raw_order_and_cleanup(registered, monkeypatch):
    processor = _processor()
    _queue_final(processor, "a", 2)
    _queue_final(processor, "b", 1)
    finish = Mock(wraps=processor._finish_request)
    monkeypatch.setattr(processor, "_finish_request", finish)
    update = Mock(wraps=processor._update_stats_from_output)
    monkeypatch.setattr(processor, "_update_stats_from_output", update)

    result = processor.process_outputs(
        [
            _raw("a", segment=True, audio=torch.tensor([1.0])),
            _raw("b", audio=torch.tensor([3.0])),
            _raw("a", audio=torch.tensor([2.0])),
            _raw("a"),  # A duplicate terminal follows ordinary upstream missing-state handling.
            _raw("missing"),
        ],
        None,
        None,
    )

    assert [(output.request_id, output.finished) for output in result.request_outputs] == [
        ("a", False),
        ("b", True),
        ("a", True),
    ]
    audio_a = torch.cat([_audio(output) for output in result.request_outputs if output.request_id == "a"])
    torch.testing.assert_close(audio_a, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(_audio(result.request_outputs[1]), torch.tensor([3.0]), rtol=0, atol=0)
    assert finish.call_count == 2
    assert update.call_count == 3
    assert not processor.request_states
    assert not processor.external_req_ids
    assert result.reqs_to_abort == []


@pytest.mark.parametrize("model_type", ["easymagpie", "easymagpie_lm", "other_codec", None])
def test_other_model_processors_are_not_wrapped(registered, model_type):
    processor = _processor(model_type=model_type)
    assert processor._update_stats_from_output.__func__ is MultimodalOutputProcessor._update_stats_from_output
    _queue_final(processor, "request", 1)
    result = processor.process_outputs([_raw("request")], None, None)
    assert [output.finished for output in result.request_outputs] == [False]
    assert "request" in processor.request_states


@pytest.mark.parametrize("case", ["segment", "no_finish", "null_segment"])
def test_only_explicit_final_codec_boundaries_are_changed(registered, case):
    processor = _processor()
    _queue_final(processor, "request", 2)
    raw = _raw("request", segment=case == "segment", audio=torch.ones(1))
    if case == "no_finish":
        raw.finish_reason = None
    if case == "null_segment":
        raw.is_segment_finished = None

    result = processor.process_outputs([raw], None, None)

    assert [output.finished for output in result.request_outputs] == [False]
    assert processor.request_states["request"].streaming_input


def test_non_streaming_completion_is_unchanged(registered):
    processor = _processor()
    processor.add_request(_request("request", resumable=False), prompt=None, queue=None)
    result = processor.process_outputs([_raw("request")], None, None)
    assert [output.finished for output in result.request_outputs] == [True]
    assert not processor.request_states
    assert not processor.external_req_ids


@pytest.mark.parametrize("missing_segment_flag", [False, True])
def test_original_stats_callback_runs_once_and_preserves_result(registered, missing_segment_flag):
    from easymagpie_vllm_omni.codec_completion import _patch_processor

    original = Mock(return_value=object())
    processor = SimpleNamespace(_update_stats_from_output=original)
    _patch_processor(processor)
    wrapped = processor._update_stats_from_output
    _patch_processor(processor)
    assert processor._update_stats_from_output is wrapped
    state = SimpleNamespace(streaming_input=True, input_chunk_queue=[SimpleNamespace(final=True)])
    output = SimpleNamespace(finish_reason=FinishReason.STOP)
    if not missing_segment_flag:
        output.is_segment_finished = False

    assert wrapped(state, output, None, None) is original.return_value
    original.assert_called_once_with(state, output, None, None)
    assert state.streaming_input is missing_segment_flag


def test_original_stats_exception_is_not_hidden_or_followed_by_state_mutation(registered):
    from easymagpie_vllm_omni.codec_completion import _patch_processor

    error = RuntimeError("original callback failed")
    original = Mock(side_effect=error)
    processor = SimpleNamespace(_update_stats_from_output=original)
    _patch_processor(processor)
    state = SimpleNamespace(streaming_input=True, input_chunk_queue=[SimpleNamespace(final=True)])

    with pytest.raises(RuntimeError) as raised:
        processor._update_stats_from_output(state, _raw("request"), None, None)

    assert raised.value is error
    assert state.streaming_input
    assert original.call_count == 1


def test_registration_and_pool_installation_are_idempotent(registered):
    constructor = StagePool.__init__
    plugin.register()
    assert StagePool.__init__ is constructor
    processor = _processor()
    callback = processor._update_stats_from_output
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="easymagpie_codec")))
    pool = StagePool(1, [], output_processor=processor, stage_vllm_config=config)
    assert pool.output_processor is processor
    assert pool.stage_vllm_config is config
    assert processor._update_stats_from_output is callback
    StagePool(0, [])  # Non-LLM pools have no model configuration or processor.


@pytest.mark.parametrize("entrypoint", ["EngineArgs", "OmniEngineArgs", "stock_registration"])
def test_registration_in_a_fresh_cpu_process(entrypoint):
    if entrypoint == "stock_registration":
        code = """
import importlib.abc
import sys

class WithoutOmni(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm_omni' or fullname.startswith('vllm_omni.'):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, WithoutOmni())
import vllm_plugin_easymagpie_omni as plugin
plugin.register()
assert not any(name.startswith('vllm_omni') for name in sys.modules)
"""
    else:
        module = "vllm.engine.arg_utils" if entrypoint == "EngineArgs" else "vllm_omni.engine.arg_utils"
        code = f"""
from {module} import {entrypoint}
args = {entrypoint}(model='unused-model-no-load')
from vllm_omni.engine.stage_pool import StagePool
assert getattr(StagePool.__init__, '_easymagpie_codec_completion', False)
from vllm_omni.engine.orchestrator import Orchestrator
assert getattr(Orchestrator._prewarm_async_chunk_stages, '_easymagpie_terminal_prewarm', False)
"""
    code += "\nimport torch\nassert not torch.cuda.is_initialized()\nprint('CPU registration passed')\n"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    env["PYTHONPATH"] = str(Path(plugin.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU registration passed" in result.stdout


def _processor(model_type="easymagpie_codec", log_stats=False):
    processor = MultimodalOutputProcessor(None, log_stats=log_stats, engine_core_output_type="audio")
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)))
    StagePool(1, [], output_processor=processor, stage_vllm_config=config)
    return processor


def _request(request_id, resumable=True):
    return OmniEngineCoreRequest(
        request_id=request_id,
        external_req_id=request_id,
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=512, detokenize=False, output_kind=RequestOutputKind.DELTA),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


def _queue_final(processor, request_id, queued_updates):
    for _ in range(queued_updates + 1):
        processor.add_request(_request(request_id), prompt=None, queue=None)
    processor.add_request(_request(request_id, resumable=False), prompt=None, queue=None)


def _raw(request_id, segment=False, audio=None):
    return OmniEngineCoreOutput(
        request_id=request_id,
        new_token_ids=[],
        finish_reason=FinishReason.STOP,
        is_segment_finished=segment,
        multimodal_output={"audio": torch.empty(0) if audio is None else audio},
    )


def _audio(output):
    payload = getattr(output.outputs[0], "multimodal_output", None)
    if payload is None or "audio" not in payload:
        return torch.empty(0)
    audio = payload["audio"]
    return torch.cat(audio) if isinstance(audio, list) else audio
