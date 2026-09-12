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
"""Tests for EasyMagpie streaming metadata on vLLM-Omni 0.26."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner, merge_streaming_additional_information

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


@pytest.mark.parametrize("padded,active", [(8, 3), (16, 9), (40, 34), (128, 65)])
def test_async_output_preserves_distinct_rows_after_contraction_and_slot_reuse(padded, active):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(omni_pooler_payload_include_hidden=False)
    hidden = torch.zeros(padded, 3)
    for epoch, count in enumerate((active, 1, active)):
        codes = torch.arange(padded * 2).view(padded, 2) + epoch * 1000
        snapshot = runner._build_omni_async_snapshot_payload(
            hidden_states=hidden,
            staged_hidden_states_cpu=None,
            multimodal_outputs={"codes": {"audio": codes}},
        )
        carrier = snapshot.get("hidden_states", hidden[:0])
        assert carrier.shape == (padded, 0)
        for row in range(count):
            output = runner._build_omni_mm_payload(
                combined_multimodal_outputs=None,
                mm_cpu={"codes.audio": codes},
                rid=f"request-{epoch}-{row}",
                idx=row,
                start=row,
                end=row + 1,
                audio_sparse_output=False,
                sparse_mm_index={},
                hidden_seq_len=carrier.shape[0],
                scheduled_seq_len=count,
            )
            torch.testing.assert_close(output["codes.audio"], codes[row : row + 1])


def test_async_payload_retains_requested_hidden_states():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(omni_pooler_payload_include_hidden=True)
    hidden = torch.ones(8, 3)
    snapshot = runner._build_omni_async_snapshot_payload(
        hidden_states=hidden, staged_hidden_states_cpu=hidden, multimodal_outputs={}
    )
    assert snapshot["hidden_states"] is hidden
    assert snapshot["staged_hidden_states_cpu"] is hidden


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resident_updates_own_storage_across_steps_and_requests(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.requests = {key: SimpleNamespace() for key in ("a", "b")}
    runner.model_intermediate_buffer = {}
    runner.model = SimpleNamespace(gpu_resident_buffer_keys={"last_audio_codes", ("hidden_states", "last")})
    codes = torch.tensor([[1, 2]], device=device)
    hidden = torch.tensor([[3.0]], device=device)
    for req_id in ("a", "b"):
        runner._update_intermediate_buffer(req_id, {"last_audio_codes": codes, "hidden_states": {"last": hidden}})
    codes.fill_(9)
    hidden.fill_(10)
    runner._update_intermediate_buffer("a", {"last_audio_codes": codes, "decode_offset": 2})

    a, b = (runner.model_intermediate_buffer[key] for key in ("a", "b"))
    torch.testing.assert_close(a["last_audio_codes"], codes)
    torch.testing.assert_close(b["last_audio_codes"], torch.tensor([[1, 2]], device=device))
    for key, cached in (("a", a), ("b", b)):
        torch.testing.assert_close(cached["hidden_states"]["last"], torch.tensor([[3.0]], device=device))
        assert runner.requests[key].additional_information_cpu is cached
        assert cached["last_audio_codes"].device == codes.device
        assert cached["last_audio_codes"].data_ptr() != codes.data_ptr()
    assert a["last_audio_codes"].data_ptr() != b["last_audio_codes"].data_ptr()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_eager_postprocess_owns_feedback_before_producer_and_slot_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.has_phoneme = True
    model._dec_phoneme_tokens = torch.arange(10, 18, device=device).view(8, 1)
    codes = torch.arange(16, device=device).view(8, 2)
    hidden = torch.zeros(8, 3, device=device)
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = model
    runner.requests = {key: SimpleNamespace() for key in ("a", "b")}
    runner.model_intermediate_buffer = {}
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="latent"))
    runner._request_needs_downstream_stage_payload = lambda req_id: True

    def eager_postprocess(req_ids, lengths, offsets):
        return runner._maybe_run_eager_omni_postprocess_before_async_output(
            hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": codes}},
            num_scheduled_tokens_np=np.array(lengths),
            scheduler_output=None,
            req_ids_output_copy=req_ids,
            query_start_loc_cpu=np.array(offsets),
        )

    # Mixed two-token prefill plus one-token decode, padded to eight rows.
    assert eager_postprocess(["a", "b"], [2, 1], [0, 2, 3])
    model._dec_phoneme_tokens.fill_(99)
    codes.fill_(99)
    for key, phoneme, audio in (("a", 11, [2, 3]), ("b", 12, [4, 5])):
        cached = runner.model_intermediate_buffer[key]
        torch.testing.assert_close(cached["last_phoneme_token"], torch.tensor([[phoneme]], device=device))
        torch.testing.assert_close(cached["last_audio_codes"], torch.tensor([audio], device=device))

    # The remaining request moves to row zero without overwriting the other cache.
    assert eager_postprocess(["b"], [1], [0, 1])
    model._dec_phoneme_tokens.zero_()
    codes.zero_()
    cached = runner.model_intermediate_buffer
    torch.testing.assert_close(cached["a"]["last_audio_codes"], torch.tensor([[2, 3]], device=device))
    torch.testing.assert_close(cached["b"]["last_audio_codes"], torch.tensor([[99, 99]], device=device))
    torch.testing.assert_close(cached["b"]["last_phoneme_token"], torch.tensor([[99]], device=device))
    assert "last_hidden" not in model.gpu_resident_buffer_keys


def test_streaming_update_preserves_model_state_and_replaces_latest_chunk():
    cached = {
        "decode_offset": 7,
        "text_tokens": [10, 20],
        "text_token": [20],
        "meta": {"num_processed_tokens": 3},
    }

    merged = merge_streaming_additional_information(cached, {"text_token": [30]})

    assert merged["decode_offset"] == 7
    assert merged["text_tokens"] == [10, 20]
    assert merged["text_token"] == [30]
    assert merged["meta"]["num_processed_tokens"] == 0
    assert merged["meta"]["resumable"] is True


def test_streaming_update_accumulates_declared_tensor_keys():
    cached = {"hidden_states": {"output": torch.tensor([[1.0]])}}
    incoming = {"hidden_states": {"output": torch.tensor([[2.0]])}}

    merged = merge_streaming_additional_information(
        cached,
        incoming,
        accumulated_keys={("hidden_states", "output")},
    )

    torch.testing.assert_close(merged["hidden_states"]["output"], torch.tensor([[1.0], [2.0]]))


def test_deploy_configs_select_compatibility_worker_for_lm():
    for filename in ("easymagpie_lm.yaml", "easymagpie.yaml"):
        deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / filename).read_text())
        lm_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
        assert lm_stage["engine_extras"]["worker_cls"] == WORKER_CLS
