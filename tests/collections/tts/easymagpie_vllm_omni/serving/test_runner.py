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

import pytest
import torch
import yaml

from conftest import EASYMAGPIE_ROOT
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
