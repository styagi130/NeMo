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
"""Tests for the standalone EasyMagpie LM pipeline topology."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("vllm_omni")

from conftest import EASYMAGPIE_ROOT  # noqa: E402
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig  # noqa: E402
from easymagpie_vllm_omni.pipeline import EASYMAGPIE_LM_PIPELINE, EASYMAGPIE_PIPELINE  # noqa: E402
from vllm_omni.config.config_factory import StageConfigFactory  # noqa: E402
from vllm_omni.config.stage_config import StagePipelineConfig  # noqa: E402
from vllm_omni.engine import stage_init_utils  # noqa: E402
from vllm_omni.engine.arg_utils import OmniEngineArgs  # noqa: E402
from vllm_plugin_easymagpie_omni import register  # noqa: E402


def test_lm_pipeline_is_single_stage():
    assert EASYMAGPIE_LM_PIPELINE.model_type == "easymagpie_lm"
    assert len(EASYMAGPIE_LM_PIPELINE.stages) == 1
    stage = EASYMAGPIE_LM_PIPELINE.stages[0]
    assert stage.stage_id == 0
    assert stage.model_stage == "easymagpie"
    assert stage.final_output is True
    assert stage.final_output_type == "audio"
    assert stage.engine_output_type == "audio"
    assert stage.custom_process_next_stage_input_func is None
    assert stage.async_chunk_process_next_stage_input_func is None


def test_two_stage_pipeline_unchanged():
    assert EASYMAGPIE_PIPELINE.model_type == "easymagpie"
    assert len(EASYMAGPIE_PIPELINE.stages) == 2
    assert EASYMAGPIE_PIPELINE.stages[1].final_output is True
    assert EASYMAGPIE_PIPELINE.stages[1].final_output_type == "audio"


def test_only_codec_declares_retained_stream_state():
    assert EASYMAGPIE_PIPELINE.stages[1].retains_state_across_chunks is True
    assert EASYMAGPIE_PIPELINE.stages[0].retains_state_across_chunks is False
    assert EASYMAGPIE_LM_PIPELINE.stages[0].retains_state_across_chunks is False
    assert StagePipelineConfig(stage_id=0, model_stage="foreign").retains_state_across_chunks is False


@pytest.mark.parametrize(
    "profile,remote_stage,replicas",
    [
        ("easymagpie", None, [1, 1]),
        ("easymagpie", 0, [0, 1]),
        ("easymagpie", 1, [1, 0]),
        ("easymagpie_lm", None, [1]),
    ],
)
def test_retention_reaches_actual_model_config(tmp_path, monkeypatch, profile, remote_stage, replicas):
    register()
    model = tmp_path / "model"
    codec = model / "codec_native"
    codec.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["EasyMagpieTTSForConditionalGeneration"],
                "model_type": "nemotron_h",
                "max_position_embeddings": 4096,
            }
        )
    )
    EasyMagpieCodecConfig(max_position_embeddings=4104).save_pretrained(codec)
    # The CPU Omni platform has no generation worker. Stub only its class-path
    # selection; factory/argument/model-config construction below stays real.
    monkeypatch.setattr(
        stage_init_utils.current_omni_platform,
        "get_omni_generation_worker_cls",
        lambda: "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker",
    )
    path = str(EASYMAGPIE_ROOT / "deploy" / f"{profile}.yaml")
    overrides = {} if remote_stage is None else {f"stage_{remote_stage}_num_replicas": 0}
    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        str(model), trust_remote_code=True, cli_overrides=overrides, deploy_config_path=path
    )
    stages = [stage.to_omegaconf() for stage in stages]
    assert stage_init_utils.compute_replica_layout(stages, allow_zero=True)[0] == replicas
    transfer = stage_init_utils.load_omni_transfer_config_for_model(str(model), path)
    for stage in stages:
        spec = stage_init_utils.get_stage_connector_spec(
            omni_transfer_config=transfer, stage_id=stage.stage_id, async_chunk=bool(stage.engine_args.async_chunk)
        )
        mapping = stage_init_utils.build_engine_args_dict(stage, str(model), stage_connector_spec=spec)
        args = OmniEngineArgs(**stage_init_utils.filter_dataclass_kwargs(OmniEngineArgs, mapping))
        config = args.create_model_config()
        codec_stage = stage.stage_id == 1
        assert config.retains_state_across_chunks is codec_stage
        assert args.retains_state_across_chunks is codec_stage
        assert mapping["retains_state_across_chunks"] is codec_stage
        assert stage.engine_args.retains_state_across_chunks is codec_stage
        assert config.active_stream_window == 0
        assert config.async_chunk is (profile == "easymagpie")
        assert config.worker_type == ("generation" if codec_stage else "ar")
        assert args.max_num_seqs == 32
        assert args.max_model_len == (4104 if codec_stage else 4096)
        if codec_stage:
            assert args.max_num_batched_tokens == 4104
            assert args.enable_chunked_prefill is False
