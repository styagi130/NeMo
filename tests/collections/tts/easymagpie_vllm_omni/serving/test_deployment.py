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
"""Resolve shipped profiles through the pinned upstream configuration path."""

import hashlib
import json

import pytest
from conftest import EASYMAGPIE_ROOT

pytest.importorskip("vllm_omni")

from vllm_omni.config.config_factory import StageConfigFactory  # noqa: E402
from vllm_omni.engine.arg_utils import OmniEngineArgs  # noqa: E402
from vllm_omni.engine.stage_init_utils import (  # noqa: E402
    build_engine_args_dict,
    compute_replica_layout,
    filter_dataclass_kwargs,
    get_stage_connector_spec,
    load_omni_transfer_config_for_model,
)
from vllm_plugin_easymagpie_omni import register  # noqa: E402


@pytest.mark.parametrize(
    "profile,capacities,replicas", [("easymagpie", [32, 32], [1, 1]), ("easymagpie_h100", [64, 128], [2, 1])]
)
def test_shipped_profile_resolves_without_engine(tmp_path, profile, capacities, replicas):
    register()
    model = tmp_path / "model"
    codec = model / "codec_native"
    codec.mkdir(parents=True)
    for directory, architecture, model_type in (
        (model, "EasyMagpieTTSForConditionalGeneration", "nemotron_h"),
        (codec, "EasyMagpieCodecForConditionalGeneration", "easymagpie_codec"),
    ):
        (directory / "config.json").write_text(json.dumps({"architectures": [architecture], "model_type": model_type}))
    path = EASYMAGPIE_ROOT / "deploy" / f"{profile}.yaml"
    assert path.is_file()
    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        str(model), trust_remote_code=True, cli_overrides={}, deploy_config_path=str(path)
    )
    stages = [stage.to_omegaconf() for stage in stages]
    transfer = load_omni_transfer_config_for_model(str(model), str(path))
    args = []
    for stage in stages:
        spec = get_stage_connector_spec(omni_transfer_config=transfer, stage_id=stage.stage_id, async_chunk=True)
        mapping = build_engine_args_dict(stage, str(model), stage_connector_spec=spec)
        args.append(OmniEngineArgs(**filter_dataclass_kwargs(OmniEngineArgs, mapping)))
    assert compute_replica_layout(stages)[0] == replicas
    assert [arg.max_num_seqs for arg in args] == capacities
    assert args[1].dtype == "float32"
    assert args[1].max_model_len == args[0].max_model_len + 8
    assert args[1].max_num_batched_tokens >= args[1].max_model_len
    assert args[1].worker_cls == "easymagpie_vllm_omni.runner.EasyMagpieCodecGPUGenerationWorker"
    if profile == "easymagpie_h100":
        assert compute_replica_layout(stages)[1] == {0: ["0", "0"]}
        assert [arg.kv_cache_memory_bytes for arg in args] == [2**31, 2**28]
        extra = args[0].stage_connector_spec["extra"]
        assert extra["codec_chunk_frames"] == 8
        assert extra["codec_startup_chunk_frames"] == [2, 2, 2, 4]
        assert extra["codec_busy_startup_chunk_frames"] == [4, 4, 4, 8]
        assert extra["stage0_admission_coalesce_ms"] == 50
        assert extra["codec_startup_coalesce_ms"] == 2
        assert extra["codec_busy_coalesce_ms"] == 4


def test_historical_benchmark_profile_preserves_settings(tmp_path):
    register()
    model = tmp_path / "model"
    codec = model / "codec_native"
    codec.mkdir(parents=True)
    for directory, architecture, model_type in (
        (model, "EasyMagpieTTSForConditionalGeneration", "nemotron_h"),
        (codec, "EasyMagpieCodecForConditionalGeneration", "easymagpie_codec"),
    ):
        (directory / "config.json").write_text(json.dumps({"architectures": [architecture], "model_type": model_type}))
    path = EASYMAGPIE_ROOT / "deploy" / "easymagpie_h100_benchmark.yaml"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "fc9e25681384cda7730e67b034b288b2952d305304780a9a256a4b2b542417ab"
    )
    original, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        str(model), trust_remote_code=True, cli_overrides={}, deploy_config_path=str(path)
    )
    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
        str(model), trust_remote_code=True, cli_overrides={"stage_0_devices": "0,0"}, deploy_config_path=str(path)
    )
    assert original[0].to_omegaconf().engine_args == stages[0].to_omegaconf().engine_args
    stages = [stage.to_omegaconf() for stage in stages]
    assert compute_replica_layout(stages) == ([2, 1], {0: ["0", "0"]})
    transfer = load_omni_transfer_config_for_model(str(model), str(path))
    args = []
    for stage in stages:
        spec = get_stage_connector_spec(omni_transfer_config=transfer, stage_id=stage.stage_id, async_chunk=True)
        mapping = build_engine_args_dict(stage, str(model), stage_connector_spec=spec)
        args.append(OmniEngineArgs(**filter_dataclass_kwargs(OmniEngineArgs, mapping)))
    assert args[0].max_cudagraph_capture_size == 64
    assert [arg.max_num_seqs for arg in args] == [64, 128]
    assert [arg.max_model_len for arg in args] == [4096, 520]
    assert [arg.max_num_batched_tokens for arg in args] == [4096, 1536]
    assert [arg.kv_cache_memory_bytes for arg in args] == [2**31, 2**28]
    assert args[1].dtype == "float32"
    assert args[1].worker_cls == "easymagpie_vllm_omni.runner.EasyMagpieCodecGPUGenerationWorker"
    extra = args[0].stage_connector_spec["extra"]
    assert extra["codec_chunk_frames"] == 8
    assert extra["codec_startup_chunk_frames"] == [2, 2, 2, 4]
    assert extra["codec_busy_startup_chunk_frames"] == [8]
    assert extra["stage0_admission_coalesce_ms"] == 10
    assert extra["codec_startup_coalesce_ms"] == 2
    assert extra["codec_busy_coalesce_ms"] == 4
