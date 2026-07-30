"""Regression checks for capacity-matched H100 scaling profiles."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from easymagpie_vllm_omni.codec_microbatch import _transfer_microbatch_settings


_DEPLOY_DIR = Path(__file__).parents[1] / "deploy"
_SCALING_LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_server_native_scaling_h100.sh"
_C16_LOW_TTFA_LAUNCHER = (
    Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c16_low_ttfa.sh"
)
_C32_LOW_TTFA_LAUNCHER = (
    Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c32_low_ttfa.sh"
)
_C64_LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c64.sh"
_C64_LOW_TTFA_LAUNCHER = (
    Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c64_low_ttfa.sh"
)
_C128_LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c128.sh"
_C128_LOW_TTFA_LAUNCHER = (
    Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c128_low_ttfa.sh"
)
_C192_LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c192.sh"
_C256_X8_LAUNCHER = Path(__file__).parents[1] / "scripts" / "run_server_native_optimized_h100_c256_stage0x8.sh"
_B16_REPLICA_LAYOUTS = (
    (64, 4, "easymagpie_native_optimized_h100_c64_stage0x4_b16.yaml"),
    (128, 8, "easymagpie_native_optimized_h100_c128_stage0x8_b16.yaml"),
    (192, 12, "easymagpie_native_optimized_h100_c192_stage0x12_b16.yaml"),
    (256, 16, "easymagpie_native_optimized_h100_c256_stage0x16_b16.yaml"),
)
_CAPACITIES = (16, 32, 64, 128, 256)


@pytest.mark.parametrize("capacity", _CAPACITIES)
def test_h100_scaling_profile_has_no_narrower_pipeline_gate(capacity: int):
    config_path = _DEPLOY_DIR / f"easymagpie_native_optimized_h100_c{capacity}.yaml"
    config = yaml.safe_load(config_path.read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stages = config["stages"]

    assert [stage["max_num_seqs"] for stage in stages] == [capacity, capacity]
    assert extra["codec_microbatch_parallelism"] == capacity
    assert extra["codec_microbatch_max_batch_size"] == capacity
    assert stages[0]["max_num_batched_tokens"] >= capacity
    assert stages[1]["max_num_batched_tokens"] >= capacity


def test_transfer_parallelism_scales_past_legacy_b64_cap():
    """Matched B128/B256 profiles must not be split by a hidden B64 gate."""
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            stage_connector_config={
                "extra": {
                    "codec_microbatch_wait_us": 1_500,
                    "codec_microbatch_parallelism": 256,
                }
            }
        ),
        scheduler_max_num_seqs=256,
    )

    wait_s, parallelism = _transfer_microbatch_settings(adapter)

    assert wait_s == pytest.approx(0.0015)
    assert parallelism == 256


def test_scaling_launcher_enables_transfer_without_legacy_scheduler_patch():
    launcher = _SCALING_LAUNCHER.read_text()

    assert "export EASYMAGPIE_CODEC_TRANSFER_PARALLEL=1" in launcher
    assert "export EASYMAGPIE_CODEC_MICROBATCH=1" not in launcher


def test_c64_launcher_enables_fused_conv_transpose():
    launcher = _C64_LAUNCHER.read_text()

    assert "export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1" in launcher


@pytest.mark.parametrize(
    ("capacity", "startup_frames", "config_name", "launcher_path"),
    (
        (16, [2], "easymagpie_native_optimized_h100_c16_low_ttfa.yaml", _C16_LOW_TTFA_LAUNCHER),
        (32, [3, 5], "easymagpie_native_optimized_h100_c32_low_ttfa.yaml", _C32_LOW_TTFA_LAUNCHER),
    ),
)
def test_small_low_ttfa_profiles_use_capacity_safe_startup_and_remove_transfer_serialization(
    capacity: int,
    startup_frames: list[int],
    config_name: str,
    launcher_path: Path,
):
    config = yaml.safe_load((_DEPLOY_DIR / config_name).read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["max_num_seqs"] == capacity
    assert stage1["max_num_seqs"] == capacity
    assert extra["codec_startup_chunk_frames"] == startup_frames
    assert extra["codec_microbatch_parallelism"] == capacity
    assert extra["codec_microbatch_max_batch_size"] == capacity
    assert stage1["max_num_batched_tokens"] >= capacity * max(extra["codec_startup_chunk_frames"])

    launcher = launcher_path.read_text()
    assert config_name in launcher
    assert 'export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"' in launcher
    assert f"export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE={capacity}" in launcher
    expected_fused = 0 if capacity == 16 else 1
    assert f"export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE={expected_fused}" in launcher


def test_c64_low_ttfa_profile_preserves_b32_layout_and_full_startup_budget():
    config_name = "easymagpie_native_optimized_h100_c64_stage0x2_low_ttfa.yaml"
    config = yaml.safe_load((_DEPLOY_DIR / config_name).read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["num_replicas"] == 2
    assert stage0["max_num_seqs"] == 32
    assert stage0["num_replicas"] * stage0["max_num_seqs"] == 64
    assert extra["codec_startup_chunk_frames"] == [10, 14, 16]
    assert extra["codec_chunk_frames"] == 12
    assert extra["codec_fixed_chunk_frames"] == 12
    assert extra["codec_microbatch_parallelism"] == 64
    assert stage1["max_num_seqs"] == 64
    assert stage1["max_num_batched_tokens"] >= 64 * max(extra["codec_startup_chunk_frames"])

    launcher = _C64_LOW_TTFA_LAUNCHER.read_text()
    assert config_name in launcher
    assert 'export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"' in launcher
    assert "export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=64" in launcher
    assert "export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1" in launcher


def test_c128_stage0x4_profile_preserves_b32_ar_cohorts():
    config_path = _DEPLOY_DIR / "easymagpie_native_optimized_h100_c128_stage0x4.yaml"
    config = yaml.safe_load(config_path.read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["num_replicas"] == 4
    assert stage0["max_num_seqs"] == 32
    assert stage0["devices"].split(",") == ["0"] * 4
    assert stage0["num_replicas"] * stage0["max_num_seqs"] == 128
    assert stage1["max_num_seqs"] == 128
    assert stage1["max_num_batched_tokens"] >= 128 * max(extra["codec_startup_chunk_frames"])
    assert extra["codec_microbatch_parallelism"] == 128
    assert extra["codec_microbatch_max_batch_size"] == 128
    assert extra["codec_startup_chunk_frames"] == [16]
    assert extra["codec_fixed_chunk_frames"] == 12


def test_c128_launcher_selects_stage0x4_and_fused_codec():
    launcher = _C128_LAUNCHER.read_text()

    assert "easymagpie_native_optimized_h100_c128_stage0x4.yaml" in launcher
    assert "export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=128" in launcher
    assert "export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1" in launcher


def test_c128_low_ttfa_profile_realigns_to_locked_steady_codec_boundaries():
    config_name = "easymagpie_native_optimized_h100_c128_stage0x4_low_ttfa.yaml"
    config = yaml.safe_load((_DEPLOY_DIR / config_name).read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["num_replicas"] == 4
    assert stage0["max_num_seqs"] == 32
    assert extra["codec_startup_chunk_frames"] == [10, 14, 16]
    assert extra["codec_chunk_frames"] == 12
    assert extra["codec_fixed_chunk_frames"] == 12
    assert extra["codec_microbatch_parallelism"] == 128
    assert stage1["max_num_seqs"] == 128
    assert stage1["max_num_batched_tokens"] >= 128 * max(extra["codec_startup_chunk_frames"])

    launcher = _C128_LOW_TTFA_LAUNCHER.read_text()
    assert config_name in launcher
    assert 'export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"' in launcher
    assert "export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=128" in launcher
    assert "export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1" in launcher


@pytest.mark.parametrize(
    ("capacity", "replicas", "config_name"),
    (
        (192, 6, "easymagpie_native_optimized_h100_c192_stage0x6.yaml"),
        (256, 8, "easymagpie_native_optimized_h100_c256_stage0x8.yaml"),
    ),
)
def test_large_replica_profiles_preserve_b32_cohorts_and_full_startup_budget(
    capacity: int,
    replicas: int,
    config_name: str,
):
    config = yaml.safe_load((_DEPLOY_DIR / config_name).read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["num_replicas"] == replicas
    assert stage0["max_num_seqs"] == 32
    assert stage0["num_replicas"] * stage0["max_num_seqs"] == capacity
    assert stage0["devices"].split(",") == ["0"] * replicas
    assert stage1["max_num_seqs"] == capacity
    assert stage1["max_num_batched_tokens"] >= capacity * max(extra["codec_startup_chunk_frames"])
    assert extra["codec_microbatch_parallelism"] == capacity
    assert extra["codec_microbatch_max_batch_size"] == capacity


@pytest.mark.parametrize(
    ("launcher_path", "capacity", "config_name"),
    (
        (_C192_LAUNCHER, 192, "easymagpie_native_optimized_h100_c192_stage0x6.yaml"),
        (_C256_X8_LAUNCHER, 256, "easymagpie_native_optimized_h100_c256_stage0x8.yaml"),
    ),
)
def test_large_replica_launchers_select_fused_codec(
    launcher_path: Path,
    capacity: int,
    config_name: str,
):
    launcher = launcher_path.read_text()

    assert config_name in launcher
    assert f"export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE={capacity}" in launcher
    assert "export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1" in launcher


@pytest.mark.parametrize(("capacity", "replicas", "config_name"), _B16_REPLICA_LAYOUTS)
def test_b16_replica_profiles_have_full_capacity_and_startup_budget(
    capacity: int,
    replicas: int,
    config_name: str,
):
    config = yaml.safe_load((_DEPLOY_DIR / config_name).read_text())
    extra = config["connectors"]["connector_of_shared_memory"]["extra"]
    stage0, stage1 = config["stages"]

    assert stage0["num_replicas"] == replicas
    assert stage0["max_num_seqs"] == 16
    assert stage0["num_replicas"] * stage0["max_num_seqs"] == capacity
    assert stage0["devices"].split(",") == ["0"] * replicas
    assert stage1["max_num_seqs"] == capacity
    assert stage1["max_num_batched_tokens"] >= capacity * max(extra["codec_startup_chunk_frames"])
    assert extra["codec_microbatch_parallelism"] == capacity
    assert extra["codec_microbatch_max_batch_size"] == capacity
