#!/usr/bin/env bash
# Launch the H100 C256 profile as eight independent B32 Stage-0 cohorts.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=256
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c256_stage0x8.yaml"
export EASYMAGPIE_CODEC_PACKED_CONV_TF32=1
export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1
export EASYMAGPIE_CUDA_IPC_DIRECT_CONTROL=1

exec "${SCRIPT_DIR}/run_server_native_scaling_h100.sh" "${MODEL}" 256 "${PORT}"
