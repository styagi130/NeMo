#!/usr/bin/env bash
# Launch the experimental low-TTFA H100 C32 profile.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=32
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-75}"
export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c32_low_ttfa.yaml"
export EASYMAGPIE_CODEC_PACKED_CONV_TF32=1
export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1
export EASYMAGPIE_CUDA_IPC_DIRECT_CONTROL=1

exec "${SCRIPT_DIR}/run_server_native_scaling_h100.sh" "${MODEL}" 32 "${PORT}"
