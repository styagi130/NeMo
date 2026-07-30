#!/usr/bin/env bash
# Launch the experimental low-TTFA H100 C16 profile.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=16
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-75}"
export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c16_low_ttfa.yaml"
export EASYMAGPIE_CODEC_PACKED_CONV_TF32=1
# The fused path wins at large codec batches; native cuDNN has lower
# end-to-end overhead for this B16 latency profile.
export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=0
export EASYMAGPIE_CUDA_IPC_DIRECT_CONTROL=1

exec "${SCRIPT_DIR}/run_server_native_scaling_h100.sh" "${MODEL}" 16 "${PORT}"
