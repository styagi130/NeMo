#!/usr/bin/env bash
# Extended isolated H100 scaling launcher for the experimental C192 profile.
# Kept separate so the validated C64 launcher's locked dependency is unchanged.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> <192> [port]}"
CAPACITY="${2:?Usage: $0 <model_dir> <192> [port]}"
PORT="${3:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CAPACITY}" != 192 ]]; then
    echo "Unsupported extended capacity: ${CAPACITY}; expected 192" >&2
    exit 2
fi

export CACHE_ROOT="${CACHE_ROOT:-/workspace/.cache/easymp_h100}"
# shellcheck source=setenv.sh
source "${SCRIPT_DIR}/setenv.sh"

export EASYMAGPIE_DEPLOY_CONFIG="${EASYMAGPIE_DEPLOY_CONFIG:-${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c192_stage0x6.yaml}"
export EASYMAGPIE_NATIVE_OPTIMIZATIONS=1
export EASYMAGPIE_STAGE0_CUDA_PAYLOAD=1
export EASYMAGPIE_LOCAL_TRANSFORMER_KV_CACHE=1
export EASYMAGPIE_CODEC_TRANSFER_PARALLEL=1
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-75}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${SCRIPT_DIR}/../moe_configs_h100}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d
exec "${SCRIPT_DIR}/run_server.sh" "${MODEL}" "${PORT}"
