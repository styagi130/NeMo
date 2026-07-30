#!/usr/bin/env bash
# Native vLLM EasyMagpie H100 profile with 32-request stage capacity.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CACHE_ROOT="${CACHE_ROOT:-/tmp/easymp_cache}"
# shellcheck source=setenv.sh
source "${SCRIPT_DIR}/setenv.sh"

export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c32.yaml"
export EASYMAGPIE_NATIVE_OPTIMIZATIONS=1
export EASYMAGPIE_STAGE0_CUDA_PAYLOAD=1
export EASYMAGPIE_LOCAL_TRANSFORMER_KV_CACHE=1
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-75}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${SCRIPT_DIR}/../moe_configs_h100}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d
exec "${SCRIPT_DIR}/run_server.sh" "${MODEL}" "${PORT}"
