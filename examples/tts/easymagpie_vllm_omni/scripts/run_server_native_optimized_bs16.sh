#!/usr/bin/env bash
# Native vLLM EasyMagpie codec optimized for 16 concurrent requests.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep compilation/JIT artifacts outside a quota-limited home directory.
# This container has only read-only bind mounts, so local runs use its writable
# /tmp layer. Cluster launchers should override CACHE_ROOT with a persistent
# Lustre location.
export CACHE_ROOT="${CACHE_ROOT:-/tmp/easymp_cache}"
# shellcheck source=setenv.sh
source "${SCRIPT_DIR}/setenv.sh"

export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_bs16.yaml"
# Enables only native vLLM hooks (transfer cohorting and codec dynamic drain).
# The Stage-0 acoustic payload remains on CUDA and crosses processes through
# the EasyMagpie CUDA IPC connector.
export EASYMAGPIE_NATIVE_OPTIMIZATIONS=1
export EASYMAGPIE_STAGE0_CUDA_PAYLOAD=1
export EASYMAGPIE_LOCAL_TRANSFORMER_KV_CACHE=1
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-50}"
# At B16 on one GPU, vLLM's auxiliary shared-expert stream adds a stream
# rendezvous at every MoE layer while Stage 1 is also competing through MPS.
# Keep the MoE work on the Stage-0 stream so CUDA-graph replay stays ordered
# without those cross-stream waits.
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
# Decode-shape Triton MoE tiles tuned on the RTX A4500 used by this profile.
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER-${SCRIPT_DIR}/../moe_configs_a4500}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d
exec "${SCRIPT_DIR}/run_server.sh" "${MODEL}" "${PORT}"
