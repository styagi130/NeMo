#!/bin/bash
# Launch the BS16 pure vLLM-Omni path with a bounded dynamic-frame TensorRT
# codec plan.  The first packet remains low latency; later packets are drained
# from the Stage-1 codec queue and decoded with their real accumulated length.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${1:-${PROJECT_DIR}/easymp_vllm_model}"
HTTP_PORT="${2:-8091}"
DEPLOY_CONFIG="${3:-${PROJECT_DIR}/deploy/easymagpie_dynamic_frames_bs16.yaml}"
PLAN_FILE="${EASYMAGPIE_CODEC_TRT_PLAN_HOST:-/home/siddhartht/.cache/easymagpie_vllm_trt_codec_dynamic_bs16/model.plan}"
export EASYMAGPIE_CODEC_TRT_PLAN_HOST="${PLAN_FILE}"
export EASYMAGPIE_VLLM_TRT_CONTAINER="${EASYMAGPIE_VLLM_TRT_CONTAINER:-easymagpie-vllm-trt-codec-bs16-dynamic}"
export EASYMAGPIE_VLLM_TRT_IMAGE="${EASYMAGPIE_VLLM_TRT_IMAGE:-easymp-vllm-omni:trt-codec}"
export EASYMAGPIE_VLLM_CACHE_DIR="${EASYMAGPIE_VLLM_CACHE_DIR:-/home/siddhartht/.cache/easymagpie_vllm_trt_codec_dynamic_bs16/vllm_cache}"

exec "${SCRIPT_DIR}/run_server_trt_codec_bs32.sh" "${MODEL_DIR}" "${HTTP_PORT}" "${DEPLOY_CONFIG}"
