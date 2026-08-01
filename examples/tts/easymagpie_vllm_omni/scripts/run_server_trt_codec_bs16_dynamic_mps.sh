#!/bin/bash
# B16 dynamic-frame TensorRT codec with CUDA MPS enabled for the two local
# vLLM worker processes. MPS reduces GPU-context serialization between Stage 0
# (acoustic-token generation) and Stage 1 (in-process TensorRT codec).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

MODEL_DIR="${1:-${PROJECT_DIR}/easymp_vllm_model}"
HTTP_PORT="${2:-8091}"
DEPLOY_CONFIG="${3:-${PROJECT_DIR}/deploy/easymagpie_dynamic_frames_bs16.yaml}"
PLAN_FILE="${EASYMAGPIE_CODEC_TRT_PLAN_HOST:-/home/siddhartht/.cache/easymagpie_vllm_trt_codec_dynamic_bs16/model.plan}"
IMAGE="${EASYMAGPIE_VLLM_TRT_IMAGE:-easymp-vllm-omni:trt-codec}"
CONTAINER_NAME="${EASYMAGPIE_VLLM_TRT_CONTAINER:-easymagpie-vllm-trt-codec-bs16-mps}"
VLLM_CACHE_DIR="${EASYMAGPIE_VLLM_CACHE_DIR:-/home/siddhartht/.cache/easymagpie_vllm_trt_codec_dynamic_bs16/vllm_cache}"
# Optional full process-tree Nsight Systems capture.  It is deliberately
# opt-in: normal serving keeps the exact reference launch command below.
NSYS_OUTPUT_DIR="${EASYMAGPIE_NSYS_OUTPUT_DIR:-}"
NSYS_DELAY_S="${EASYMAGPIE_NSYS_DELAY_S:-90}"
NSYS_DURATION_S="${EASYMAGPIE_NSYS_DURATION_S:-20}"

for required in "${MODEL_DIR}/config.json" "${DEPLOY_CONFIG}" "${PLAN_FILE}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 2; }
done
mkdir -p "${VLLM_CACHE_DIR}"
if [[ -n "${NSYS_OUTPUT_DIR}" ]]; then
    mkdir -p "${NSYS_OUTPUT_DIR}"
    NSYS_MOUNT=( -v "$(cd "${NSYS_OUTPUT_DIR}" && pwd):/nsys" )
    LAUNCH_COMMAND='exec /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=graph --delay="$1" --duration="$2" --force-overwrite=true --output=/nsys/easymagpie_bs16_mps /workspace/examples/tts/easymagpie_vllm_omni/scripts/run_server.sh /model "$0" /deploy.yaml'
else
    NSYS_MOUNT=()
    LAUNCH_COMMAND='exec /workspace/examples/tts/easymagpie_vllm_omni/scripts/run_server.sh /model "$0" /deploy.yaml'
fi

exec docker run --rm --name "${CONTAINER_NAME}" --gpus all --ipc host -p "${HTTP_PORT}:${HTTP_PORT}" \
    -v "${WORKSPACE_DIR}:/workspace:ro" \
    -v "$(cd "${MODEL_DIR}" && pwd):/model:ro" \
    -v "$(cd "$(dirname "${DEPLOY_CONFIG}")" && pwd)/$(basename "${DEPLOY_CONFIG}"):/deploy.yaml:ro" \
    -v "$(cd "$(dirname "${PLAN_FILE}")" && pwd)/$(basename "${PLAN_FILE}"):/codec-plan/model.plan:ro" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
    "${NSYS_MOUNT[@]}" \
    -e "PYTHONPATH=/workspace/examples/tts/easymagpie_vllm_omni:/opt/tritonserver/backends/dali/wheel/dali" \
    -e "VLLM_PLUGINS=easymagpie_omni" \
    -e "EASYMAGPIE_CODEC_TRT_PLAN=/codec-plan/model.plan" \
    -e "EASYMAGPIE_TRT_CODEC_USE_PRIVATE_STREAM=${EASYMAGPIE_TRT_CODEC_USE_PRIVATE_STREAM:-1}" \
    -e "EASYMAGPIE_NVTX=${EASYMAGPIE_NVTX:-0}" \
    -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" \
    -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" \
    "${IMAGE}" bash -lc \
    "nvidia-cuda-mps-control -d; ${LAUNCH_COMMAND}" \
    "${HTTP_PORT}" "${NSYS_DELAY_S}" "${NSYS_DURATION_S}"
