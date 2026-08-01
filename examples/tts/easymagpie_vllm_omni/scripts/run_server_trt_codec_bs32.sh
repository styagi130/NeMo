#!/bin/bash
# Launch pure vLLM-Omni with the B32 codec TensorRT plan owned by Stage 1.
# Stage 0 emits acoustic-code windows; the Stage-1 scheduler cohorts ready
# windows and enqueues each cohort directly into the plan on the vLLM worker's
# current CUDA stream. No Triton BLS request is in this data path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

MODEL_DIR="${1:-${PROJECT_DIR}/easymp_vllm_model}"
HTTP_PORT="${2:-8091}"
DEPLOY_CONFIG="${3:-${PROJECT_DIR}/deploy/easymagpie_dynamic_bs32.yaml}"
PLAN_FILE="${EASYMAGPIE_CODEC_TRT_PLAN_HOST:-/home/siddhartht/.cache/easymagpie_triton_dynamic_bs32/codec/1/model.plan}"
IMAGE="${EASYMAGPIE_VLLM_TRT_IMAGE:-easymp-vllm-omni:trt-codec}"
CONTAINER_NAME="${EASYMAGPIE_VLLM_TRT_CONTAINER:-easymagpie-vllm-trt-codec-bs32}"
VLLM_CACHE_DIR="${EASYMAGPIE_VLLM_CACHE_DIR:-/home/siddhartht/.cache/easymagpie_vllm_trt_codec}"

for required in "${MODEL_DIR}/config.json" "${DEPLOY_CONFIG}" "${PLAN_FILE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required file: ${required}" >&2
        exit 2
    fi
done
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"
DEPLOY_CONFIG="$(cd "$(dirname "${DEPLOY_CONFIG}")" && pwd)/$(basename "${DEPLOY_CONFIG}")"
PLAN_FILE="$(cd "$(dirname "${PLAN_FILE}")" && pwd)/$(basename "${PLAN_FILE}")"
mkdir -p "${VLLM_CACHE_DIR}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Building ${IMAGE} with TensorRT Python bindings..."
    docker build \
        -t "${IMAGE}" \
        -f "${SCRIPT_DIR}/Dockerfile.vllm_trt_codec" \
        "${SCRIPT_DIR}"
fi

echo "Starting pure vLLM-Omni TensorRT-codec B32 on http://0.0.0.0:${HTTP_PORT}"
echo "  model:  ${MODEL_DIR}"
echo "  plan:   ${PLAN_FILE}"
echo "  deploy: ${DEPLOY_CONFIG}"

exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc host \
    -p "${HTTP_PORT}:${HTTP_PORT}" \
    -v "${WORKSPACE_DIR}:/workspace:ro" \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${DEPLOY_CONFIG}:/deploy.yaml:ro" \
    -v "${PLAN_FILE}:/codec-plan/model.plan:ro" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
    -e "PYTHONPATH=/workspace/examples/tts/easymagpie_vllm_omni:/opt/tritonserver/backends/dali/wheel/dali" \
    -e "VLLM_PLUGINS=easymagpie_omni" \
    -e "EASYMAGPIE_CODEC_TRT_PLAN=/codec-plan/model.plan" \
    -e "EASYMAGPIE_CODEC_BATCH_LOG_EVERY=${EASYMAGPIE_CODEC_BATCH_LOG_EVERY:-50}" \
    "${IMAGE}" \
    bash -lc 'exec /workspace/examples/tts/easymagpie_vllm_omni/scripts/run_server.sh /model "$0" /deploy.yaml' \
    "${HTTP_PORT}"
