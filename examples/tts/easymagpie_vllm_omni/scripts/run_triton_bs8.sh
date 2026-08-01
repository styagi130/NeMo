#!/bin/bash
# Launch the EasyMagpie Triton + vLLM-Omni BS8 deployment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

MODEL_REPOSITORY="${1:-/home/siddhartht/.cache/easymagpie_triton_bs8}"
HTTP_PORT="${2:-8000}"
GRPC_PORT="$((HTTP_PORT + 1))"
METRICS_PORT="$((HTTP_PORT + 2))"
CONTAINER_NAME="${EASYMAGPIE_TRITON_CONTAINER:-easymagpie-triton-bs8}"
IMAGE="${EASYMAGPIE_TRITON_IMAGE:-easymp-vllm-omni:latest}"

if [[ ! -f "${MODEL_REPOSITORY}/codec/1/model.plan" ]]; then
    echo "Missing BS8 TensorRT codec plan: ${MODEL_REPOSITORY}/codec/1/model.plan" >&2
    exit 2
fi
if [[ ! -f "${MODEL_REPOSITORY}/easymp/1/model.py" ]]; then
    echo "Missing Triton Python backend: ${MODEL_REPOSITORY}/easymp/1/model.py" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_DIR}/triton_backend/model.py" ]]; then
    echo "Missing EasyMagpie Triton backend template: ${PROJECT_DIR}/triton_backend/model.py" >&2
    exit 2
fi

echo "Starting EasyMagpie Triton BS8: repo=${MODEL_REPOSITORY}, http=${HTTP_PORT}, grpc=${GRPC_PORT}"

exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc host \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -p "${METRICS_PORT}:8002" \
    -v "${MODEL_REPOSITORY}:/models:ro" \
    -v "${PROJECT_DIR}/triton_backend/model.py:/models/easymp/1/model.py:ro" \
    -v "${WORKSPACE_DIR}:/workspace:ro" \
    -e "PYTHONPATH=/workspace/examples/tts/easymagpie_vllm_omni/easymagpie_vllm_omni:/opt/tritonserver/backends/dali/wheel/dali" \
    -e "VLLM_PLUGINS=easymagpie_omni" \
    -e "EASYMAGPIE_CODEC_MICROBATCH_WAIT_US=${EASYMAGPIE_CODEC_MICROBATCH_WAIT_US:-10000}" \
    -e "EASYMAGPIE_CODEC_MICROBATCH_MAX_BATCH_SIZE=${EASYMAGPIE_CODEC_MICROBATCH_MAX_BATCH_SIZE:-8}" \
    "${IMAGE}" \
    tritonserver \
        --model-repository=/models \
        --model-control-mode=explicit \
        --load-model=codec \
        --load-model=easymp \
        --log-verbose=0
