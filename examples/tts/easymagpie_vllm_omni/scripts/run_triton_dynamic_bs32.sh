#!/bin/bash
# Launch the EasyMagpie Triton + vLLM-Omni B32 deployment with dynamic codec windows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

# This repository contains a static 32-model-frame codec TensorRT plan.  Keep it
# separate from the F15 repository used by run_triton_bs32.sh.
MODEL_REPOSITORY="${1:-/home/siddhartht/.cache/easymagpie_triton_dynamic_bs32}"
HTTP_PORT="${2:-8000}"
GRPC_PORT="$((HTTP_PORT + 1))"
METRICS_PORT="$((HTTP_PORT + 2))"
CONTAINER_NAME="${EASYMAGPIE_TRITON_CONTAINER:-easymagpie-triton-dynamic-bs32}"
IMAGE="${EASYMAGPIE_TRITON_IMAGE:-easymp-vllm-omni:latest}"
VLLM_CACHE_DIR="${EASYMAGPIE_VLLM_CACHE_DIR:-${MODEL_REPOSITORY}/vllm_cache}"
PYTHONPATH_VALUE="/workspace/examples/tts/easymagpie_vllm_omni/easymagpie_vllm_omni:"
PYTHONPATH_VALUE+="/opt/tritonserver/backends/dali/wheel/dali"

if [[ ! -f "${MODEL_REPOSITORY}/codec/1/model.plan" ]]; then
    echo "Missing dynamic B32 TensorRT codec plan: ${MODEL_REPOSITORY}/codec/1/model.plan" >&2
    exit 2
fi
if [[ ! -f "${MODEL_REPOSITORY}/easymp/1/model.py" ]]; then
    echo "Missing dynamic B32 Triton Python backend: ${MODEL_REPOSITORY}/easymp/1/model.py" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_DIR}/triton_backend/model.py" ]]; then
    echo "Missing EasyMagpie Triton backend template: ${PROJECT_DIR}/triton_backend/model.py" >&2
    exit 2
fi

# Keep vLLM/Triton kernel compilation artifacts outside the read-only model
# repository mount.  This avoids cold JIT/autotuning on every container restart.
mkdir -p "${VLLM_CACHE_DIR}"

echo "Starting EasyMagpie Triton dynamic BS32: repo=${MODEL_REPOSITORY}, http=${HTTP_PORT}, grpc=${GRPC_PORT}"
echo "Post-first codec drain deadline: ${EASYMAGPIE_CODEC_DYNAMIC_CHUNK_WAIT_US:-400000} us"
# The queued codec scheduler remains opt-in until it batches by generation
# cohort rather than merely by queue arrival order.

exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc host \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -p "${METRICS_PORT}:8002" \
    -v "${MODEL_REPOSITORY}:/models:ro" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
    -v "${PROJECT_DIR}/triton_backend/model.py:/models/easymp/1/model.py:ro" \
    -v "${WORKSPACE_DIR}:/workspace:ro" \
    -e "PYTHONPATH=${PYTHONPATH_VALUE}" \
    -e "VLLM_PLUGINS=easymagpie_omni" \
    -e "EASYMAGPIE_CODEC_DYNAMIC_CHUNK_WAIT_US=${EASYMAGPIE_CODEC_DYNAMIC_CHUNK_WAIT_US:-400000}" \
    -e "EASYMAGPIE_CODEC_MICROBATCH_WAIT_US=${EASYMAGPIE_CODEC_MICROBATCH_WAIT_US:-10000}" \
    -e "EASYMAGPIE_CODEC_MICROBATCH_MAX_BATCH_SIZE=${EASYMAGPIE_CODEC_MICROBATCH_MAX_BATCH_SIZE:-32}" \
    -e "EASYMAGPIE_CODEC_DEVICE_IO=${EASYMAGPIE_CODEC_DEVICE_IO:-1}" \
    -e "EASYMAGPIE_CODEC_QUEUE_SCHEDULER=${EASYMAGPIE_CODEC_QUEUE_SCHEDULER:-0}" \
    "${IMAGE}" \
    tritonserver \
        --model-repository=/models \
        --model-control-mode=explicit \
        --load-model=codec \
        --load-model=easymp \
        --log-verbose=0
