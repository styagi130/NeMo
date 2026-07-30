#!/usr/bin/env bash
# Launch the validated serialization-free H100 C64 profile. Two same-GPU
# Stage-0 replicas each own a B32 AR cohort and feed one B64 codec stage.
set -euo pipefail

MODEL="${1:?Usage: $0 <model_dir> [port]}"
PORT="${2:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep admission, Stage-0 transfer, and the Stage-1 steady codec cohort at B64.
# TF32 accelerates only the conservative packed FP32 fallback convolution; the
# regular cuDNN codec paths already use TF32 on H100.
export EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE=64
export EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE="${EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
export EASYMAGPIE_DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie_native_optimized_h100_c64_stage0x2.yaml"
export EASYMAGPIE_CODEC_PACKED_CONV_TF32=1
export EASYMAGPIE_CODEC_FUSED_CONV_TRANSPOSE=1
# Carry the tiny codec control envelope in the existing local Unix datagram.
# CUDA IPC still owns the tensor payload, and POSIX SHM remains the automatic
# fallback if the bounded datagram queue is temporarily full.
export EASYMAGPIE_CUDA_IPC_DIRECT_CONTROL=1

exec "${SCRIPT_DIR}/run_server_native_scaling_h100.sh" "${MODEL}" 64 "${PORT}"
