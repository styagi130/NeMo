#!/usr/bin/env bash
# Redirect JIT, compilation, and download caches away from a quota-limited
# home directory. Override CACHE_ROOT before sourcing this file on a cluster:
#
#   CACHE_ROOT=/lustre/<project>/<user>/easymp_cache source setenv.sh
#
# TMPDIR is intentionally left unchanged. vLLM uses it for ZMQ/Unix sockets,
# whose paths are limited to 107 characters.

: "${CACHE_ROOT:=$HOME/.cache/easymp_cache}"

export CACHE_ROOT
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export CUDA_CACHE_PATH="$CACHE_ROOT/nv"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/hf"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export SPEAKER_SAMPLES_DIR="$CACHE_ROOT/vllm-omni/speakers"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export FLASHINFER_CUBIN_DIR="$CACHE_ROOT/flashinfer/cubins"

mkdir -p \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$CUDA_CACHE_PATH" \
    "$TORCH_HOME" \
    "$HF_HOME" \
    "$VLLM_CACHE_ROOT" \
    "$SPEAKER_SAMPLES_DIR" \
    "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer" \
    "$FLASHINFER_CUBIN_DIR"

echo "Caches redirected to: $CACHE_ROOT"
