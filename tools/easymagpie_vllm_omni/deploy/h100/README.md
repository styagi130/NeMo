# Optional single-H100 deployment

This bundle runs the complete [H100 profile](../easymagpie_h100.yaml) through upstream vLLM-Omni:
two Stage0 replicas, each supporting 64 requests, and one FP32 codec supporting 128. It requires the complete
EasyMagpie implementation, including the custom AR and codec workers. It has no companion-service dependency.
The ordinary `scripts/run_server.sh` remains unchanged; MPS and the historical tuning tables are opt-in here.

Run from the package checkout, or `/opt/easymagpie` in its standalone image. Supply a GPU reserved for this
deployment, a converted model containing `codec_native/`, and a new writable output directory whose parent
already exists. Set `EASYMAGPIE_GPU_UUID` to one full UUID from `nvidia-smi --query-gpu=uuid,name --format=csv,noheader`.
The UUID is required so device remapping cannot silently select a different GPU. No compute-mode changes,
GPU resets, fixed UID, global daemon shutdown or existing deployment mutation are performed.

```bash
# Preview commands only; does not validate GPU availability or claim readiness.
python3 scripts/launch_h100.py --model /models/tts --gpu "$EASYMAGPIE_GPU_UUID" \
  --output /runs/h100-preview --plan-only

# Supervised upstream serving, without starting MPS or enabling the tables.
python3 scripts/launch_h100.py --model /models/tts --gpu "$EASYMAGPIE_GPU_UUID" \
  --output /runs/h100-plain --api-port 8091 --master-port 29600

# Explicit private MPS plus the compatibility-checked historical kernel tables.
bash scripts/run_h100_mps.sh --model /models/tts --gpu "$EASYMAGPIE_GPU_UUID" \
  --output /runs/h100-mps --api-port 8091 --master-port 29600 --tuned-kernels
```

### Historical short-input comparison

The optional [benchmark profile](../easymagpie_h100_benchmark.yaml) preserves the exact YAML bytes from
`91770235` (SHA-256 `fc9e25681384cda7730e67b034b288b2952d305304780a9a256a4b2b542417ab`). Select it explicitly:

```bash
bash scripts/run_h100_mps.sh --model /models/tts --gpu "$EASYMAGPIE_GPU_UUID" \
  --config deploy/easymagpie_h100_benchmark.yaml --output /runs/h100-benchmark \
  --api-port 8091 --master-port 29600 --tuned-kernels
```

It uses a 64-token LM graph capture cap, 10 ms admission wait, busy startup ramp `[8]`, and codec
history/token budgets of 520/1536. Use it only for matched short-input performance comparisons; these
settings and any resulting timings do not establish long-input or WER/CER acceptance. Keep the default
H100 profile for its long-safe 4104-frame codec history and token budget.

The launcher accepts the historical Stage0 device template `0` and supplies upstream placement overrides:
`0,0` on the API/codec head and `0` on each headless LM. This keeps both LM replicas on the selected GPU
without editing the YAML or changing its performance settings. Use the same launcher, overrides, dependency
versions, model, workload and measurement procedure for both revisions in a comparison.

The H100 launcher sets `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`: shared experts do not get the separate auxiliary
stream. This does not promise that every backend-internal operation uses one global CUDA stream. It starts the
API+codec first, then each headless LM sequentially, then checks HTTP health. Priorities are placed in each child's
environment **before exec**: codec/API priority 0, LM priority 1. These are scheduling hints, not TTFA guarantees. The launcher
uses fresh per-run compilation/upload directories and enables INFO logging needed by its pinned startup markers.
Custom inherited vLLM logging configuration and tuning paths are not reused. Private mode replaces inherited MPS
settings; plain mode refuses them and refuses a pre-existing default MPS control PID file. This prevents accidental
attachment to known pre-existing MPS state; reserve the GPU and avoid concurrent daemon startup.
It never deletes stale/default control files. Local health checks bypass proxies and support IPv4/IPv6 binds.
Before starting children, temporary bind probes reject occupied API and loopback Omni master ports. This check
cannot reserve ports across startup; upstream binding remains authoritative. Preview mode performs no probes.

## Ownership and shutdown

Private MPS uses a fresh, current-user-owned directory and the foreground control daemon. Only that daemon's
private pipe receives `quit`; SIGTERM/SIGINT and failed startup stop only the launched process groups. The
supervisor first gives upstream launchers `--shutdown-timeout` seconds to reap their own workers, then records
any required group termination. It waits for/reaps its foreground daemon and records private-quit success.
Failed shutdown is not silently reported as clean. Temporary private MPS logs/directories are retained for
inspection, not recursively deleted. These operations follow NVIDIA's [MPS control interface](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html).

`launches.json` records parent PIDs, commands and pre-exec priorities. `ready.json` records startup completion;
`result.json` records final cleanup, including forced kills. A signal shutdown returns 128 plus its signal number.
Do not treat exit 0, a health probe or configured environment values alone as full inference/MPS validation.
In particular, verify both LM workers and the codec actually belong to the private daemon and inspect actual
worker stream/router/backend state. Do not attach to or terminate another deployment's MPS pipe.

Before promoting this new launcher, verify the installed `nvidia-cuda-mps-control -h` supports foreground `-f`,
run real fresh-start/failed-start/SIGTERM cleanup cases, and confirm no owned workers or daemon remain afterward.
The source/mock checks do not establish the exact image's daemon behavior, natural drain, WER/CER or performance.

## Exact optional tuning bundle

[manifest.json](manifest.json) pins the two table hashes, model dimensions, GPU name, runtime versions and the
three relevant vLLM source hashes. The launcher rejects mismatches and nonempty HF configuration overrides before
enabling `VLLM_TUNED_CONFIG_FOLDER`.
The supported profile is unquantized FP16 with FP32 Mamba cache, TP=1, 24 experts, hidden size 1536, intermediate
size 768, top-k=4, 64 Mamba heads, head dimension 24, state size 128 and 8 groups. The codec remains FP32. Weights and router math
are not rewritten. Start with tables disabled on any unsupported hardware/runtime/model; do not weaken checks
merely to suppress a warning. Verify both named tables are actually loaded in both LM worker startup logs.

The files are historical runtime snapshots, not a new autotuning result. Their original bytes are retained.
The Mamba file has 16 entries, but retained sweep evidence covers only effective batches 64/128/256/512/1024.
Its effective lookup batch is request batch times 64; both readers use nearest available keys. An entry beyond
the measured sweep, or reuse above the last key, is not proven tuning coverage. No per-table improvement or
percentage contribution is claimed.

## Regeneration and provenance

Use an otherwise idle reserved GPU and new output/cache directories, never a live serving cache. The existing
Mamba tuner validates candidate outputs and next state, and supports this model directly:

```bash
python3 scripts/tune_mamba_ssu.py --model /models/tts --tensor-parallel-size 1 \
  --model-dtype float16 --cache-dtype float32 \
  --batch-sizes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 24 32 40 48 56 64 \
  --output-dir /runs/new-mamba-tuning
```

This produces a **new** sweep, not guaranteed byte-for-byte reproduction of the historical file. Preserve every
measured shape/configuration, validation result, timing sample, runtime/source version and output hash.

The exact original MoE sweep harness is not retained, so full historical regeneration cannot be claimed. The
[pinned upstream benchmark](https://github.com/vllm-project/vllm/blob/ffd46bfab/benchmarks/kernels/benchmark_moe.py)
is a reference for config search, not a drop-in EasyMagpie recipe: it does not recognize the EasyMagpie architecture
name and its generic weight/activation path assumes gated SiLU. A fresh EasyMagpie sweep must explicitly use
the loaded non-gated SiLU Triton expert path, FP16 E=24/H=1536/I=768/top-k=4, real routing inputs, matching output and
state checks, and the desired token-batch distribution. Do not rename the model or use generic defaults to
manufacture a successful sweep. Until that harness is supplied and validated, preserve the historical table
with this limitation or leave the opt-in disabled.

Refresh the manifest only after numerical, quality and paired end-to-end checks. Restart and rewarm after
changing tables because readers and graphs cache choices. Nsight timings are diagnostic, not acceptance numbers.

## Focused checks

From the Speech root, the launcher tests can run without importing vLLM or touching a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/collections/tts/easymagpie_vllm_omni/serving/test_h100_launcher.py -v
```

The pinned serving environment should additionally run `test_deployment.py`, source/install provenance checks,
and the complete inference/quality/performance/lifecycle gates before this bundle is promoted.
