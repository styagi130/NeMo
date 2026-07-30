# EasyMagpie MoE changes: H100 porting and run guide

## Measured H100 conclusion (2026-07-28)

On the target NVIDIA H100 NVL, the controlled winner was:

```text
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
VLLM_TUNED_CONFIG_FOLDER=.../moe_configs_h100
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=75
```

The five-seed B16 result was 57.280x mean RTFX, 9.228 requests/s, 232.00 ms
mean TTFA, and 87.52 ms mean ITL, with 80/80 successful requests and zero
startup/steady underruns. Holding stream and MPS constant, H100 tiles improved
strict mean RTFX by 17.12% over vLLM defaults. See
[`H100_OPTIMIZATION_RESULTS.md`](H100_OPTIMIZATION_RESULTS.md) and the H100
lock manifest for complete attribution and reproduction details.

## Scope

This guide isolates the two Mixture-of-Experts optimizations used by the
EasyMagpie Stage-0 model:

1. keep shared and routed expert work on one CUDA stream;
2. load device-specific Triton fused-MoE tile configurations.

No MoE model architecture or checkpoint data was changed.

Unchanged model properties:

- 24 experts;
- top-4 routing;
- hidden size 1536;
- intermediate/output size 768;
- FP16 weights and activations;
- router weights, expert weights, shared expert, and numerical model outputs.

The changes affect CUDA scheduling and fused-GEMM launch geometry only.

---

## A4500 result that motivated the port

Five matched seeds were run at 16 requests and concurrency 16:

| Configuration | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL |
| --- | ---: | ---: | ---: | ---: |
| Auxiliary shared-expert stream, default tiles | 38.15x | 6.078 | 198.2 ms | 138.7 ms |
| Single stream, default tiles | 43.55x | 6.972 | 184.6 ms | 121.0 ms |
| Single stream, A4500 tiles | **44.98x** | **7.188** | **182.9 ms** | **116.4 ms** |

Attribution:

- single-stream expert execution: approximately **+14.2% RTFX**;
- A4500 tile tuning after the stream change: approximately **+3.3% RTFX**;
- combined controlled improvement: approximately **+17.9% RTFX**.

These percentages are A4500 results, not expected H100 gains.

---

## Change 1: shared-expert CUDA stream policy

The optimized A4500 launcher contains:

```bash
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
```

With `1`, vLLM runs the shared and routed expert paths on the Stage-0 stream.
With `0`, the shared expert can use its auxiliary CUDA stream.

Nsight on the A4500 showed repeated cross-stream rendezvous around the shared
expert. Because the FP32 codec was also competing for the same GPU through
CUDA MPS, the auxiliary stream added synchronization instead of useful
overlap.

H100 has substantially more execution and memory-bandwidth headroom. Do not
assume the A4500 winner remains optimal; run a matched `0` versus `1` A/B.

No source patch is required for this change. It is a vLLM runtime setting.

---

## Change 2: device-specific fused-MoE tiles

The A4500 configuration is:

```text
examples/tts/easymagpie_vllm_omni/moe_configs_a4500/
  E=24,N=768,device_name=NVIDIA_RTX_A4500.json
```

It maps observed active-token counts to Triton launch parameters:

```json
{
  "1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3
  }
}
```

The complete file has entries for:

```text
1, 2, 4, 6, 8, 13, 16
```

Parameter meanings:

| Parameter | Purpose |
| --- | --- |
| `BLOCK_SIZE_M` | Active-token rows handled by one Triton program |
| `BLOCK_SIZE_N` | Output/intermediate columns per program |
| `BLOCK_SIZE_K` | Reduction dimension per program |
| `GROUP_SIZE_M` | Program grouping along token rows |
| `num_warps` | Warps assigned to a program |
| `num_stages` | Software-pipeline depth |

The launcher selects the configuration folder with:

```bash
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER-${SCRIPT_DIR}/../moe_configs_a4500}"
```

No fused-MoE source code was modified.

---

## Why the A4500 JSON must not be reused as an H100 result

vLLM selects the filename using the exact GPU device name. Therefore the
A4500 file is not selected automatically on H100.

The H100 file will look similar to:

```text
E=24,N=768,device_name=NVIDIA_H100_80GB_HBM3.json
```

The exact name may differ for PCIe, SXM, cloud aliases, or future driver
reporting. Always obtain it from the target instance:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Replace spaces with underscores when constructing the vLLM filename:

```bash
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
GPU_FILE_NAME="${GPU_NAME// /_}"
printf 'GPU device: %s\nExpected filename: E=24,N=768,device_name=%s.json\n' \
  "$GPU_NAME" "$GPU_FILE_NAME"
```

Do not copy and rename the A4500 JSON and call it tuned. A4500 uses SM86;
H100 uses SM90 with different Tensor Cores, shared memory, occupancy, and
software-pipeline behavior.

Manually renamed A4500 tiles may be valid enough to launch, but they can
underperform or exceed a kernel resource limit. Use them only as optional
search seeds, never as the H100 final result.

---

## Files to transfer to the H100 instance

Transfer the complete working tree when possible because several files are
currently untracked. At minimum include:

```text
examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_bs16.sh
examples/tts/easymagpie_vllm_omni/scripts/setenv.sh
examples/tts/easymagpie_vllm_omni/scripts/benchmark_server.py
examples/tts/easymagpie_vllm_omni/bench_corpus.tsv
examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_optimized_bs16.yaml
examples/tts/easymagpie_vllm_omni/moe_configs_a4500/
```

The A4500 directory is a schema and comparison reference. The H100 instance
should create a separate directory:

```text
examples/tts/easymagpie_vllm_omni/moe_configs_h100/
```

Do not overwrite the A4500 configuration.

---

## H100 setup

### 1. Redirect caches

Use a writable persistent Lustre location:

```bash
export CACHE_ROOT=/lustre/<project>/<user>/easymp_cache
source examples/tts/easymagpie_vllm_omni/scripts/setenv.sh
```

Leave `TMPDIR` at local `/tmp`; long Lustre paths can exceed the Unix socket
path limit used by vLLM/ZMQ.

If vLLM runs inside a container, the Lustre cache root must be mounted into the
container and the variables must be set or sourced inside that container.

### 2. Create the H100 configuration folder

```bash
H100_MOE_DIR="$PWD/examples/tts/easymagpie_vllm_omni/moe_configs_h100"
mkdir -p "$H100_MOE_DIR"
export VLLM_TUNED_CONFIG_FOLDER="$H100_MOE_DIR"
```

An empty folder is valid for the initial baseline. vLLM will use its generic or
built-in fallback because no exact EasyMagpie H100 file exists yet.

### 3. Capture the environment

```bash
nvidia-smi --query-gpu=name,driver_version,pstate,power.limit \
  --format=csv,noheader
python3 - <<'PY'
import torch
import triton
import vllm

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
PY
```

Record whether MIG is enabled and whether clocks or power are externally
capped.

---

## Phase A: test the stream policy before tuning tiles

Start with the H100 folder empty so both stream variants use the same default
tiles.

Run shared-expert auxiliary stream mode:

```bash
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
export VLLM_TUNED_CONFIG_FOLDER="$H100_MOE_DIR"

examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_bs16.sh \
  /path/to/model 8091
```

After a full warmup, run five seeds:

```bash
for seed in 20260729 20260730 20260731 20260732 20260733; do
  python3 examples/tts/easymagpie_vllm_omni/scripts/benchmark_server.py \
    --text-file examples/tts/easymagpie_vllm_omni/bench_corpus.tsv \
    -n 16 -c 16 \
    --url http://127.0.0.1:8091 \
    --timeout 100 \
    --seed "$seed" \
    --no-warmup
done
```

Restart with:

```bash
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
```

Then repeat the same five seeds. Do not compare one cold run against one warm
run. Verify that no new JIT compilation warnings appear during measurement.

Choose using:

1. mean and median RTFX;
2. request throughput;
3. TTFA and ITL;
4. underruns and deadline misses;
5. Stage-0/Stage-1 overlap in a warmed Nsight trace.

---

## Phase B: tune H100 fused-MoE tiles

Use the vLLM 0.24 MoE benchmark/tuner from the exact source revision matching
the runtime image. The installed Python wheel may not include the benchmark,
so use a matching vLLM source checkout or copy its
`benchmarks/kernels/benchmark_moe.py` into the tuning environment.

Tune the EasyMagpie shape:

```text
E = 24
hidden size K = 1536
intermediate/output N = 768
top-k = 4
dtype = FP16
effective active-token sizes = 1, 2, 4, 6, 8, 13, 16
tensor parallel size = 1
```

Recommended initial H100 grid:

```text
BLOCK_SIZE_M: 16, 32, 64
BLOCK_SIZE_N: 64, 128, 256
BLOCK_SIZE_K: 64, 128, 256
GROUP_SIZE_M: 1
num_warps: 4, 8
num_stages: 3, 4, 5
```

For each active-token size:

1. reject configurations that fail compilation or exceed resources;
2. warm every candidate before timing;
3. measure enough repetitions to suppress first-launch noise;
4. retain the median kernel time;
5. emit the best valid configuration into the device-specific JSON.

The output schema must match the A4500 reference:

```json
{
  "triton_version": "<target-version>",
  "1": {
    "BLOCK_SIZE_M": 0,
    "BLOCK_SIZE_N": 0,
    "BLOCK_SIZE_K": 0,
    "GROUP_SIZE_M": 1,
    "num_warps": 0,
    "num_stages": 0
  }
}
```

Save it as:

```text
$H100_MOE_DIR/E=24,N=768,device_name=<exact_H100_device_name>.json
```

Do not include a dtype suffix for the current unquantized FP16 EasyMagpie
weights unless the matching vLLM runtime itself generates one.

---

## Phase C: verify that vLLM loaded the H100 file

Restart the service with:

```bash
export VLLM_TUNED_CONFIG_FOLDER="$H100_MOE_DIR"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=<winner-from-phase-A>
```

The startup log must contain a line equivalent to:

```text
Using configuration from .../moe_configs_h100/E=24,N=768,device_name=....json
```

If it reports a default configuration:

1. compare the exact `nvidia-smi` device name;
2. check underscore normalization;
3. check `E=24,N=768`;
4. check file readability inside the container;
5. verify `VLLM_TUNED_CONFIG_FOLDER` in the Stage-0 process environment.

---

## Phase D: end-to-end validation

Repeat three full-service variants with matched seeds:

| Variant | Purpose |
| --- | --- |
| Shared-expert stream, default tiles | Neutral baseline |
| Winning stream mode, default tiles | Attribute stream-policy gain |
| Winning stream mode, H100 tiles | Attribute tile gain |

Do not accept a tile file based only on the standalone kernel benchmark.
Stage 0 and the FP32 Stage-1 codec share the H100, so a tile that wins in
isolation can lose through occupancy or interference in the complete service.

Acceptance requirements:

- 80/80 requests succeed across five `n=16`, `c=16` runs;
- no quality or waveform corruption;
- zero steady-state underruns;
- no request-time JIT warnings;
- improved mean and median RTFX;
- no unacceptable TTFA/ITL regression.

Also rerun:

```bash
pytest -q \
  examples/tts/easymagpie_vllm_omni/tests/test_local_transformer_cache.py \
  examples/tts/easymagpie_vllm_omni/tests/test_runner.py \
  examples/tts/easymagpie_vllm_omni/tests/test_scheduler.py \
  examples/tts/easymagpie_vllm_omni/tests/test_stage_processors.py
```

Generate at least 10 WAVs and use:

```text
examples/tts/easymagpie_vllm_omni/scripts/evaluate_wer_cer.py
```

before saving the H100 lock manifest.

---

## Recommended H100 experiment matrix

After tile tuning, the following combined matrix is sufficient:

| Experiment | Shared-expert stream | Tile config | Stage-1 MPS |
| --- | ---: | --- | ---: |
| A | Enabled | Default | 50 |
| B | Disabled | Default | 50 |
| C | Stream-policy winner | H100 tuned | 50 |
| D | Complete winner | H100 tuned | 75 |
| E | Complete winner | H100 tuned | 100 |

This keeps stream, tile, and MPS effects attributable without testing every
Cartesian-product combination.

---

## Files produced on H100

Save:

```text
examples/tts/easymagpie_vllm_omni/moe_configs_h100/
  E=24,N=768,device_name=<exact_H100_name>.json

examples/tts/easymagpie_vllm_omni/deploy/
  easymagpie_native_best_h100_bs16.lock.yaml

nsys_traces/h100/
  matched benchmark logs
  warmed Nsight Systems report
  WER/CER results
```

The H100 lock should record:

- exact GPU name and MIG state;
- driver, CUDA, Triton, vLLM, and image versions;
- stream-policy value;
- MPS percentage;
- exact MoE JSON path;
- five seeds and aggregate metrics;
- underrun counts;
- WER/CER result path.

---

## Porting takeaway

Only two MoE runtime artifacts need to be reproduced:

```text
VLLM_DISABLE_SHARED_EXPERTS_STREAM=<H100 A/B winner>
VLLM_TUNED_CONFIG_FOLDER=<folder containing exact H100 JSON>
```

Everything else about the MoE model stays unchanged. The A4500 results justify
testing these optimizations, but the H100 stream decision and tiles must be
measured on the target instance.
