# Prompt for the EasyMagpie H100 optimization agent

Copy the prompt below into a new Codex session on the H100 instance. Replace
the values in angle brackets before sending it.

---

## Copy-paste prompt

You are continuing an EasyMagpie native vLLM-Omni performance investigation on
an NVIDIA H100.

Work autonomously until the H100 configuration is measured, validated, and
saved. Do not stop after reading the handoffs or proposing a plan: inspect the
workspace, run the service, execute the controlled experiments, analyze the
results, and save the best validated H100 configuration. Ask me only if a
required external resource or materially ambiguous choice cannot be discovered
locally.

### Paths and runtime

```text
Repository: <ABSOLUTE_PATH_TO_NEMODUPLEXREALTIME>
Model: <ABSOLUTE_PATH_TO_MODEL>
Lustre cache root: <ABSOLUTE_WRITABLE_LUSTRE_CACHE_ROOT>
Service port: 8091
Target workload: 16 requests at concurrency 16
```

Before making changes:

1. Read any applicable `AGENTS.md` files.
2. Read these files completely:

```text
CODEX_HANDOFF.md
EASYMAGPIE_OPTIMIZATION_SESSION_SUMMARY.md
H100_OPTIMIZATION_HANDOFF.md
H100_MOE_PORTING_GUIDE.md
examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_bs16.lock.yaml
examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_optimized_bs16.yaml
examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_bs16.sh
examples/tts/easymagpie_vllm_omni/scripts/setenv.sh
```

3. Inspect `git status` before editing. The transferred worktree may contain
   important untracked implementation files. Do not clean, reset, overwrite,
   or discard unrelated work.
4. Confirm that the complete working tree was transferred. A normal
   `git diff` is insufficient because several required files were untracked on
   the A4500 host.

### Objective

Find the best stable H100 configuration for the native two-stage EasyMagpie
pipeline:

- Stage 0: EasyMagpie autoregressive model;
- Stage 1: native stateful FP32 codec;
- same-GPU CUDA IPC transport;
- LocalTransformer frame-local cache;
- 16-request, concurrency-16 serving.

Primary objective:

```text
maximize warmed mean and median RTFX
```

Hard constraints:

- 100% request success;
- zero steady-state playback underruns;
- preferably zero total underruns using the validated two-frame startup;
- no waveform corruption;
- no request-time JIT compilation in measured runs;
- no unacceptable TTFA or ITL regression;
- WER/CER must remain consistent with the A4500 quality baseline.

The A4500 reference points are:

```text
Maximum-throughput profile:
  mean RTFX: 44.98x
  median RTFX: 45.80x
  mean throughput: 7.19 req/s
  TTFA: 182.9 ms
  ITL: 116.4 ms
  underruns: confined to the old 80 ms startup packet

Active zero-underrun profile with redirected caches:
  mean RTFX: 43.42x
  median RTFX: 43.69x
  mean throughput: 6.93 req/s
  TTFA: 209.6 ms
  ITL: 120.7 ms
  underruns: 0 / 1,130 chunks across 80 requests
```

Do not compare H100 numbers directly against A4500 as a hardware-normalized
claim. Use matched H100 A/B experiments to attribute gains.

### Preserve the validated structural optimizations

Start with these enabled:

```bash
export EASYMAGPIE_NATIVE_OPTIMIZATIONS=1
export EASYMAGPIE_STAGE0_CUDA_PAYLOAD=1
export EASYMAGPIE_LOCAL_TRANSFORMER_KV_CACHE=1
```

Retain:

- structured `codes.audio` CUDA IPC;
- GPU-resident Stage-0 recurrent state;
- NeMo-compatible LocalTransformer cache for one complete
  `num_codebook_channels * frame_stacking_factor` autoregressive run;
- distribution-equivalent `Exp(1)` / `-log(E)` Gumbel sampling;
- append-only stateful Stage-1 codec requests;
- lock-safe contiguous successor drain;
- FP16 Stage 0;
- FP32 eager Stage 1;
- current startup `[2]` and steady `6` packet configuration;
- request-scoped profiling fixes and underrun accounting.

Do not enable the legacy generic codec scheduler microbatch monkeypatch. It
deadlocks with the vLLM-Omni 0.24 callback lifecycle.

Do not enable the Stage-0 in-process burst unless a separate matched experiment
proves a full-service improvement; it is intentionally disabled by default.

### Cache setup

Before launching the service:

```bash
cd <ABSOLUTE_PATH_TO_NEMODUPLEXREALTIME>
export CACHE_ROOT=<ABSOLUTE_WRITABLE_LUSTRE_CACHE_ROOT>
source examples/tts/easymagpie_vllm_omni/scripts/setenv.sh
```

Verify all requested cache variables in the actual API and worker process
environments. Confirm from startup logs that vLLM, TorchInductor, Triton, and
FlashInfer artifacts use the redirected cache.

Do not set `TMPDIR` to Lustre. Keep it on local `/tmp` to avoid the
107-character Unix-domain socket path limit.

The first launch into an empty cache is not a benchmark. Warm every relevant
request shape until no request-time JIT warning appears. Preserve the populated
cache for subsequent restarts.

### Record the H100 environment

Capture:

```bash
nvidia-smi --query-gpu=name,driver_version,pstate,power.limit \
  --format=csv,noheader
```

Also record:

- H100 PCIe or SXM device name exactly as reported;
- MIG state and slice size, if any;
- driver version;
- CUDA version;
- PyTorch version;
- Triton version;
- vLLM/vLLM-Omni revision;
- container image/digest;
- clock and power restrictions;
- cache root;
- model checksum or immutable model identifier.

### Experiment 1: H100 shared-expert stream policy

Create an initially empty H100 configuration directory:

```bash
export H100_MOE_DIR="$PWD/examples/tts/easymagpie_vllm_omni/moe_configs_h100"
mkdir -p "$H100_MOE_DIR"
export VLLM_TUNED_CONFIG_FOLDER="$H100_MOE_DIR"
```

Run a controlled A/B with default MoE tiles:

```text
A: VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
B: VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
```

For each setting:

1. restart cleanly;
2. wait until the health endpoint is ready;
3. warm request-path kernels;
4. verify no JIT warnings during measurement;
5. run the same five seeds `20260729`–`20260733`;
6. record mean, median, min, and max RTFX, request throughput, TTFA, ITL,
   underruns, deadline misses, generated audio duration, and failures.

Use:

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

Choose the H100 stream policy from the full-service result, not the A4500
winner.

### Experiment 2: H100 MoE tile tuning

Use the vLLM 0.24 MoE benchmark/tuner from the exact source revision matching
the runtime image.

Tune:

```text
experts E: 24
hidden/reduction K: 1536
intermediate/output N: 768
top-k: 4
dtype: FP16
tensor parallel size: 1
effective active-token sizes: 1, 2, 4, 6, 8, 13, 16
```

Search at least:

```text
BLOCK_SIZE_M: 16, 32, 64
BLOCK_SIZE_N: 64, 128, 256
BLOCK_SIZE_K: 64, 128, 256
GROUP_SIZE_M: 1
num_warps: 4, 8
num_stages: 3, 4, 5
```

Reject invalid or resource-exhausting candidates. Warm candidates before
timing and select by median kernel time.

Save the output using the exact H100 name reported by vLLM:

```text
examples/tts/easymagpie_vllm_omni/moe_configs_h100/
  E=24,N=768,device_name=<EXACT_H100_DEVICE_NAME_WITH_UNDERSCORES>.json
```

Do not simply rename the A4500 JSON. It may be used only as a search seed.

Restart and require a startup log showing:

```text
Using configuration from .../moe_configs_h100/E=24,N=768,device_name=....json
```

Then repeat the matched five-seed benchmark. Compare:

```text
1. shared-expert auxiliary stream, default tiles
2. winning stream policy, default tiles
3. winning stream policy, H100 tiles
```

This attribution table is required in the final report.

### Experiment 3: Stage-1 MPS allocation

After selecting the stream policy and H100 tiles, test:

```text
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=50
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=75
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=100
```

Leave Stage 0 uncapped initially. Use the same warm, matched-seed procedure.
Select the full-service winner based on RTFX while enforcing the latency and
underrun constraints.

Do not infer that the A4500 50% winner is optimal on H100.

### Nsight validation

Capture a warmed request-triggered Nsight Systems trace of the winning H100
profile.

Analyze:

- Stage-0 versus Stage-1 GPU time;
- shared/routed expert synchronization;
- CUDA stream waits;
- Stage-0/Stage-1 overlap;
- fused-MoE kernel time and selected tiles;
- host-side inter-step gaps;
- blocking D2H operations;
- codec convolution time;
- CUDA IPC ranges;
- any request-time compilation.

Keep cold initialization and JIT outside the measurement range.

### Underrun validation

Use the corrected playback metric that compares an arrival gap with the
duration of the previous packet.

Report underruns separately after:

- 160 ms startup packets;
- 480 ms steady packets.

The current starting configuration is:

```yaml
codec_startup_chunk_frames: [2]
codec_chunk_frames: 6
```

If the H100 still has startup underruns, first measure the first-to-second gap.
Use the smallest additional startup buffer that covers the measured p99/max
gap with margin. Do not jump directly to large `[6,6]` startup packets; that
configuration severely reduced A4500 RTFX and TTFA.

### Correctness and quality

Run:

```bash
pytest -q \
  examples/tts/easymagpie_vllm_omni/tests/test_local_transformer_cache.py \
  examples/tts/easymagpie_vllm_omni/tests/test_runner.py \
  examples/tts/easymagpie_vllm_omni/tests/test_scheduler.py \
  examples/tts/easymagpie_vllm_omni/tests/test_stage_processors.py
```

Generate at least 10 unique WAVs from the benchmark corpus and run:

```text
examples/tts/easymagpie_vllm_omni/scripts/evaluate_wer_cer.py
```

Compare against:

```text
10 samples
190 reference words
WER: 5.26%
CER: 1.74%
6 exact normalized utterances
```

Two short “De Mohrenschildt” prompts were ASR-sensitive; the other eight
measured 2.79% WER and 0.40% CER. Preserve per-utterance hypotheses and error
counts so proper-name ASR behavior is distinguishable from real model
regressions.

### Saving the result

When the H100 winner is validated:

1. save the exact H100 MoE JSON;
2. create:

```text
examples/tts/easymagpie_vllm_omni/deploy/
  easymagpie_native_best_h100_bs16.lock.yaml
```

3. record:

- hardware and MIG state;
- driver/CUDA/PyTorch/Triton/vLLM versions;
- image digest;
- cache root;
- launcher and deploy config;
- `VLLM_DISABLE_SHARED_EXPERTS_STREAM`;
- `VLLM_TUNED_CONFIG_FOLDER`;
- H100 MoE JSON path;
- Stage-1 MPS percentage;
- packet settings;
- five seeds and complete aggregate metrics;
- underrun counts by packet duration;
- Nsight report path;
- WER/CER results path;
- focused test result.

4. update `CODEX_HANDOFF.md`, `H100_OPTIMIZATION_HANDOFF.md`, and
   `H100_MOE_PORTING_GUIDE.md` with measured H100 results;
5. leave the A4500 lock and tile file intact.

### Reporting requirements

Lead the final report with the measured outcome. Include:

1. final H100 configuration;
2. controlled stream-policy attribution;
3. controlled H100-tile attribution;
4. MPS attribution;
5. mean/median/min/max RTFX and request throughput;
6. TTFA and ITL;
7. underruns and failures;
8. quality results;
9. Nsight bottlenecks;
10. exact files saved.

Clearly separate:

- cold-cache results;
- warmup runs;
- steady measured runs;
- isolated outliers;
- strict matched A/B comparisons;
- architectural expectations that were not measured.

Do not claim success from a single favorable run. Use the five fixed seeds and
save raw logs.

---

## End of copy-paste prompt
