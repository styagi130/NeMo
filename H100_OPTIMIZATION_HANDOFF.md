# EasyMagpie H100 optimization handoff

## C16/C32 low-TTFA experiments not promoted (2026-07-30)

The C64/C128 startup treatment does not transfer profitably to the smaller
capacities because their active profiles already start with two frames. The
best safe C32 staged candidate `[3,5]` measured 72.666x RTFX and 365.52 ms
TTFA with zero underruns, versus the active 76.087x / 339.67 ms result. The
safe C16 candidate measured 53.134x / 239.66 ms, versus the active locked
57.280x / 232.00 ms result.

An 80 ms B16 `[1,1,2]` ramp reached 220.7 ms in one warmed screen but caused
cold and transition underruns. It was rejected. Keep the existing C16 and C32
profiles active. See `H100_C16_C32_LOW_TTFA_RESULTS.md` and
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_h100_c16_c32_low_ttfa_experiment.yaml`.

## C64 low-TTFA 2xB32 winner (2026-07-30)

The validated C64 topology remains two same-GPU B32 Stage-0 replicas feeding
one C64 codec. With `[10,14,16]` startup packets, a complete 1,024-token
Stage-1 startup budget, MPS100, and the workspace plugin forced first on
`PYTHONPATH`, five matched seeds measured **595.58 ms TTFA** and **130.631x
RTFX**. All 320 requests completed with zero underruns and zero deadline
misses. This is 186.57 ms lower TTFA and 1.60% higher RTFX than the prior
locked C64 result.

See `H100_C64_SCALING_RESULTS.md` and
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64_low_ttfa.lock.yaml`.
The prior C64 lock remains byte-identical.

## C128 low-TTFA winner (2026-07-29)

The selected four-B32-replica profile uses a `[10,14,16]` startup schedule and
75% Stage-1 MPS. Five matched C128 seeds measured **747.5 ms mean TTFA** and
**214.760x mean RTFX**, completing 640/640 requests with zero underruns and
zero deadline misses. Relative to the locked `[16]` B32 baseline, TTFA is
137.6 ms lower (-15.54%) while RTFX changes by only -0.073%.

The investigation also found an environment-dependent transfer serialization:
without the workspace package first on `PYTHONPATH`, the runtime image imports
an older site-packages plugin and falls to approximately 154-163x. The launcher
now sets `PYTHONPATH` itself. A valid start prints four transfer-microbatch
messages with `parallelism=32`, one for each Stage-0 replica.

See `H100_C128_LOW_TTFA_RESULTS.md` and
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c128_low_ttfa.lock.yaml`.
Use the absolute in-container model path and source `setenv.sh` with
`CACHE_ROOT=/workspace/.cache/easymp_h100`; leave `TMPDIR` unchanged.

## Rejected B16-replica scaling experiment (2026-07-29)

The experiment using 4xB16 at C64, 8xB16 at C128, 12xB16 at C192, and
16xB16 at C256 is stopped. C64 improved modestly to 134.201x mean RTFX, but
C128 fell to 148.831x and C192 to 153.173x; C192 also produced three
underruns and 897 deadline misses. C256 16xB16 could not allocate any KV cache
for its first engine at the tested memory settings.

Keep the validated B32-pool locks active. See
`H100_B16_REPLICA_REJECTED_RESULTS.md` for cold, matched, outlier, and startup
failure details.

## C128/C192/C256 B32-replica scaling (2026-07-29)

The B32-replica architecture now scales to four, six, and eight same-GPU
Stage-0 engines. Five matched seeds measured 214.917x mean RTFX at C128,
255.544x at C192, and 270.711x at C256, with zero failed requests, underruns,
or deadline misses. C256 reaches 99-100% GPU utilization and 391-400 W, so the
single-H100 design is compute/power saturated there; C192 is the better
latency-efficiency point.

B128 profiling exposed and removed a remaining Stage-1 512-token scheduling
gate. The accepted profiles budget one complete 16-frame startup cohort:
2,048 tokens at C128, 3,072 at C192, and 4,096 at C256.

See `H100_REPLICA_SCALING_RESULTS.md` and the C128/C192/C256 lock manifests in
`examples/tts/easymagpie_vllm_omni/deploy/`.

## Current C64 winner (2026-07-29)

The active launcher now uses two same-GPU B32 Stage-0 replicas feeding one
Stage-1 codec with B64 capacity. Five matched C64 seeds measured **128.573x
mean RTFX** (121.824–136.494x), 21.081 requests/s, zero underruns, and zero
deadline misses. The final cold run was 75.323x. The real codec batch maximum
was B42; forcing every launch to B64 timed out 7 requests and was rejected.

Use `H100_C64_SCALING_RESULTS.md` and
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64.lock.yaml`
as the current results and reproduction sources. The remainder of this handoff
documents the earlier single-Stage-0 path and rejected variants.

## Completed 64-request scaling result (2026-07-29)

The serialization-free C64 service now keeps both stage capacities, Stage-0
transfer parallelism, and the Stage-1 steady codec cohort at B64. With H100
TF32 packed convolutions, fused ConvTranspose1d, successor draining disabled,
Stage-1 MPS at 100%, and a 640 ms startup packet, five exact matched seeds
completed 320/320 requests and 4,117 chunks at 69.678x mean RTFX and 11.370
requests/s with zero underruns. The fused warmed Nsight trace reduces
ConvTranspose GPU time from 170.680 ms to 6.032 ms; packed causal Conv1D is now
the largest Stage-1 kernel. End-to-end RTFX is 2.824% below the prior cuDNN
winner despite removing that kernel bottleneck, and this tradeoff is recorded
in the lock.

See [`H100_C64_SCALING_RESULTS.md`](H100_C64_SCALING_RESULTS.md) for the
controlled B16/B32 cohort and MPS A/B, compute-bound diagnosis, cold run, and
separated outliers. The C64 lock is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64.lock.yaml`.

## Completed H100 result (2026-07-28)

The H100 port and controlled validation are complete. The selected NVIDIA H100
NVL B16 configuration uses shared-expert stream 0, the device-specific H100 MoE
tile JSON, and 75% Stage-1 MPS. Five fixed seeds completed 80/80 requests at
57.280x mean RTFX, 9.228 mean requests/s, 232.00 ms mean TTFA, and 87.52 ms
mean ITL, with zero startup or steady-state underruns.

See [`H100_OPTIMIZATION_RESULTS.md`](H100_OPTIMIZATION_RESULTS.md) for the
strict A/B attribution, separated cold/warmup/outlier results, warmed Nsight
analysis, quality evaluation, and saved artifact paths. The lock manifest is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_bs16.lock.yaml`.

For the focused shared-expert stream and fused-MoE tile workflow, see
`H100_MOE_PORTING_GUIDE.md`.

For a copy-paste task prompt to start a Codex agent on the H100 instance, see
`H100_AGENT_PROMPT.md`.

Last updated: 2026-07-28

## Objective

Move the best validated native vLLM-Omni EasyMagpie configuration from the
RTX A4500 development host to an H100 host, preserve output quality, tune the
H100-specific kernels, and measure the result at 16 concurrent requests.

The current A4500 profile is a starting point, not an H100 performance claim.

## Source workspace and artifacts

- Repository:
  `/home/siddhartht/tts/speechLM/NeMoDuplexRealtime`
- Branch: `codex/migrate-easymagpie-vllm-omni`
- General experiment history: `CODEX_HANDOFF.md`
- Reproducibility manifest:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_bs16.lock.yaml`
- Operational deploy configuration:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_optimized_bs16.yaml`
- Operational launcher:
  `examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_bs16.sh`
- Cache environment:
  `examples/tts/easymagpie_vllm_omni/scripts/setenv.sh`
- A4500 MoE reference:
  `examples/tts/easymagpie_vllm_omni/moe_configs_a4500/E=24,N=768,device_name=NVIDIA_RTX_A4500.json`
- Benchmark:
  `examples/tts/easymagpie_vllm_omni/scripts/benchmark_server.py`
- Benchmark corpus:
  `examples/tts/easymagpie_vllm_omni/bench_corpus.tsv`
- WER/CER evaluator:
  `examples/tts/easymagpie_vllm_omni/scripts/evaluate_wer_cer.py`

Several implementation and deployment files are currently untracked. Moving
only a normal `git diff` will omit them. Transfer the complete working tree, or
explicitly include every path listed in this document and in
`CODEX_HANDOFF.md`.

## Best controlled A4500 result

Five matched seeds (`20260729` through `20260733`) at `n=16`, `c=16`:

| Variant | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL |
| --- | ---: | ---: | ---: | ---: |
| Shared-expert stream, default tiles | 38.148x | 6.078 | 198.2 ms | 138.7 ms |
| Single stream, default tiles | 43.548x | 6.972 | 184.6 ms | 121.0 ms |
| Single stream, A4500 tiles | **44.984x** | **7.188** | **182.9 ms** | **116.4 ms** |

The complete winner improved mean RTFX by 17.9%. Keeping shared and routed
experts on one stream supplied most of the gain; the A4500-specific tiles added
approximately 3.3% more RTFX.

Quality validation on 10 unique WAVs with Whisper-large-v3 was 5.26% WER and
1.74% CER. Excluding two short proper-name prompts, the other eight samples
measured 2.79% WER and 0.40% CER.

Raw results:

- `nsys_traces/rerun_20260728/`
- `nsys_traces/rerun_20260728/quality_wer_cer.json`
- `nsys_traces/easymagpie_native_optimized_rerun_c16.nsys-rep`

## Latest underrun remeasurement

The running optimized service was remeasured on 2026-07-28 using five warm,
fixed-seed runs at `n=16`, `c=16`:

| Aggregate | Result |
| --- | ---: |
| Successful requests | 80 / 80 |
| Total audio chunks | 1,150 |
| Playback underruns | 80 (6.96%) |
| Requests with an underrun | 80 / 80 |
| Underruns after an 80 ms startup packet | 80 / 80 transitions |
| Underruns after a 480 ms steady packet | **0 / 990 transitions** |
| Mean RTFX | 45.13x |
| Mean request throughput | 7.19 req/s |
| Mean TTFA | 183.1 ms |
| Mean ITL | 116.5 ms |

Every warm request incurred exactly one underrun at the first transition. The
first-to-second packet gaps were approximately 142.0–152.9 ms, exceeding the
80 ms of initial audio by approximately 62–73 ms. The largest observed steady
gap was 225.3 ms, comfortably below the 480 ms steady packet duration.

One preceding idle/cold trial had 1.96 s TTFA but 0 underruns: after the delayed
first response, its first two packets arrived only 33.9 ms apart. It is
reported separately and excluded from the warm aggregate because its startup
state differs materially from normal service operation.

## Changes that should transfer directly to H100

1. `EasyMagpieCudaIpcConnector` keeps the actual `codes.audio` tensor on the
   GPU and transfers it between Stage 0 and Stage 1 with CUDA IPC. Small
   metadata remains in shared memory.
2. The native codec scheduler safely drains already-published, contiguous
   per-request code windows and cohorts ready Stage-1 work.
3. The local transformer uses NeMo-style, frame-local attention caching when
   `EASYMAGPIE_LOCAL_TRANSFORMER_KV_CACHE=1`. K/V and self-attention outputs
   live for one complete
   `num_audio_codebooks * frame_stacking_factor` autoregressive run and reset
   for the next acoustic frame.
4. Gumbel sampling uses the distribution-equivalent `Exp(1)` / `-log(E)`
   formulation, eliminating eager pointwise operations.
5. Request-triggered CUDA profiling records and restores the CUDA device in the
   profiler stop thread and validates the profiler API return codes.

The current zero-underrun packet/cohort settings are:

```yaml
codec_startup_chunk_frames: [2]
codec_chunk_frames: 6
codec_dynamic_chunking: true
codec_dynamic_chunk_wait_us: 5000
codec_dynamic_wait_only_underfilled: true
codec_microbatch_wait_us: 1500
codec_microbatch_parallelism: 16
codec_microbatch_max_batch_size: 16
```

These are safe initial H100 settings. Retune bounded waits only after obtaining
a matched H100 baseline.

## Settings that must be retested on H100

### Shared-expert CUDA stream

The A4500 winner uses:

```bash
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
```

This removes cross-stream rendezvous that were expensive on the A4500 while
Stage 1 competed through MPS. H100 has much more execution capacity and memory
bandwidth, so the auxiliary stream may become beneficial. Run a matched
`0` versus `1` A/B before locking the H100 setting.

### MPS allocation

The A4500 winner leaves Stage 0 uncapped and launches Stage 1 with:

```bash
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=50
```

Retest at least `50`, `75`, and `100` on H100. Use RTFX as the primary
objective, with TTFA, Stage-0/Stage-1 overlap, and underruns as constraints.

### MoE tiles

The JSON keys select Triton fused-MoE GEMM launch parameters:

- `BLOCK_SIZE_M`: token rows per program
- `BLOCK_SIZE_N`: output/intermediate columns per program
- `BLOCK_SIZE_K`: reduction dimension per program
- `num_warps`: participating warps
- `num_stages`: software-pipeline depth

They alter kernel scheduling, not model weights or output semantics.

vLLM selects the file using the exact GPU device name. The A4500 file therefore
will not be selected on H100. For example, an SXM H100 may require:

```text
E=24,N=768,device_name=NVIDIA_H100_80GB_HBM3.json
```

Do not rename the A4500 JSON and treat it as tuned. A4500 is SM86 and H100 is
SM90; their Tensor Core scheduling, shared-memory capacity, occupancy, and
software-pipeline trade-offs differ. With no matching H100 file, vLLM safely
uses its generic configuration.

Tune the actual EasyMagpie MoE shape:

- experts `E=24`
- intermediate/output `N=768`
- hidden size `1536`
- top-k `4`
- FP16 weights/activations
- observed decode batch sizes `1, 2, 4, 6, 8, 13, 16`

Start with the A4500 search dimensions and broaden for H100:

```text
BLOCK_SIZE_M: 16, 32, 64
BLOCK_SIZE_N: 64, 128, 256
BLOCK_SIZE_K: 64, 128, 256
GROUP_SIZE_M: 1
num_warps: 4, 8
num_stages: 3, 4, 5
```

Reject invalid/resource-exhausting candidates and validate winners inside the
full service. A standalone kernel win is only a candidate because Stage 0 and
the FP32 codec share one GPU.

The H100 folder can sit beside the A4500 folder:

```text
examples/tts/easymagpie_vllm_omni/moe_configs_h100/
  E=24,N=768,device_name=<exact_nvidia-smi_device_name>.json
```

Point the launcher at it with:

```bash
VLLM_TUNED_CONFIG_FOLDER=/workspace/examples/tts/easymagpie_vllm_omni/moe_configs_h100
```

After validation, create an H100-specific lock manifest. Do not overwrite the
A4500 manifest.

## Recommended H100 experiment order

1. Set `CACHE_ROOT` to a writable persistent Lustre path, source
   `scripts/setenv.sh`, and leave `TMPDIR` on local `/tmp`.
2. Record `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`,
   CUDA version, image digest, vLLM version, clocks/power mode, and whether the
   H100 is MIG-partitioned.
3. Start from the operational deploy YAML with no user MoE config and warm the
   service fully.
4. Run five fixed-seed `n=16`, `c=16` baselines.
5. A/B `VLLM_DISABLE_SHARED_EXPERTS_STREAM=0/1`.
6. A/B the Stage-1 MPS percentages.
7. Tune H100 MoE tiles for the observed batch sizes, then repeat the same five
   matched seeds.
8. Capture a warmed, request-triggered Nsight Systems trace of the winner.
9. Generate at least 10 unique WAVs and rerun WER/CER before saving the H100
   lock manifest.
10. Run the focused tests:

```bash
pytest -q \
  examples/tts/easymagpie_vllm_omni/tests/test_local_transformer_cache.py \
  examples/tts/easymagpie_vllm_omni/tests/test_runner.py \
  examples/tts/easymagpie_vllm_omni/tests/test_scheduler.py \
  examples/tts/easymagpie_vllm_omni/tests/test_stage_processors.py
```

Benchmark command:

```bash
timeout 120s python3 \
  examples/tts/easymagpie_vllm_omni/scripts/benchmark_server.py \
  --text-file examples/tts/easymagpie_vllm_omni/bench_corpus.tsv \
  -n 16 -c 16 --url http://127.0.0.1:8091 \
  --timeout 100 --no-warmup
```

## Making playback underruns zero

### Validated server-side profile

The active configuration now uses two startup frames, providing approximately
160 ms of initial audio:

```yaml
codec_startup_chunk_frames: [2]
codec_chunk_frames: 6
```

With the redirected caches warm, five fixed-seed `n=16`, `c=16` runs produced
**0 underruns across 1,130 chunks and 80 requests**. There were 0/80 underruns
after the 160 ms startup packet and 0/970 after 480 ms steady packets. Mean
RTFX was 43.42x, mean request throughput was 6.93 req/s, TTFA was 209.6 ms, and
ITL was 120.7 ms.

Relative to the immediately preceding one-frame measurement, this exchanged
approximately 3.9% mean RTFX and 27.8 ms TTFA for zero measured underruns.

### What currently underruns

The previous maximum-RTFX A4500 profile emitted one 80 ms codec frame
immediately, followed by six-frame, approximately 480 ms steady packets.
Approximately 7% of chunks were counted as underruns, but every observed
underrun was the transition after the deliberately short initial packet. There
were zero steady-packet underruns.

The benchmark currently assumes playback starts as soon as the first 80 ms
packet arrives. If the second packet takes more than 80 ms to arrive, playback
necessarily empties even though the service is comfortably faster than real
time overall.

### Optional production playout policy

For additional protection against network or process jitter, do not start the
audio device until queued audio reaches a configurable minimum.

Recommended initial policy:

```text
minimum startup buffer: 480 ms
preferred startup buffer: 640 ms (the 160 ms packet plus one 480 ms packet)
low-water mark: 240 ms
resume threshold after a rare underflow: 480 ms
```

Conceptually:

```python
queue.append(audio_chunk)
buffered_ms += duration_ms(audio_chunk)

if not playing and buffered_ms >= 480:
    start_audio_device()

while playing:
    write_queued_audio_to_device()
```

With the current two-frame startup, waiting for the first steady packet gives
approximately 640 ms of queued audio. This adds roughly one inter-chunk
interval to audible startup latency while leaving transport TTFA unchanged.

Use a jitter-derived threshold in production: collect the maximum or p99.9
steady inter-arrival gap under the intended concurrency, then set
`startup_buffer_ms` above that value with a safety margin. No finite buffer can
mathematically guarantee zero underruns under an unbounded network/process
stall, so also pause and refill to the resume threshold if the low-water mark
is crossed.

### Larger server-side alternative

A server can withhold the first packet until it has generated an even larger
startup window, for example:

```yaml
codec_startup_chunk_frames: [6]
codec_chunk_frames: 6
```

This makes the first emitted packet approximately 480 ms. It is not the
preferred path: earlier large-startup experiments reached zero underruns but
substantially worsened TTFA and RTFX (the `[6, 6]` experiment measured 15.49x
RTFX and 3.75 s TTFA). Client-side playout buffering separates transport TTFA
from audible playout policy and preserves the optimized server schedule.

When comparing policies, report both:

- transport TTFA: arrival time of the first audio bytes;
- audible-start latency: time at which buffered playback actually starts.
