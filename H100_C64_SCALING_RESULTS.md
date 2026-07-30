# H100 all-B64 optimization result

## Low-TTFA 2xB32 update (2026-07-30)

The C128 startup treatment was applied to the validated C64 topology without
changing its two same-GPU B32 Stage-0 replicas. The separate low-TTFA profile
uses startup packets `[10,14,16]`, retains Stage-1 MPS at the C64 winner's
100%, raises the Stage-1 scheduling budget to the complete 64x16 = 1,024-token
startup cohort, and forces the transferred workspace plugin to the front of
`PYTHONPATH`.

Five matched seeds completed 320/320 requests and 2,098 chunks at **130.631x
mean RTFX**, **21.393 requests/s**, **595.58 ms mean TTFA**, and 286.74 ms mean
ITL. There were zero underruns and zero deadline misses. Compared with the
locked `[16]` C64 result, TTFA improved by 186.57 ms (-23.85%), RTFX improved
by 1.60%, and request throughput improved by 1.48%.

| C64 2xB32 profile | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Locked `[16]` | 128.573x | 21.081 | 782.15 ms | 268.30 ms | 0 | 0 |
| Low-TTFA `[10,14,16]` | **130.631x** | **21.393** | **595.58 ms** | 286.74 ms | **0** | **0** |

The process-cold run is separate: 75.892x RTFX, 11.471 requests/s, 3,251.98 ms
TTFA, and zero underruns/deadlines. The transition cohort measured 138.372x
RTFX and 579.00 ms TTFA, also with zero underruns/deadlines. No matched outlier
was removed.

The new lock is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64_low_ttfa.lock.yaml`.
Raw matched measurements are under
`nsys_traces/h100/codec_b64_low_ttfa/results/`. The earlier C64 lock remains
unchanged.

## Final Stage-0x2 update

The current H100 C64 winner uses two same-GPU Stage-0 replicas, each capped at
B32, feeding one Stage-1 codec with B64 capacity. Five matched seeds completed
320/320 requests at **128.573x mean RTFX** (129.338x median,
121.824–136.494x range), 21.081 requests/s, 782.15 ms mean TTFA, and 268.30 ms
mean ITL. All 2,080 chunks had zero underruns and zero deadline misses. The
separately measured final cold run was 75.323x RTFX.

The observed codec batch maximum was B42. Stage 1 remains capacity-matched at
B64, but the two asynchronous Stage-0 replicas do not always publish in the
same scheduler interval. A forced 20 ms B64 aggregation was rejected after
timing out 7/64 cold requests; it is reported separately.

The warmed winner trace is
`nsys_traces/h100/codec_b64/profile/codec_b64_stage0x2_warm_20260729.nsys-rep`.
It records 336.430 ms of aggregate GPU kernel work. Packed causal Conv1D is the
largest codec kernel at 25.4%; fused ConvTranspose1d is only 1.7%. CUDA API
time is led by async copies (67.4%), launches (14.8%), and stream
synchronization (10.1%), confirming a host/launch-orchestration bottleneck.

Ten final-layout WAVs scored 3.68% WER and 2.85% CER with Whisper large-v3.
The focused final suite passed 47/47 tests. The exact deployment is locked in
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64.lock.yaml`.

The single-Stage-0 results below are retained as historical measurements.

The final service now keeps every relevant capacity at 64:

- request admission: B64;
- Stage-0 connector publication: 64 parallel streams;
- Stage-1 steady codec cohort: B64.

It completes five independent matched seeds at **69.678x mean RTFX**, with
zero underruns. This is 3.369% faster than the original serialization-free
codec-B64 baseline and 0.581% faster than the earlier B16-codec partition.
The H100-only launcher routes uniform Stage-1 batches through the fused Triton
ConvTranspose1d kernel and keeps the best fused Stage-1 MPS cap at 100%.

The fused kernel removes the measured ConvTranspose bottleneck, but the exact
end-to-end set is 2.824% below the prior cuDNN winner (71.703x). The fused
profile is locked because it was explicitly selected, not because it is the
highest end-to-end RTFX configuration.

## Final matched result

Each run contains 64 requests at concurrency 64 after a cold run and separate
warmup.

| Seed | RTFX | Requests/s | TTFA | ITL | Underruns | Deadline misses |
|---:|---:|---:|---:|---:|---:|---:|
| 20260729 | 71.182x | 10.755 | 879.3 ms | 316.5 ms | 0 | 3 |
| 20260730 | 72.635x | 11.383 | 795.1 ms | 309.7 ms | 0 | 0 |
| 20260731 | 67.825x | 12.402 | 854.8 ms | 278.9 ms | 0 | 0 |
| 20260732 | 68.516x | 11.241 | 831.0 ms | 305.9 ms | 0 | 0 |
| 20260733 | 68.232x | 11.068 | 852.1 ms | 304.9 ms | 0 | 0 |
| **Aggregate** | **69.678x mean / 68.516x median** | **11.370** | **842.5 ms** | **303.2 ms** | **0 / 4,117 chunks** | **3** |

The separately measured cold run was 47.532x RTFX, 8.060 requests/s,
3,382.6 ms TTFA, 278.1 ms ITL, and zero underruns or deadline misses.

## Nsight diagnosis

The warmed baseline trace showed that the fallback packed causal Conv1D was
the dominant GPU kernel:

| Warmed trace | Relevant GPU time | Kernel-time share |
|---|---:|---:|
| Baseline packed Conv1D | 398.719 ms | 46.7% |
| TF32 packed Conv1D | 79.530 ms | 14.6% |
| cuDNN ConvTranspose dgrad before fusion | 170.680 ms | 31.4% |
| Fused ConvTranspose1d | 6.032 ms | 1.7% |

The fused kernel combines state gathering, grouped deconvolution, and output
layout. At the four real decoder shapes its isolated warmed B64 speedup over
cuDNN is 3.3x–14.2x, with maximum FP32 error `4.77e-7`. In the service trace,
its aggregate GPU time is 96.466% below the prior cuDNN dgrad time.

The next GPU bottleneck is packed causal Conv1D at 29.4% of traced GPU kernel
time. CUDA API host time remains led by asynchronous copies (43.4%), kernel
launches (15.3%), and stream synchronization (10.5%).

## Scheduler finding

The uncapped codec scheduler previously drained queued six-frame successors
into per-request windows as large as 960–3,840 ms. This created head-of-line
stalls and warmup underruns even when aggregate RTFX was high.

The final C64 profile sets `codec_fixed_chunk_frames: 6`. This retains the
normal 480 ms codec cadence and prevents successor merging without reducing
the B64 cohort or serializing the pipeline. Observed packet durations are only
480 ms steady and 640 ms startup.

## Baselines and outliers

| Profile | Mean RTFX | Median | Range | Underruns |
|---|---:|---:|---:|---:|
| Original codec B64 | 67.407x | 66.521x | 66.034–69.791x | 0 |
| First optimized set | 69.718x | 69.996x | 58.006–77.654x | 0 |
| cuDNN final set | 71.703x | 71.465x | 69.634–74.177x | 0 |
| Fused, MPS 50% tuning screen | 68.938x | 68.205x | 66.899–70.922x | 0 |
| Fused, MPS 75% exact | 69.477x | 69.588x | 67.998–70.603x | 0 |
| **Fused, MPS 100% exact** | **69.678x** | **68.516x** | **67.825–72.635x** | **0** |

The MPS 50% screen and the earlier `29–33` screens used different corpus RNG
seeds and are not compared to the locked exact-seed runs.

The first optimized set's seed 20260732 result was a 58.006x external
wall-time outlier: per-request latency and deadline metrics were normal, and
an exact-seed rerun recovered to 68.811x. It remains in its original strict
aggregate and is reported separately rather than discarded.

Nsight-instrumented requests are also excluded from performance aggregates.
The fused trace had 245/860 underruns from instrumentation overhead.

## Quality and validation

Ten new WAVs generated by the fused service scored 3.16% WER and 1.84% CER
with Whisper large-v3. The fused/default CUDA integration matrix passed all
18 focused tests; the previously completed broader suite remains at 38
focused and 8 benchmark/metric tests.

`setenv.sh` was sourced before server and benchmark launches. All JIT,
compile, and download caches remained under
`/workspace/.cache/easymp_h100`; `TMPDIR` remained unset.

## Artifacts

- Final launcher:
  `examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_h100_c64.sh`
- Final lock:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64.lock.yaml`
- Baseline trace:
  `nsys_traces/h100/codec_b64/profile/codec_b64_warm_20260729.nsys-rep`
- Optimized trace:
  `nsys_traces/h100/codec_b64/profile/codec_b64_tf32_nodrain_warm_20260729.nsys-rep`
- Fused trace:
  `nsys_traces/h100/codec_b64/profile/codec_b64_fused_deconv_mps75_warm_20260729.nsys-rep`
- Fused quality result:
  `nsys_traces/h100/codec_b64/quality_fused_deconv_wer_cer.json`

The A4500 lock and tile remain byte-identical at SHA-256
`cf2e84eb8a00b8ca0e712339bd8fc75439bf1a58736ecdf89072f134a1ad2a41`
and
`ebea4cb146da1c19f3eddb1e361255d5949e4ad8b69b1509ffa2c09626e6b2fa`.
