# EasyMagpie H100 optimization results

Measured on 2026-07-28 on one full NVIDIA H100 NVL (95,830 MiB, MIG
disabled), the selected B16 configuration is:

```text
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
VLLM_TUNED_CONFIG_FOLDER=.../moe_configs_h100
EASYMAGPIE_CODEC_MPS_ACTIVE_THREAD_PERCENTAGE=75
codec_startup_chunk_frames=[2]
codec_chunk_frames=6
```

Across the five fixed seeds `20260729`–`20260733`, this configuration completed
80/80 requests at 57.280x mean RTFX (57.19x median, 51.76x minimum, 62.39x
maximum) and 9.228 mean requests/s (9.28 median, 8.43 minimum, 9.85 maximum).
Mean TTFA was 232.00 ms and mean ITL was 87.52 ms. There were zero failures,
zero deadline misses, zero startup underruns across 80 startup transitions, and
zero steady-state underruns across 953 steady transitions.

The exact reproducibility manifest is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_bs16.lock.yaml`.

## Strict matched experiments

All comparisons below used the same image, model, B16 workload, corpus, packet
schedule, cache root, and five seeds.

| Experiment | Stream | Tiles | Stage-1 MPS | Mean/median RTFX | Mean req/s | Mean TTFA | Mean ITL | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Shared-stream A | 0 | default | 50% | 48.832 / 51.80 | 7.970 | 230.02 ms | 99.14 ms | 80/80 |
| Shared-stream B | 1 | default | 50% | 42.720 / 53.28 | 6.696 | 231.08 ms | 89.72 ms | 79/80 |
| H100 tiles | 0 | H100 | 50% | 57.190 / 57.77 | 9.190 | 243.50 ms | 88.64 ms | 80/80 |
| MPS winner | 0 | H100 | 75% | 57.280 / 57.19 | 9.228 | 232.00 ms | 87.52 ms | 80/80 |
| MPS alternative | 0 | H100 | 100% | 57.088 / 58.06 | 9.240 | 237.04 ms | 89.18 ms | 80/80 |

Stream 0 won because stream 1 had one request remain stuck until client
cancellation after 150 seconds. The strict five-seed stream-1 mean includes
that failed seed; it is not silently discarded.

With stream 0 and MPS 50% held constant, H100 tiles improved strict mean RTFX
by 17.12% and median RTFX by 11.53% over vLLM defaults. The tuned-tile five-run
series had no deadline misses, while the default series had 13.

MPS 50%, 75%, and 100% were close. MPS 75% led mean RTFX by 0.16% over 50%
and 0.34% over 100%, and also had the best mean TTFA/ITL. This small difference
should be treated as a measured tie-break, not a large architectural effect.

## Cold, warmup, and outlier observations

These runs are deliberately excluded from the strict matched aggregates:

| Class | Observation |
| --- | --- |
| Cold first request path | 8.90x RTFX, 1.25 req/s, 7,671 ms TTFA, 16/16 startup underruns while request-path kernels compiled. |
| First warmup | 43.67x RTFX, 16 startup underruns; compilation/settling was still visible. |
| Isolated warmup outlier | 8.00x RTFX with a 9,849.9 ms maximum steady gap and 12 steady underruns despite no new JIT warning. |
| Matched default outliers | Seeds 20260729 and 20260730 had 810.8 ms and 1,626.8 ms maximum steady gaps. Buffer headroom prevented underruns, but they caused 11 and 2 deadline misses. |
| Stream-1 outlier/failure | Seed 20260733 produced 15/16 successes; one request was cancelled after 150 seconds. |
| Nsight-instrumented run | 36.67x RTFX and one 160 ms startup underrun under profiler overhead; steady state remained 0/212. |

The unprofiled selected configuration is therefore the zero-underrun result.
The instrumented trace is useful for attribution but is not substituted into
the uninstrumented performance aggregate.

## Post-lock concurrency scaling

The same locked H100 configuration was subsequently measured at concurrency 32
and 64. Shape-compilation runs were excluded, a second 32/64 warm-check
produced no new JIT warnings, and the five fixed seeds were then repeated at
each level.

| Concurrency | Requests | Mean/median RTFX | Mean req/s | Mean TTFA | Mean ITL | Startup underruns | Steady underruns | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 160 | 53.666 / 53.16 | 8.688 | 1,046.06 ms | 103.72 ms | 5/160 | 0/1,915 | 0 |
| 64 | 320 | 51.930 / 51.64 | 8.486 | 2,814.82 ms | 117.18 ms | 55/320 | 0/3,793 | 0 |

All requests succeeded and all 480 ms steady transitions remained
underrun-free. The 160 ms startup packet is too small for every queued request
at these concurrency levels: startup underruns affected 3.1% of requests at
concurrency 32 and 17.2% at concurrency 64. The raw logs and machine-readable
aggregate are under `nsys_traces/h100/results/`.

### Experimental 32-wide admission profile

Both stage `max_num_seqs` values and the capacity-coupled codec transfer/cohort
limits were raised from 16 to 32 in a separate H100 profile. This removed the
original 16-request wave: C32 became one admission band and C64 became two
32-request bands.

| Concurrency | Mean/median RTFX | Mean req/s | Mean TTFA | Mean ITL | Startup underruns | Steady underruns |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 77.718 / 78.70 | 12.652 | 326.12 ms | 129.56 ms | 9/160 | 0/1,892 |
| 64 | 69.558 / 70.03 | 11.336 | 1,604.68 ms | 162.16 ms | 164/320 | 0/3,801 |

Relative to the 16-wide capacity profile, C32 throughput improved 45.63% and
mean TTFA fell 68.82%; C64 throughput improved 33.58% and mean TTFA fell
42.99%. The tradeoff is slower inter-packet cadence: ITL increased 24.91% at
C32 and 38.39% at C64. The 160 ms startup packet consequently underruns on
5.6% of C32 requests and 51.3% of C64 requests, although every steady 480 ms
transition remains underrun-free.

This profile is saved for further startup-buffer, MPS, and batch-32 MoE tile
tuning but has not replaced the zero-underrun B16 lock.

## Cache and environment

The host has no Lustre mount. All CUDA, Triton, TorchInductor, XDG, PyTorch,
Hugging Face, vLLM, FlashInfer, and speaker caches were redirected to the
persistent local ext4 path `/workspace/.cache/easymp_h100`. Startup logs
confirmed that both vLLM compile caches and the H100 MoE JSON were loaded from
the redirected paths. `TMPDIR` remained unset on the host and in the service
configuration.

The first cold initialization took 181.13 seconds, including 106.23 seconds of
compilation. With the redirected cache populated, subsequent startup loaded
the Stage-0 compiled graph in 0.729 seconds and the local-transformer graph in
1.417 seconds.

## Warmed Nsight trace

The trace was armed only after a full B16 profiler-context warmup followed by a
second 56.44x RTFX, zero-underrun warm-check with no new inference JIT
warnings. The request-triggered `cudaProfilerApi` range captured the next B16
batch:

```text
nsys_traces/h100/h100_warm_bs16_mps75.nsys-rep
```

GPU-kernel time leaders were packed causal Conv1D (8.5%), FlashAttention
(7.9%), cuDNN dgrad (6.3%), and fused MoE (5.5%). NVTX time was dominated by
Stage-1 IPC event waiting (53.6%). CUDA API host time was led by asynchronous
copies (54.3%), followed by kernel launches (14.4%), stream synchronization
(14.1%), and graph launches (11.5%). These percentages describe the
instrumented run and should not be interpreted as uninstrumented wall-time
fractions.

## Quality and tests

Ten unique steady-state WAVs were generated with 10/10 success, zero
underruns, and zero deadline misses. Whisper-large-v3 measured 3.16% WER and
1.84% CER. The A4500 baseline was 5.26% WER and 1.74% CER, so H100 WER
improved while CER changed by +0.10 percentage point. Excluding the same two
short proper-name prompts (`utt01`, `utt10`) gave 0.56% WER and 0.20% CER,
versus the A4500 baseline of 2.79% and 0.40%.

The final focused suite passed 26/26 tests. The four benchmark underrun-metric
tests also passed.

## Saved artifacts

- H100 tile JSON:
  `examples/tts/easymagpie_vllm_omni/moe_configs_h100/E=24,N=768,device_name=NVIDIA_H100_NVL.json`
- H100 lock:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_bs16.lock.yaml`
- Raw benchmark results: `nsys_traces/h100/results/`
- Server and tuning logs: `nsys_traces/h100/logs/`
- Nsight report and SQLite export: `nsys_traces/h100/`
- WAVs and WER/CER JSON: `nsys_traces/h100/quality/`

The pre-existing A4500 lock and tile were not edited. Their final SHA-256
values remain `cf2e84eb...ad2a41` and `ebea4cb1...e6b2fa`, respectively.

The H100 advantage expected from Hopper-specific tensor cores and memory
bandwidth was not isolated as an architecture-only experiment. Only the
stream, tile, and Stage-1 MPS effects listed above were measured.
