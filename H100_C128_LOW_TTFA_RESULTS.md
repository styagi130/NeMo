# H100 C128 low-TTFA result

Date: 2026-07-29

## Outcome

The selected four-B32-replica C128 profile reduces matched mean TTFA from
885.1 ms to **747.5 ms** while preserving mean throughput at **214.76x RTFX**.
Across five matched seeds it completed 640/640 requests and 4,207 audio chunks
with **zero underruns and zero deadline misses**.

| C128 profile | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Locked B32 baseline, startup `[16]`, MPS100 | 214.917x | 34.906 | 885.1 ms | 311.7 ms | 0 | 0 |
| Selected low-TTFA, startup `[10,14,16]`, MPS75 | **214.760x** | **35.040** | **747.5 ms** | 338.5 ms | **0** | **0** |

The matched change is -137.6 ms TTFA (-15.54%), -0.073% RTFX, +0.386%
request throughput, and +26.7 ms ITL. The ITL increase reflects the smaller
early packets and catch-up cadence; it did not create a playback failure. The
minimum per-run p05 playback headroom remained 322.0 ms.

The startup schedule emits 800, 1,120, and 1,280 ms packets at frame boundaries
10, 24, and 40, then returns to the locked 12-frame cadence at 52, 64, and so
on. It therefore rejoins the baseline boundary at frame 40 and never exceeds
the already validated 16-frame / 2,048-token Stage-1 budget.

## Transfer serialization found and removed

Several apparent 154-163x regressions were not properties of the packet
schedule. The container image has an older
`/usr/local/lib/python3.12/dist-packages/vllm_plugin_easymagpie_omni` copy that
does not install transfer-only parallelism. Runs that imported it showed no
transfer-microbatch startup messages and serialized work through the old
path.

The low-TTFA launcher now prepends the transferred workspace package to
`PYTHONPATH`. A valid launch prints exactly four messages, one per B32 Stage-0
replica:

```text
EasyMagpie codec transfer microbatching enabled: wait=1.50ms parallelism=32
```

The model must also be supplied by its absolute in-container path:

```text
/workspace/examples/tts/easymagpie_vllm_omni/easymp_vllm_model
```

Together these checks prevent both admission and transfer serialization. Do
not compare the stale-plugin controls to the matched series.

## Matched measurements

Seeds 20260729 through 20260733, each at 128 requests and concurrency 128:

| Seed | RTFX | req/s | TTFA | ITL | p05 headroom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260729 | 222.589x | 33.409 | 798.7 ms | 345.3 ms | 366.7 ms |
| 20260730 | 217.542x | 35.017 | 718.7 ms | 350.1 ms | 327.8 ms |
| 20260731 | 204.852x | 34.706 | 759.2 ms | 336.0 ms | 356.7 ms |
| 20260732 | 212.745x | 35.937 | 732.2 ms | 331.4 ms | 322.0 ms |
| 20260733 | 216.073x | 36.133 | 728.8 ms | 329.5 ms | 332.3 ms |

No matched sample was discarded. Seed 20260731 is retained as the minimum
RTFX measurement; there are no separated matched outliers.

Raw results:
`nsys_traces/h100/codec_b128_low_ttfa/results/c128_startup10_14_16_mps75_parallel_matched_seed*.json`.

## Cold and transition runs

These are reported separately and are excluded from the matched aggregate:

| State | RTFX | req/s | TTFA | ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Process-cold | 132.443x | 20.680 | 3268.6 ms | 318.3 ms | 0 | 0 |
| Transition warmup | 224.968x | 35.656 | 686.5 ms | 322.1 ms | 0 | 0 |

“Process-cold” means a fresh server process while preserving the required
redirected caches. Earlier experiments had already exercised some codec
shapes; it is not a cache-deletion experiment.

## Candidate and rejected schedules

| Schedule | Mean RTFX | Mean TTFA | Mean ITL | Result |
| --- | ---: | ---: | ---: | --- |
| `[14,14]` | 213.149x | 840.0 ms | 325.1 ms | Safe, smaller TTFA gain |
| `[12,16]` | 214.156x | 750.8 ms | 341.0 ms | Safe, slightly below selected |
| `[10,14,16]` | **214.760x** | **747.5 ms** | 338.5 ms | Selected |
| `[1,1,2,4,8,12]` | 135.62x warmed | 552 ms warmed | — | Rejected: 425 underruns |

The aggressive ramp demonstrates why a 500 ms C128 target is not accepted
under the zero-underrun and unchanged-RTFX constraints: its first packets are
shorter than the following codec gaps. GPU clock locking at 1,785 MHz did not
improve the controlled result and was reverted with `nvidia-smi -rgc`.

## Startup, playback, tests, and quality

- Matched service: 640/640 successful, zero startup or steady-state
  underruns, zero deadline misses.
- Process-cold service: 128/128 successful, zero underruns, zero deadline
  misses.
- Focused profile tests: 19/19 passed.
- Combined cache, runner, scheduler, Stage processor, and profile tests:
  46/46 passed.
- New ten-utterance generation check: 10/10 successful, zero underruns and
  zero deadline misses.
- Whisper-large-v3 on those ten outputs: **4.21% WER, 2.30% CER**.

Quality result:
`nsys_traces/h100/codec_b128_low_ttfa/c128_low_ttfa_whisper_large_v3_wer_cer.json`.

## Reproduction contract

Source the cache environment before launching the server, benchmark, tests,
or WER/CER evaluation. It redirects compile/download caches but deliberately
leaves `TMPDIR` unchanged:

```bash
cd /workspace
before_tmp=${TMPDIR-__UNSET__}
CACHE_ROOT=/workspace/.cache/easymp_h100 \
  source examples/tts/easymagpie_vllm_omni/scripts/setenv.sh
test "$before_tmp" = "${TMPDIR-__UNSET__}"

examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_h100_c128_low_ttfa.sh \
  /workspace/examples/tts/easymagpie_vllm_omni/easymp_vllm_model 8091
```

The selected deployment configuration is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_optimized_h100_c128_stage0x4_low_ttfa.yaml`.
Its reproducibility lock is
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c128_low_ttfa.lock.yaml`.

The A4500 tile and lock artifacts, prior C64/C128/C192/C256 locks, shared
scaling launcher, H100 MoE tile, and H100 Mamba SSU tile were not changed.
