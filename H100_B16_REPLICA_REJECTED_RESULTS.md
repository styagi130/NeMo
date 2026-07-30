# H100 B16-replica experiment (rejected)

## Decision

The B16-replica scaling experiment is stopped. It provides a small C64 gain
but regresses sharply at C128 and C192, introduces playback deadline misses,
and produces nonzero underruns at C192. The existing B32-pool profiles remain
the validated deployment choices.

All matched measurements use seeds 20260729 through 20260733:

| Concurrency | B16 layout | Mean RTFX | B32 mean | Change | Req/s | TTFA | ITL | Underruns | Deadlines |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4x B16 | 134.201x | 128.573x | +4.4% | 21.887 | 635 ms | 270 ms | 0 | 0 |
| 128 | 8x B16 | 148.831x | 214.917x | -30.8% | 24.198 | 914 ms | 503 ms | 0 | 27 |
| 192 | 12x B16 | 153.173x | 255.544x | -40.1% | 25.217 | 1,269 ms | 741 ms | 3 | 897 |
| 256 | 16x B16 | no startup | 270.711x | rejected | - | - | - | - | - |

C64's initial post-cold seed 20260729 measured 109.37x with four deadline
misses. It is retained as a warm-transition outlier. The exact-seed hot rerun
measured 136.39x with zero underruns and deadlines and is used in the matched
aggregate above.

## Cold runs

| Concurrency | Cold RTFX | TTFA | Underruns | Deadlines |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 64.02x | 3,869 ms | 0 | 40 |
| 128 | 100.58x | 3,625 ms | 0 | 0 |
| 192 | 110.84x | 4,673 ms | 0 | 125 |

Cold measurements are not mixed into the warmed aggregates.

## Diagnosis

Splitting B32 autoregressive cohorts into twice as many B16 engines duplicates
model state, CUDA contexts, schedulers, graph pools, and launch work. At C128
and C192, additional engine contention costs more than the smaller per-engine
cohort saves. This is compute/launch contention, not admission serialization:
all requests completed successfully.

The compiled C256 16xB16 attempt at 0.06 Stage-0 GPU utilization failed on the
first engine with 0.0 GiB available KV cache. Disabling CUDA graphs did not
make 0.06, 0.07, or 0.08 allocations viable; each eager attempt also reported
0.0 GiB available KV cache. The experiment was stopped without attempting
larger overlapping quotas because C128 and C192 had already disproved the
throughput hypothesis.

All server and benchmark processes sourced `setenv.sh`; caches remained under
`/workspace/.cache/easymp_h100`, and `TMPDIR` remained unset.

Raw results are under:

- `nsys_traces/h100/codec_b64_b16x4/results/`
- `nsys_traces/h100/codec_b128_b16x8/results/`
- `nsys_traces/h100/codec_b192_b16x12/results/`

The B16 configs and launchers are retained only to reproduce the rejected
experiment. Existing B32 and A4500 lock manifests were not modified.
