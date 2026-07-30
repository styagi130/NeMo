# H100 C16/C32 low-TTFA experiments

Date: 2026-07-30

## Outcome

No C16 or C32 candidate was promoted. At these smaller capacities, the active
profiles already use a short two-frame startup packet. The `[10,14,16]`
treatment that helps C64/C128 cannot reduce their first-audio boundary.

| Capacity | Active result | Best safe new candidate | Decision |
| --- | --- | --- | --- |
| C16 | 57.280x, 232.00 ms TTFA, 0 underruns | 53.134x, 239.66 ms, 0 underruns | Keep active C16 |
| C32 | 76.087x, 339.67 ms TTFA, 0 underruns | 72.666x, 365.52 ms, 0 underruns | Keep active C32 |

The workspace-plugin transfer path was verified at parallelism 16 and 32,
respectively. `setenv.sh` was sourced before every server, benchmark, and test;
all caches remained under `/workspace/.cache/easymp_h100`, and `TMPDIR`
remained unset.

## C32 measurements

The lowest safe staged candidate uses startup `[3,5]`: 240 ms of first audio,
a 400 ms bridge, then fixed 12-frame / 960 ms steady packets.

| C32 series | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Active `[2]`, six-frame steady | **76.087x** | **12.387** | **339.67 ms** | **129.99 ms** | **0** | **0** |
| Safe `[3,5]`, 12-frame steady | 72.666x | 11.732 | 365.52 ms | 242.39 ms | **0** | **0** |

The safe candidate completed 160/160 matched requests and 1,319 chunks. It
regressed RTFX by 4.50%, TTFA by 25.84 ms, and ITL by 86.47%, so it is not a
TTFA improvement.

Its process-cold run was 37.937x RTFX and 2,992.8 ms TTFA; the transition run
was 71.103x and 404.6 ms. Both had zero underruns/deadlines and are excluded
from the matched aggregate.

Rejected B32 candidates:

- `[2]` with the new fused path averaged 67.504x and 355.0 ms TTFA but caused
  115 matched startup underruns.
- `[3]` with six-frame steady packets retained one warmed underrun and only
  about 65x RTFX.
- `[4]` jumping directly to 12 frames caused 31/32 startup underruns in its
  warmed screen.
- `[4,6]` was playback-safe but slower. Its strict series contained a
  seed-20260732 client wall-time outlier at 53.56x; an exact rerun recovered
  to 73.84x. Both files are retained separately.

The accepted `[3,5]` matched series had no discarded outliers.

## C16 measurements

The safe corrected-path candidate retained the active `[2]` startup and
six-frame steady schedule:

| C16 series | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Active locked profile | **57.280x** | **9.228** | **232.00 ms** | **87.52 ms** | **0** | **0** |
| Safe corrected-path candidate | 53.134x | 8.538 | 239.66 ms | 96.07 ms | **0** | **0** |

The candidate completed 80/80 matched requests and 1,129 chunks, with no
outliers. It regressed RTFX by 7.24% and TTFA by 7.66 ms.

Its process-cold run was 20.569x and 2,882.7 ms TTFA; the transition run was
49.384x and 256.0 ms. Both had zero underruns/deadlines.

The `[1,1,2]` 80 ms ramp reached 220.7 ms TTFA in one warmed screen, but it
caused 16 cold and two transition underruns. It was rejected rather than
substituted for the safe matched result.

## Validation and artifacts

The expanded H100 scaling-profile suite passed 22/22 tests. Quality was not
rerun because neither candidate was promoted and neither active model/codec
path changed.

Raw results:

- `nsys_traces/h100/codec_b32_low_ttfa/results/`
- `nsys_traces/h100/codec_b16_low_ttfa/results/`

Reproducibility manifest:
`examples/tts/easymagpie_vllm_omni/deploy/easymagpie_h100_c16_c32_low_ttfa_experiment.yaml`.

The existing H100 B16 lock, C64 low-TTFA lock, and A4500 artifacts were not
changed.
