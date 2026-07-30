# H100 B32-replica scaling results

## Outcome

The single-H100 service now has validated C128, C192, and C256 profiles that
retain one B32 autoregressive cohort per Stage-0 replica:

| Concurrency | Layout | Mean RTFX | Req/s | TTFA | ITL | Underruns | Deadline misses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2x B32 -> codec C64 | 128.573x | 21.081 | 782 ms | 268 ms | 0 | 0 |
| 128 | 4x B32 -> codec C128 | 214.917x | 34.906 | 885 ms | 312 ms | 0 | 0 |
| 192 | 6x B32 -> codec C192 | 255.544x | 41.993 | 1,067 ms | 388 ms | 0 | 0 |
| 256 | 8x B32 -> codec C256 | 270.711x | 44.310 | 1,321 ms | 491 ms | 0 | 0 |

All rows use the same five seeds, 20260729 through 20260733. C128 completed
640/640 requests, C192 completed 960/960, and C256 completed 1,280/1,280.
There were no failed requests, underruns, deadline misses, or discarded
matched outliers.

C192 is the best latency/throughput tradeoff. C256 is the maximum measured
aggregate throughput, but adding the seventh and eighth B32 replicas improves
RTFX by only 5.94% over C192 while mean TTFA rises 23.8% and mean ITL rises
26.5%.

## Serialization fix

The Stage-0 fix scales directly: use four, six, or eight independent B32
engines instead of widening one autoregressive scheduler.

B128 also exposed a second gate. Stage 1 retained a 512-token global scheduling
budget, but a full C128 startup cohort contains 128 requests times 16 frames,
or 2,048 placeholders. Balanced uninstrumented arrivals often landed as exact
32-request blocks; profiler jitter mixed startup and steady requests and caused
one 16-frame payload to be sliced to 12 scheduled placeholders. The model
correctly rejected that mismatch.

The accepted profiles set Stage-1 `max_num_batched_tokens` to the complete
startup requirement:

- C128: 2,048
- C192: 3,072
- C256: 4,096

After this correction, the observed codec maxima were B127, B120, and B159.
The fact that asynchronous producers do not always arrive in one scheduler
interval is not a capacity serialization; forcing every codec launch to the
configured maximum was already shown unsafe in the C64 work.

Admission was exactly balanced during each cold-plus-matched set: every Stage-0
replica handled 192 requests. Engine construction itself is sequential in
vLLM-Omni, but this is startup control-plane behavior and is reported
separately from the concurrent request pipeline.

## Cold runs

| Concurrency | Cold RTFX | Req/s | TTFA | ITL | Underruns |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128.969x | 20.161 | 3,592.6 ms | 295.6 ms | 0 |
| 192 | 157.279x | 25.327 | 4,107.2 ms | 384.6 ms | 0 |
| 256 | 180.348x | 29.052 | 4,401.6 ms | 482.4 ms | 0 |

A separate post-restart C128 reproduction measured 135.040x with 3,485.2 ms
TTFA and zero underruns. It is classified as cold and is not mixed into the
warmed matched aggregate.

## Saturation and profiling

The corrected warmed C128 Nsight trace records 739.09 ms aggregate GPU kernel
time. Packed causal Conv1D is the largest codec kernel at 314.854 ms (42.6%);
fused ConvTranspose1d is only 11.972 ms (1.6%). CUDA API time is led by
`cudaMemcpyAsync` (63.7%), `cudaStreamSynchronize` (21.3%), and kernel launch
(8.4%). The instrumented request measured 170.89x and is excluded from
performance aggregates.

During an uninstrumented C256 run, the H100 stayed at 99-100% GPU utilization
and approximately 391-400 W. Memory-controller utilization ranged from 16-50%.
The C256 plateau is therefore compute/power saturation, not HBM saturation or
another request-capacity gate.

Stage-1 MPS 75% was rejected at C128. Across exact seeds 20260729 and 20260730,
it averaged 220.280x versus 223.129x at 100% (-1.28%). Its 130.98x cold result
is retained separately.

## Quality, tests, and cache contract

The focused integration suite passed 53/53 tests. Ten WAVs from the final
eight-replica code path scored 4.21% WER and 1.19% CER with Whisper large-v3.

Every server, benchmark, test, and quality command sourced `setenv.sh`. Caches
remained under `/workspace/.cache/easymp_h100`, and `TMPDIR` remained unset.

The A4500 lock and tile remain byte-identical at SHA-256
`cf2e84eb8a00b8ca0e712339bd8fc75439bf1a58736ecdf89072f134a1ad2a41`
and
`ebea4cb146da1c19f3eddb1e361255d5949e4ad8b69b1509ffa2c09626e6b2fa`.
The C64 lock and shared scaling launcher also remain byte-identical at
`f43fce99633cd6a34f4c314286b93f92824646001a07e9d74fb25ecd87e56112`
and
`f1a149e70f0295d826afbf4b6f9c6c8748da77d1d06b72d0622803c27d8fc8d1`.

## Locked artifacts

- C128 lock:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c128.lock.yaml`
- C192 lock:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c192.lock.yaml`
- C256 lock:
  `examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c256_stage0x8.lock.yaml`
- Warmed C128 trace:
  `nsys_traces/h100/codec_b128/profile/codec_b128_stage0x4_token2048_warm_20260729.nsys-rep`
- C256 utilization samples:
  `nsys_traces/h100/codec_b256/results/b256_stage0x8_gpu_samples.csv`
- Quality:
  `nsys_traces/h100/codec_b256/quality_stage0x8_wer_cer.json`
