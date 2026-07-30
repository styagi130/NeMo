# H100 C64 low-TTFA listening guide

## Listen to the generated samples

These ten samples were generated together as one concurrent 10-request cohort
against the real H100 C64 low-TTFA service. The service was configured for 64
admitted requests, with two B32 autoregressive replicas feeding one C64 codec.
The listening cohort is C10 because the corpus contains ten distinct prompts;
its timing is therefore not a C64 performance benchmark.

| ID | Prompt | Duration | Audio |
| --- | --- | ---: | --- |
| utt01 | Mrs. De Mohrenschildt thought that Oswald, | 2.48 s | [Listen to utt01](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt01.wav) |
| utt02 | The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent. | 8.32 s | [Listen to utt02](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt02.wav) |
| utt03 | Between the hours of eight and nine p.m. they were occupied with the children in the bedrooms located at the extreme east end of the house. | 6.96 s | [Listen to utt03](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt03.wav) |
| utt04 | The prisoner had nothing to deal with but wooden panels, and by dint of cutting and chopping he got both the lower panels out. | 7.28 s | [Listen to utt04](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt04.wav) |
| utt05 | Under these circumstances, unnatural as they are, with proper management, the bean will thrust forth its radicle and its plumule; | 7.28 s | [Listen to utt05](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt05.wav) |
| utt06 | Oswald demonstrated his thinking in connection with his return to the United States by preparing two sets of identical questions of the type which he might have thought | 8.64 s | [Listen to utt06](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt06.wav) |
| utt07 | it is not possible to state with scientific certainty that a particular small group of fibers come from a certain piece of clothing | 8.00 s | [Listen to utt07](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt07.wav) |
| utt08 | has confidence in the dedicated Secret Service men who are ready to lay down their lives for him | 5.28 s | [Listen to utt08](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt08.wav) |
| utt09 | Since these agencies are already obliged constantly to evaluate the activities of such groups, | 5.52 s | [Listen to utt09](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt09.wav) |
| utt10 | Jeanne De Mohrenschildt said, quote, | 1.84 s | [Listen to utt10](nsys_traces/h100/codec_b64_low_ttfa/listening_samples_seed20260735/utt10.wav) |

All files are 22,050 Hz, 16-bit, mono PCM WAVs.

## What “C64 with a B32 layout” means

```text
                         one H100
64 admitted requests
        |
        +--> Stage 0 replica A: up to B32 --+
        |                                   +--> Stage 1 codec: up to B64 --> streamed WAV
        +--> Stage 0 replica B: up to B32 --+
             32 parallel transfers each
```

Stage 0 predicts codec tokens. Splitting it into two independent B32 engines
lets the H100 process two request pools concurrently instead of forcing one
large B64 autoregressive step. Stage 1 combines the available work and decodes
up to 64 streams into audio.

The important distinction is that **capacity** is C64 while the Stage-0
execution layout is **2xB32**. During this sample run, ten requests used that
same deployed service in parallel.

## Improvements and why they help

### 1. Removed hidden transfer serialization

The container had an older EasyMagpie package in `site-packages`. If that copy
was imported first, Stage-0-to-codec transfers serialized and C64/C128 scaling
dropped sharply. The launcher now puts the transferred workspace package first
on `PYTHONPATH`.

A correct C64 startup prints two independent messages:

```text
EasyMagpie codec transfer microbatching enabled: wait=1.50ms parallelism=32
EasyMagpie codec transfer microbatching enabled: wait=1.50ms parallelism=32
```

Both messages were observed for this generation run: one for each B32 Stage-0
replica.

### 2. Lowered TTFA with a safe startup ramp

The old profile waited for one 16-frame, 1,280 ms startup packet. The low-TTFA
profile emits startup packets of 10, 14, and 16 codec frames, corresponding to
800, 1,120, and 1,280 ms of audio. This makes the first playable audio
available earlier, while the larger follow-up packets preserve enough playback
buffer to prevent underruns.

After the ramp, the codec uses fixed 12-frame, 960 ms packets. Fixed packet
size prevents the scheduler from merging many successors into a large
per-request window, which previously caused head-of-line stalls.

### 3. Removed a Stage-1 scheduling gate

At C64, a full 16-frame codec cohort contains `64 x 16 = 1,024` tokens. Stage
1 now has a 1,024-token scheduler budget, so a complete startup cohort fits in
one scheduling step instead of being split by a narrower global limit.

### 4. Applied H100-specific execution paths

The launcher enables:

- the H100-tuned MoE and Mamba kernel tiles;
- TF32 packed causal convolution;
- fused ConvTranspose1d;
- direct CUDA IPC control between stages;
- 100% Stage-1 MPS active-thread allocation.

The warmed Nsight trace showed that fused ConvTranspose1d reduced its aggregate
GPU time from 170.680 ms to 6.032 ms. Packed causal Conv1D then became the
largest codec kernel. At this topology, host launch/copy orchestration is also
a material limiter, which is why removing transfer serialization and using two
B32 producers matters.

### 5. Kept caches safe without touching `TMPDIR`

Before both server launch and generation, `setenv.sh` redirected compile, JIT,
CUDA, Hugging Face, vLLM, and FlashInfer caches to
`/workspace/.cache/easymp_h100`. `TMPDIR` remained unset. This avoids the home
quota failure without creating overlong Unix-domain socket paths.

## Performance measurements

The controlled comparison below uses five exact matched C64 seeds, 64 requests
per seed. These are the representative warmed service numbers.

| C64 2xB32 profile | Mean RTFX | Mean req/s | Mean TTFA | Mean ITL | Underruns | Deadlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous locked `[16]` startup | 128.573x | 21.081 | 782.15 ms | 268.30 ms | 0 | 0 |
| Low-TTFA `[10,14,16]` startup | **130.631x** | **21.393** | **595.58 ms** | 286.74 ms | **0** | **0** |

The change reduced mean TTFA by 186.57 ms (23.85%), while RTFX improved 1.60%
and request throughput improved 1.48%. Mean ITL increased by 18.44 ms (6.87%),
but the playback buffer still prevented every underrun and deadline miss.

Definitions:

- **TTFA** is time to first audio; lower is better.
- **RTFX** is generated audio duration divided by wall time; higher is better.
- **ITL** is the mean time between streamed responses; lower is generally
  better, provided packet sizes are compared consistently.
- **Underrun** means the client consumed its buffered audio before the next
  packet arrived.

### Cold, transition, matched, and listening results

These categories are deliberately kept separate:

| Measurement | Load | RTFX | Requests/s | Mean TTFA | Underruns / deadlines |
| --- | ---: | ---: | ---: | ---: | ---: |
| Process-cold validation | C64 | 75.892x | 11.471 | 3,251.98 ms | 0 / 0 |
| Transition cohort | C64 | 138.372x | not aggregated here | 579.00 ms | 0 / 0 |
| Five-seed warmed matched mean | C64 | 130.631x | 21.393 | 595.58 ms | 0 / 0 |
| This first listening generation | C10 | 15.810x | 2.567 | 2,954.82 ms | 0 / 0 |

The listening cohort was the first request set after this fresh server start,
so its TTFA includes cold execution effects. It is useful for the saved audio
and confirms gap-free streaming, but it must not be compared directly with the
warmed C64 matched result. All 10/10 requests completed, producing 61.60
seconds of audio in 3.90 seconds of wall time, with 0/66 chunk underruns and
zero missed deadlines.

No matched measurement was removed as an outlier. The transition cohort is
reported separately rather than included in the matched aggregate.

## Reproduction and source artifacts

- [Low-TTFA deployment config](examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_optimized_h100_c64_stage0x2_low_ttfa.yaml)
- [Low-TTFA launcher](examples/tts/easymagpie_vllm_omni/scripts/run_server_native_optimized_h100_c64_low_ttfa.sh)
- [Locked manifest](examples/tts/easymagpie_vllm_omni/deploy/easymagpie_native_best_h100_c64_low_ttfa.lock.yaml)
- [Raw listening-generation JSON](nsys_traces/h100/codec_b64_low_ttfa/results/c64_low_ttfa_listening_generation_seed20260735.json)
- [Full C64 optimization report](H100_C64_SCALING_RESULTS.md)
- [H100 handoff](H100_OPTIMIZATION_HANDOFF.md)

The previous C64 lock and all A4500-specific locks and tile files were left
unchanged by this sample-generation task.
