# Tokenizer Optimization and Phase-Timing Results

Date: 2026-07-31

## Implementation

The Rust full-pipeline cohort path now:

1. Tokenizes every target in a BS16 cohort with one `encode_batch` call.
2. Builds Stage-0 and Stage-1 request values in parallel with Rayon.
3. Serializes independent MessagePack requests in parallel.
4. Moves serialized buffers directly into ZMQ sends without another payload
   clone.
5. Records tokenization, request build, serialization, codec admission, talker
   admission, first talker output, and first audio timing for every cohort.

The single-request path and Python Stage-0 tokenizer fallback remain supported.

## Tokenizer-only methodology

All modes used the same converted checkpoint and 10-utterance corpus. Each
trial performed 100 untimed warmup corpus passes followed by 5,000 measured
passes, or 50,000 encoded texts. Ten independent process trials were run per
mode, totaling 500,000 measured texts per mode.

Python sequential calls the same `AutoTokenizer.encode` method used by
EasyMagpie Stage 0. Python batch uses its fast-tokenizer batch API. Rust
sequential uses one `Tokenizer.encode` call per text; Rust batch uses
`Tokenizer.encode_batch`.

## Tokenizer-only results

| Mode | Mean text latency | Text throughput | Speedup vs Python sequential |
| --- | ---: | ---: | ---: |
| Python sequential | 24.087 ± 0.315 us | 41,523 ± 542 text/s | 1.00x |
| Python batch | 10.138 ± 0.280 us | 98,709 ± 2,735 text/s | 2.38x |
| Rust sequential | 13.602 ± 0.121 us | 73,522 ± 650 text/s | 1.77x |
| **Rust batch** | **6.395 ± 0.059 us** | **156,383 ± 1,455 text/s** | **3.77x** |

Rust batch is 2.13x faster than Rust sequential and 1.59x faster than Python
batch. At BS16, its isolated average is about 102 microseconds per cohort,
versus 385 microseconds for the production Python sequential path: an average
saving of approximately 283 microseconds per cohort.

## Batch-size scaling

BS32, BS64, and BS128 were measured directly rather than extrapolated. Each
mode encoded 64,000 texts per trial across five independent trials, with trial
order alternated. Python sequential represents the current Stage-0 per-request
`AutoTokenizer.encode` path.

| Batch size | Rust batch | Python sequential | Python batch | Rust speedup vs sequential | Rust cohort saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | **292,812 text/s** (0.109 ms/cohort) | 42,534 text/s (0.752 ms) | 146,670 text/s (0.218 ms) | **6.88x** | **0.643 ms** |
| 64 | **360,855 text/s** (0.177 ms/cohort) | 41,510 text/s (1.542 ms) | 179,891 text/s (0.356 ms) | **8.69x** | **1.364 ms** |
| 128 | **427,551 text/s** (0.300 ms/cohort) | 41,642 text/s (3.074 ms) | 209,851 text/s (0.612 ms) | **10.25x** | **2.774 ms** |

Against Python's own batch API, Rust remains approximately 2.00x, 2.01x, and
2.04x faster at BS32, BS64, and BS128 respectively. The increasing speedup
against Python sequential comes from parallel tokenization plus amortization
of per-call overhead; absolute savings remain below 3 ms even at BS128.

Cold loading is startup-only. Across sequential and batch trials, Rust loaded
`tokenizer.json` in roughly 237 ms versus roughly 552 ms for Python
`AutoTokenizer`, about 2.32x faster.

## Instrumented BS16 pipeline

The stable optimized Rust rerun used one warmup plus five measured BS16
cohorts. Mean per-cohort frontend timings were:

| Phase | Mean time |
| --- | ---: |
| Batch tokenization | 0.293 ms |
| Parallel request build | 0.100 ms |
| Parallel MessagePack serialization | 0.105 ms |
| Codec queue admission | 0.114 ms |
| Talker queue admission | 0.088 ms |
| First talker output after admission | 53.365 ms |
| First audio after admission | 162.067 ms |

Under live engine CPU contention, tokenization is slower than the isolated
microbenchmark but remains about 0.18% of first-audio time. Tokenization,
request construction, serialization, and both queue admissions total about
0.70 ms per BS16 cohort.

The stable rerun measured 44.413x aggregate RTFX, 7.087 requests/s, 197.442 ms
mean TTFA, and zero underruns across 1,133 audio chunks. These end-to-end values
remain dominated by stochastic generation and engine variance; phase timers,
rather than RTFX, isolate the frontend optimization.

## Reproduction

Rust sequential tokenizer-only benchmark:

```bash
target/release/easymagpie-rust-engine-poc \
  --rust-tokenizer-model /model \
  --tokenize-only \
  --text-file ../bench_corpus.tsv \
  --tokenize-warmup-iterations 100 \
  --tokenize-iterations 5000
```

Add `--tokenize-batch` for the batch path. The matched Python tool is
`benchmark_python_tokenizer.py`; add `--batch` for Python batch mode. Use
`--tokenize-batch-size N` in Rust or `--batch-size N` in Python to construct an
exact tokenizer-only cohort size.

Raw tokenizer and instrumented pipeline logs are under:

```text
../nsys_traces/tokenizer_micro_20260731/
../nsys_traces/tokenizer_scaling_20260731/
```
