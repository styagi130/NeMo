# Direct Rust Tokenizer: Implementation and Results

Date: 2026-07-31

## What changed

The Rust EngineCore client can now load the converted EasyMagpie model's
`tokenizer.json` directly with the Hugging Face `tokenizers` Rust crate.
`--rust-tokenizer-model /model` enables the path:

1. Context text is tokenized once at startup.
2. Each target utterance is tokenized in Rust.
3. The EasyMagpie text EOS ID is read from `config.json` or derived from the
   checkpoint text vocabulary.
4. `context_token_ids` and `text_tokens` are sent in the existing
   `additional_information` MessagePack field.
5. The Python Stage-0 model consumes supplied IDs without invoking its
   `AutoTokenizer`.

The raw text fields are retained for compatibility and logging. Omitting
`--rust-tokenizer-model` retains the original Python-tokenizer behavior.

The Rust receive loop also yields cooperatively after each ZMQ output message.
This prevents the immediately-ready multipart receive future from starving
Tokio's reactor/timer tasks during long back-to-back cohorts.

## Correctness

- The `[EN]` context IDs match exactly: `[1091, 4456, 1093]`.
- `Hello, world!` matches exactly:
  `[22177, 1044, 4304, 1033, 131073]`.
- All 10 benchmark utterances plus four punctuation, Unicode, and whitespace
  cases matched the current Python checkpoint tokenizer: **14/14 exact**,
  **0 mismatches**.
- Rust unit tests: **6 passed**.
- Focused Python suite: **29 passed**.
- A BS1 full-pipeline smoke test produced a valid 22.05 kHz WAV.

## Matched BS16 benchmark

Hardware was the development RTX A4500. Both runs used the same optimized
deployment, MPS codec share, A4500 MoE tiles, compile caches, corpus, and seeds.
Each result is one warmup cohort followed by five measured BS16 cohorts
(80 requests total, seeds `20260729` through `20260733`).

| Metric | Direct Rust tokenizer | Python Stage-0 tokenizer | Delta |
| --- | ---: | ---: | ---: |
| Measured wall time | 11.184 s | 11.183 s | +0.002 s |
| Request throughput | 7.153 req/s | 7.154 req/s | -0.01% |
| Mean TTFA | **193.976 ms** | 195.184 ms | **-1.208 ms** |
| P95 TTFA | **205.598 ms** | 209.786 ms | **-4.188 ms** |
| Mean ITL | **118.143 ms** | 118.227 ms | -0.084 ms |
| Mean latency | **1744.608 ms** | 1763.168 ms | -18.560 ms |
| Aggregate RTFX | 44.928x | 45.320x | -0.86% |
| Playback underruns | **0 / 1,130** | **0 / 1,141** | both zero |

RTFX is not the clean tokenizer comparison because sampling produced different
audio lengths: 502.48 seconds for the Rust-tokenizer run and 506.80 seconds
for the Python-tokenizer run. Wall time and request throughput are effectively
identical. This single run suggested a small TTFA reduction; the repeated test
below shows that the difference is within engine restart variance.

## Repeated benchmark

The matched test was subsequently repeated five times per mode. Mode order was
alternated (`Rust/Python`, then `Python/Rust`) to reduce cache, thermal, and
ordering bias. Each repetition still contained one warmup plus five measured
BS16 cohorts, or 80 measured requests.

An unrelated `riva-deploy` process began using the GPU exactly when repetition
5 started. That pair is reported separately as a co-tenancy result and is not
included in the clean distribution below. The clean sample therefore contains
four independent repetitions, 20 measured cohorts, and 320 requests per mode.

| Metric | Direct Rust tokenizer, mean ± SD | Python Stage-0 tokenizer, mean ± SD | Mean paired Rust−Python delta |
| --- | ---: | ---: | ---: |
| Aggregate RTFX | 44.264 ± 1.021x | 44.513 ± 0.810x | -0.248x |
| Request throughput | 7.049 ± 0.167 req/s | 7.099 ± 0.110 req/s | -0.050 req/s |
| Mean TTFA | 195.585 ± 4.329 ms | 198.154 ± 3.420 ms | **-2.569 ms** |
| P95 TTFA | 207.581 ± 3.711 ms | 210.575 ± 7.829 ms | **-2.994 ms** |
| Mean latency | 1790.460 ± 37.406 ms | 1775.587 ± 18.928 ms | +14.873 ms |
| Mean ITL | 120.964 ± 2.557 ms | 119.900 ± 1.303 ms | +1.064 ms |
| Playback underruns | **0 / 4,539 chunks** | **0 / 4,530 chunks** | — |

The paired TTFA differences were `-6.409`, `+1.297`, `+1.621`, and
`-6.784` ms. Their mean is -2.569 ms, but the approximate 95% confidence
interval is -9.98 to +4.84 ms. The result therefore does **not** establish a
statistically reliable latency or throughput improvement. Direct Rust
tokenization is performance-neutral at BS16 within engine restart variance;
its established benefits are removing request-time Python tokenizer work and
making the frontend more self-contained.

The contaminated fifth repetition illustrates co-tenancy sensitivity:

| Metric | Rust tokenizer + `riva-deploy` | Python tokenizer + `riva-deploy` |
| --- | ---: | ---: |
| Throughput | 5.448 req/s | 5.321 req/s |
| Mean TTFA | 379.991 ms | 266.216 ms |
| P95 TTFA | 1026.261 ms | 309.449 ms |
| Playback underruns | 42 / 1,129 (3.720%) | 27 / 1,134 (2.381%) |

This slowdown affected both modes and coincided with an external process using
approximately 1.47 GiB of GPU memory. It should not be attributed to the
tokenizer implementation, but it does show that exclusive-GPU admission or
resource isolation is required for stable latency and zero underruns.

## Reproduction

Add this option to the matched full-pipeline command in `README.md`:

```text
--rust-tokenizer-model /model
```

Remove it for the Python Stage-0 tokenizer baseline. Raw client and engine logs
are stored under:

```text
../nsys_traces/rust_tokenizer_20260731/
../nsys_traces/rust_tokenizer_repeated_20260731/
```
