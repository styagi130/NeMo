# Riva C++ Text Preprocessor Integration for EasyMagpie vLLM-Omni

## Purpose

This document describes a high-level architecture for integrating the existing
Riva C++ TTS preprocessor with the Python-only EasyMagpie vLLM-Omni server.

The design preserves the responsibilities of both systems:

- Riva C++ performs text normalization, supported SSML processing,
  abbreviation expansion, punctuation cleanup, and sentence handling.
- EasyMagpie uses the Hugging Face tokenizer shipped with its checkpoint,
  predicts acoustic and phoneme streams, and decodes waveform audio.
- vLLM-Omni retains request scheduling, continuous batching, model execution,
  CUDA IPC transport, and streaming response handling.

The central integration contract is **normalized UTF-8 text**. Riva-generated
character or phoneme token IDs must not be passed to EasyMagpie because they do
not belong to the checkpoint's Hugging Face tokenizer vocabulary.

## Design summary

| Question | Decision |
|---|---|
| What crosses the C++/Python boundary? | Ordered normalized UTF-8 text segments |
| Where does the reusable C++ code live? | `riva/tts/preprocessing/` |
| Where does the pybind11 module live? | `riva/tts/bindings/python/riva_tts_preprocessor/` |
| Where does the EasyMagpie vLLM code live? | `riva/cbe/python-tts/vllm_extensions/easymagpie/` |
| Which tokenizer creates model token IDs? | The Hugging Face tokenizer shipped with the EasyMagpie checkpoint |
| How does preprocessing run concurrently? | A bounded Python CPU executor calls C++ with the GIL released |
| How many C++ instances are used? | One shared instance after thread-safety validation; otherwise one instance per active worker |
| Where do production normalization assets live? | Versioned runtime assets, for example `/opt/riva/assets/tts/preprocessor/<language>/` |

The intended dependency chain is:

```text
EasyMagpie vLLM adapter
  -> public riva_tts_preprocessor Python package
  -> private pybind11 _native module
  -> reusable Riva C++ text-preprocessing library
  -> normalization assets and utility libraries
```

Suggested reading order:

1. [High-level architecture](#high-level-architecture)
2. [Repository structure](#proposed-riva-speech-repository-structure)
3. [pybind11 boundary](#pybind11-extension-boundary)
4. [Parallel preprocessing](#parallel-preprocessing-and-instance-ownership)
5. [Implementation phases](#implementation-phases)

## High-level architecture

```text
TTS client
POST /v1/audio/speech
        |
        v
+---------------------------------------------------------------------+
| Python vLLM-Omni API server                                         |
| Speech route -> EasyMagpieTTSAdapter -> bounded CPU executor         |
|                                             |                       |
|                                             v                       |
|                              pybind11 extension                     |
|                           riva_tts_preprocessor                      |
+---------------------------------------------|-----------------------+
                                              | C++ call; GIL released
                                              v
+---------------------------------------------------------------------+
| Riva C++ preprocessing library                                     |
|                                                                     |
| EasyMagpieTextPreprocessor                                          |
|   -> supported SSML parsing -> WFST text normalization              |
|   -> abbreviation expansion -> whitespace and punctuation cleanup   |
|   -> sentence handling                                              |
|                                                                     |
| Normalizer and abbreviation assets                                  |
+---------------------------------------------|-----------------------+
                                              | normalized UTF-8 text
                                              v
+---------------------------------------------------------------------+
| Python vLLM-Omni API server                                         |
| pybind11 result conversion -> EasyMagpie prompt builder             |
|   -> checkpoint HF tokenizer -> vLLM-Omni EngineClient              |
+---------------------------------------------|-----------------------+
                                              |
                                              v
+---------------------------------------------------------------------+
| vLLM-Omni model workers                                             |
| Stage 0 EasyMagpie AR model -> CUDA IPC -> Stage 1 native codec      |
+---------------------------------------------|-----------------------+
                                              |
                                              v
                                  Streaming PCM response
```

## Text contract

```text
Riva request input_string
          │
          ▼
Riva C++ normalization
          │
          ▼
normalized UTF-8 output_string
          │
          ▼
EasyMagpie checkpoint HF tokenizer
          │
          ▼
EasyMagpie text token IDs
```

The existing Riva `output` tensor contains IDs from `CharacterMapping` and
optional G2P processing. Those IDs are valid for traditional Riva TTS model
interfaces but are not valid EasyMagpie text IDs.

## C++ preprocessing mode

Add an EasyMagpie-specific mode such as:

```text
model_class=easymagpie_vllm
text_output_only=true
enable_g2p=false
```

### Enabled operations

- WFST text normalization
- Abbreviation expansion
- Whitespace normalization
- Punctuation handling
- Sentence boundary detection
- SSML `<sub alias="...">` conversion to ordinary spoken text
- Input validation and maximum-length enforcement

### Disabled operations

- Dictionary G2P
- Neural G2P
- IPA conversion
- Riva character-token IDs as model input
- Pitch, duration, volume, and emotion tensors
- `<phoneme>` handling until an explicit EasyMagpie-compatible contract exists
- Prosody controls that EasyMagpie does not consume

The C++ implementation can continue using internal mappings for validation or
sentence-length bookkeeping, but the model-facing output must remain normalized
text.

## pybind11 extension boundary

The Riva preprocessor should be compiled into a pybind11 extension module that
links the existing C++ preprocessor library. pybind11 is the preferred boundary
for this integration because the exposed interface consists of C++ classes,
strings, vectors, and result structs. It provides their Python conversions,
object lifetime management, and C++ exception translation without changing the
parallel execution model.

Using the CPython C API directly would provide the same GIL-release behavior but
would require manual reference counting, type registration, argument and result
conversion, and exception handling. It is only preferable if the deployment
requires the CPython Limited API/stable ABI or has a policy forbidding pybind11.
pybind11 is header-only, so it does not add a runtime shared-library dependency.

The existing Riva preprocessor also contains a `py::module_` member for its
pypinyin integration. A direct CPython wrapper would therefore not remove
pybind11 from the underlying implementation unless the text-only facade were
fully separated from that class.

An intentionally narrow facade keeps Riva pipeline details out of the vLLM
serving layer:

```cpp
struct TextSegment {
  std::string text;
  int32_t sentence_num;
  bool is_last_sentence;
};

class EasyMagpieTextPreprocessor {
 public:
  explicit EasyMagpieTextPreprocessor(const PreprocessorConfig& config);

  std::vector<TextSegment> Preprocess(
      const std::string& text,
      const std::string& language_code,
      int32_t speaker_id);
};
```

The extension validates and converts Python arguments while holding the GIL,
releases the GIL only around `EasyMagpieTextPreprocessor::Preprocess`, then
reacquires it before converting results or translating exceptions:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
  py::class_<TextSegment>(module, "TextSegment")
      .def_readonly("text", &TextSegment::text)
      .def_readonly("sentence_num", &TextSegment::sentence_num)
      .def_readonly("is_last_sentence", &TextSegment::is_last_sentence);

  py::class_<EasyMagpieTextPreprocessor>(module, "TextPreprocessor")
      .def(py::init([](const py::dict& config) {
        return std::make_unique<EasyMagpieTextPreprocessor>(
            PreprocessorConfigFromDict(config));
      }))
      .def(
          "preprocess",
          &EasyMagpieTextPreprocessor::Preprocess,
          py::arg("text"),
          py::arg("language_code"),
          py::arg("speaker_id"),
          py::call_guard<py::gil_scoped_release>());
}
```

`py::call_guard<py::gil_scoped_release>()` limits the unlocked region to the
native C++ method call. Argument conversion happens before the call and result
conversion and exception translation happen after the guard reacquires the
GIL. The C++ text-only path must not access Python objects or invoke Python
callbacks while the GIL is released. In particular, G2P and the pypinyin path
must remain disabled.

The C++ preprocessor object and its normalization assets must be initialized
during API-server startup, not once per request. Whether workers share that
object or use an instance pool is determined by the thread-safety validation
described below.

## Request lifecycle

```text
Client        API server       C extension       HF tokenizer       Engine / Stage 0 / Stage 1
  |               |                 |                  |                         |
  | POST speech   |                 |                  |                         |
  |-------------->| validate        |                  |                         |
  |               | preprocess(...) |                  |                         |
  |               |---------------->| SSML -> TN ->    |                         |
  |               |                 | abbreviations -> |                         |
  |               | normalized text | cleanup          |                         |
  |               |<----------------|                  |                         |
  |               | encode text                        |                         |
  |               |----------------------------------->|                         |
  |               | checkpoint-compatible token IDs   |                         |
  |               |<-----------------------------------|                         |
  |               | submit request                                              |
  |               |------------------------------------------------------------>|
  |               |                     acoustic frames -> codec -> PCM chunks   |
  | PCM bytes     |<------------------------------------------------------------|
  |<--------------|                 repeated while audio is generated            |
```

## EasyMagpie serving integration

Preprocessing belongs in the asynchronous
`EasyMagpieTTSAdapter.build()` path, before the normalized text is stored in
`additional_information.text`.

Conceptual Python integration:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

from riva_tts_preprocessor import TextPreprocessor


class EasyMagpieTTSAdapter(ARTTSAdapter):
    def __init__(self, ctx):
        super().__init__(ctx)
        self._preprocessor = TextPreprocessor(load_preprocessor_config())
        self._preprocess_executor = ThreadPoolExecutor(max_workers=8)
        self._preprocess_slots = asyncio.Semaphore(16)

    async def _normalize(self, text, language_code, speaker_id):
        loop = asyncio.get_running_loop()
        async with self._preprocess_slots:
            return await loop.run_in_executor(
                self._preprocess_executor,
                self._preprocessor.preprocess,
                text,
                language_code,
                speaker_id,
            )

    async def build(self, request, sampling_params_list, has_inline_ref_audio):
        extra = request.extra_params or {}
        language_code = extra.get("language_code", "en-US")
        voice = (request.voice or "eng").strip()
        riva_speaker_id = resolve_riva_speaker_id(voice)

        segments = await self._normalize(
            request.input,
            language_code,
            riva_speaker_id,
        )
        normalized_text = " ".join(segment.text for segment in segments)

        prompt = {
            "prompt_token_ids": [0] * self._prompt_len(voice),
            "additional_information": {
                "text": normalized_text,
                "context_text": context_for_language(language_code),
                "speaker_id": voice,
                "temperature": float(extra.get("temperature", 0.7)),
                "top_k": int(extra.get("top_k", 80)),
            },
        }
        return PreparedRequest(
            prompt=prompt,
            tts_params={},
            model_type="easymagpie",
        )
```

The sample uses a dedicated bounded executor and assumes that concurrent calls
on one preprocessor instance have passed the validation below. Do not
additionally invoke an internally parallel C++ batch API from every executor
worker, because nested thread pools can oversubscribe the host.

## Parallel preprocessing and instance ownership

pybind11 and the direct CPython C API have equivalent native parallelism when
the GIL is released. The binding choice does not determine concurrency. The
important requirements are:

- Release the GIL for the complete CPU-bound C++ preprocessing call.
- Ensure that every object and normalization asset touched by that call is
  thread-safe.
- Bound both active workers and queued requests.
- Choose one request-parallelism layer rather than nesting Python and C++
  request pools.

The existing Riva `ForwardBatchDecoupled` method submits
`ForwardDecoupled` calls on the same preprocessor object to its internal thread
pool. This is evidence that shared-instance concurrency is intended. The class
also protects its pypinyin path with a mutex. It is not, however, sufficient
proof that every WFST normalizer and text-only facade dependency is safe under
concurrent calls.

Use this ownership decision:

1. **Preferred after validation:** one shared, immutable-after-startup
   `TextPreprocessor` per API-server process. All executor workers call it
   concurrently while the GIL is released. This minimizes asset memory.
2. **Safe fallback:** a pool of `N` initialized preprocessors, with at most one
   in-flight call per instance. Acquire an instance before executor dispatch and
   return it to the pool afterward. Share immutable normalization assets between
   instances where the Riva APIs allow it.
3. Do not place a mutex around the complete preprocessing call. That would make
   the API safe but serialize all requests and defeat the CPU executor.

Before selecting the shared-instance design, run repeated mixed-input calls at
concurrency 1, 8, and 16, compare every result with the serial baseline, and run
a ThreadSanitizer build where supported. Include multiple languages, supported
SSML, failure paths, and client cancellation. If thread safety cannot be
established, use the instance pool.

## Proposed `riva-speech` repository structure

The repository already keeps Python TTS backends under
`riva/cbe/python-tts/` and the current Riva pipeline preprocessor under
`riva/tts/pipeline/preprocessor/`. Preserve those ownership boundaries while
extracting the reusable text-only logic from the pipeline implementation:

| Layer | Recommended path | Responsibility |
|---|---|---|
| Reusable C++ library | `riva/tts/preprocessing/` | Framework-neutral normalization, supported SSML, abbreviation expansion, cleanup, sentence handling, configuration, and result types |
| Existing Riva pipeline adapter | `riva/tts/pipeline/preprocessor/` | Triton tensors, legacy outputs, G2P, pypinyin, and traditional Riva pipeline behavior |
| Python binding | `riva/tts/bindings/python/riva_tts_preprocessor/` | Thin pybind11 module, GIL policy, conversions, exceptions, and public Python wrapper |
| Shared vLLM utilities | `riva/cbe/python-tts/vllm_extensions/common/` | Bounded executor and optional preprocessor-instance pool |
| EasyMagpie integration | `riva/cbe/python-tts/vllm_extensions/easymagpie/` | Serving adapter, prompt construction, speaker/language mapping, and vLLM integration |
| Packaging | `riva/package/` and `docker/` | Assemble the Python package, native module, shared libraries, and runtime assets |

```text
riva-speech/
├── riva/
│   ├── tts/
│   │   ├── preprocessing/                         # Reusable C++ domain library
│   │   │   ├── BUILD
│   │   │   ├── text_preprocessor.h
│   │   │   ├── text_preprocessor.cc
│   │   │   ├── text_segment.h
│   │   │   ├── preprocessor_config.h
│   │   │   ├── preprocessor_config.cc
│   │   │   ├── normalizer_assets.h
│   │   │   ├── normalizer_assets.cc
│   │   │   └── text_preprocessor_test.cc
│   │   │
│   │   ├── pipeline/
│   │   │   └── preprocessor/                      # Existing Riva pipeline adapter
│   │   │       ├── BUILD
│   │   │       ├── preprocessor.h
│   │   │       ├── preprocessor.cc
│   │   │       ├── aggregated_tokenizer.h
│   │   │       ├── aggregated_tokenizer.cc
│   │   │       └── ...
│   │   │
│   │   └── bindings/
│   │       └── python/
│   │           └── riva_tts_preprocessor/         # Public package + native module
│   │               ├── BUILD
│   │               ├── __init__.py                # Stable public Python API
│   │               ├── module.cc                  # PYBIND11_MODULE(_native, ...)
│   │               ├── _native.pyi                # Native API type declarations
│   │               └── text_preprocessor_test.py
│   │
│   ├── cbe/
│   │   └── python-tts/
│   │       ├── vllm_extensions/                   # vLLM-specific Python code
│   │       │   ├── BUILD
│   │       │   ├── __init__.py
│   │       │   ├── common/
│   │       │   │   ├── BUILD
│   │       │   │   ├── __init__.py
│   │       │   │   ├── bounded_executor.py
│   │       │   │   └── preprocessor_pool.py
│   │       │   └── easymagpie/
│   │       │       ├── BUILD
│   │       │       ├── __init__.py
│   │       │       ├── adapter.py
│   │       │       ├── config.py
│   │       │       ├── mappings.py
│   │       │       └── tests/
│   │       │           ├── test_adapter.py
│   │       │           └── test_concurrency.py
│   │       ├── chatterbox_vllm/                   # Existing backend code
│   │       └── model_*.py                         # Existing Triton Python backends
│   │
│   └── package/
│       └── BUILD                                  # Runtime artifact assembly
│
├── docker/
│   └── Dockerfile.riva                            # Copies packaged runtime artifacts
├── MODULE.bazel
└── WORKSPACE
```

### Dependency direction

Dependencies flow toward the reusable C++ library:

```text
vllm_extensions/easymagpie
            |
            v
riva_tts_preprocessor public Python package
            |
            v
riva_tts_preprocessor/_native.so
            |
            v
//riva/tts/preprocessing:text_preprocessor
            |
            v
normalizer, abbreviation, SSML, and utility libraries

//riva/tts/pipeline/preprocessor:preprocessor
            |
            +-------> //riva/tts/preprocessing:text_preprocessor
```

The reusable C++ library must not depend on the binding, vLLM, or the Riva
pipeline adapter. Both the existing pipeline adapter and the pybind11 module
depend on the same library. This prevents normalization behavior from diverging
between traditional Riva TTS and EasyMagpie serving.

### Python package and native module

Use a small public Python package rather than exposing the compiled module
directly:

```python
# riva_tts_preprocessor/__init__.py
from ._native import TextPreprocessor, TextSegment

__all__ = ["TextPreprocessor", "TextSegment"]
```

The compiled filename is
`riva_tts_preprocessor/_native.<python-SOABI>.so`, while callers continue using:

```python
from riva_tts_preprocessor import TextPreprocessor
```

Keeping `_native` private permits validation, compatibility checks, typing, and
future implementation changes without changing the serving-layer import.

### Bazel targets

The repository already configures `pybind11_bazel`. Use its
`pybind_extension` rule rather than a handwritten generic shared-library
target.

| Package | Target | Rule | Main dependency |
|---|---|---|---|
| `//riva/tts/preprocessing` | `text_preprocessor` | `cc_library` | Normalizer, string-processing, and SSML libraries |
| `//riva/tts/preprocessing` | `text_preprocessor_test` | `cc_test` | `:text_preprocessor` |
| `//riva/tts/bindings/python/riva_tts_preprocessor` | `_native` | `pybind_extension` | `//riva/tts/preprocessing:text_preprocessor` |
| `//riva/tts/bindings/python/riva_tts_preprocessor` | `riva_tts_preprocessor` | `py_library` | `:_native` as runtime data |
| `//riva/tts/bindings/python/riva_tts_preprocessor` | `text_preprocessor_test` | `py_test` | `:riva_tts_preprocessor` |
| `//riva/cbe/python-tts/vllm_extensions/easymagpie` | `easymagpie` | `py_library` | Public `riva_tts_preprocessor` package |

```python
# riva/tts/bindings/python/riva_tts_preprocessor/BUILD
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "_native",
    srcs = ["module.cc"],
    visibility = ["//visibility:private"],
    deps = ["//riva/tts/preprocessing:text_preprocessor"],
)
```

Keep the core visible only to the existing pipeline and Python-binding
packages. Keep `_native` private; vLLM code depends on the public
`riva_tts_preprocessor` wrapper. Exact dependency labels follow the final C++
extraction.

### Runtime assets and packaging

Do not place production WFST archives or large abbreviation assets inside
`vllm_extensions` or the binding directory. The C++ library accepts asset paths
through `PreprocessorConfig`; tests use small fixtures beside their owning
tests. Production assets remain versioned deployment/model artifacts and are
installed under a path such as:

```text
/opt/riva/assets/tts/preprocessor/<language>/
```

The package layer assembles:

```text
/opt/riva/python/riva_tts_preprocessor/__init__.py
/opt/riva/python/riva_tts_preprocessor/_native.<python-SOABI>.so
/opt/riva/backends/vllm_extensions/
/opt/riva/assets/tts/preprocessor/
```

Add `/opt/riva/python` to `PYTHONPATH` or install the package into the serving
environment. The Dockerfile should copy a Bazel-produced package artifact
rather than independently reconstructing the file layout.

The extraction and rollout order is captured in
[Implementation phases](#implementation-phases).

## Language and speaker mapping

Riva and EasyMagpie currently use different identifiers:

| Riva input | EasyMagpie input |
|---|---|
| Numeric `speaker` | Registered string speaker ID such as `eng` |
| `language_code=en-US` | `context_text=[EN]` |

The first deployment can use explicit static maps:

```text
Riva speaker 0  → EasyMagpie speaker "eng"
language en-US  → context token "[EN]"
```

Unsupported speakers or languages should fail validation before preprocessing
or model submission. Additional mappings should only be enabled when matching
speaker embeddings and language-conditioning tokens exist in the checkpoint.

## Sentence handling

### Initial implementation

Join normalized sentence segments and submit one EasyMagpie request.

Benefits:

- One speaker/context prefill
- Continuous EasyMagpie acoustic state
- Simple cancellation and error handling
- No sentence-boundary audio gaps
- Better GPU batching efficiency

### Future long-text implementation

```text
Normalized sentence segments
             |
             v
  Per-request ordered queue
      |        |        |
      v        v        v
  request 0  request 1  request N
      |        |        |
      +--------+--------+
               |
               v
      Ordered PCM stream
```

Sentence fan-out requires explicit ordering, cancellation propagation, and
audio-boundary handling. It also repeats speaker prefill work, so it should not
be the initial production path.

## Incremental text streaming

Arbitrary text fragments cannot always be normalized independently:

```text
fragment 1: "The total is $1"
fragment 2: ",250.50."
```

Normalizing these fragments separately can produce a different result from
normalizing the complete sentence.

Recommended rollout:

1. Support complete-text `/v1/audio/speech` requests first.
2. For the incremental endpoint, initially buffer until `input.done`.
3. Later, normalize complete sentence units as they become available.
4. Retain direct `input.tokens` support for trusted clients that already use
   the checkpoint tokenizer.

Audio remains streaming after model execution starts, even when input text is
buffered for normalization.

## Scaling architecture

```text
Concurrent TTS requests
          |
          v
Bounded preprocessing queue
          |
    +-----+-----+--------+-----+
    |           |        |     |
    v           v        v     v
 worker 1    worker 2  worker 3 ... worker 8
    |           |        |     |
    +-----------+--------+-----+
                |
                v
      pybind11 calls with GIL released
                |
                v
   shared thread-safe preprocessor
        or N-instance pool
                |
                v
      vLLM continuous batching
                |
                v
       Stage 0 + Stage 1 GPU
                |
                v
       Concurrent PCM streams
```

For eight CPU workers and a representative 4 ms preprocessing time:

```text
theoretical preprocessing capacity = 8 / 0.004 = 2,000 requests/second
```

This is substantially higher than the EasyMagpie GPU request rate. The GPU
should remain the throughput bottleneck as long as preprocessing is configured
with sufficient CPU concurrency.

Recommended initial settings:

```text
preprocessing workers:  min(available CPU cores, 8 to 16)
maximum queued calls:   16 to 32
GIL during C++ work:    released
preprocessor instances: one shared if verified thread-safe;
                        otherwise one per active worker
```

Each horizontally scaled API-server replica owns its preprocessor or
preprocessor pool. Load and share immutable normalization assets once per
replica where the Riva API permits it. The objects must not be instantiated in
every vLLM model worker.

## Latency budget

The added preprocessing work is entirely before vLLM request admission:

```text
TTFA =
  request parsing
+ Riva C++ preprocessing
+ HF tokenization
+ vLLM queueing and prefill
+ first Stage-0 acoustic frame
+ first codec decode
+ response transport
```

Expected warm-request planning ranges:

| Component | Typical | P95 |
|---|---:|---:|
| Python-to-pybind11 call | less than 0.1 ms | less than 0.3 ms |
| Executor dispatch | 0.1–0.5 ms | about 1 ms |
| English WFST normalization | 0.5–3 ms | 5–10 ms |
| SSML, abbreviation, and cleanup | 0.1–2 ms | 3–5 ms |
| Result conversion | less than 0.2 ms | less than 0.5 ms |
| **Total addition** | **1–6 ms** | **5–15 ms** |

Long or normalization-heavy inputs require more work:

| Input type | Planning range |
|---|---:|
| Short plain-text sentence | 1–4 ms |
| Numbers, currencies, and abbreviations | 3–10 ms |
| Moderate supported SSML | 5–15 ms |
| More than 1,000 characters | 10–50+ ms |

These values are estimates and must be verified on the production CPU and
normalizer assets.

### Semantic expansion

Normalization can expand the spoken content:

```text
"$1,250" → "one thousand two hundred fifty dollars"
```

The C++ computation may take only a few milliseconds, but the normalized text
contains more model tokens and produces more audio. That increase in total
generation time is expected speech content rather than preprocessing overhead.

## Admission jitter

At concurrency 16, requests with different normalization complexity may reach
vLLM a few milliseconds apart. vLLM continuous batching should absorb most of
this spread.

If profiling shows that preprocessing fragments the initial GPU cohort, add a
small 1–2 ms admission-coalescing window after preprocessing. This should only
be enabled from measured evidence because it directly trades TTFA for larger
initial batches.

## Error handling

| Failure | API behavior |
|---|---|
| Empty or punctuation-only input | HTTP 400 |
| Unsupported language | HTTP 400 |
| Unsupported speaker | HTTP 400 |
| Unsupported SSML tag | HTTP 400 with a specific tag error |
| Preprocessor queue full | HTTP 429 or bounded wait with timeout |
| No preprocessor instance available | HTTP 429 or the same bounded wait used by the preprocessing queue |
| Normalizer runtime error | HTTP 400 for invalid text, otherwise HTTP 500 |
| Model submission failure | Existing vLLM error path |
| Client cancellation | Cancel queued work when possible and abort the vLLM request |

Errors should be detected before GPU request admission whenever possible.

## Deployment

The vLLM serving image must contain:

- The compiled `riva_tts_preprocessor` pybind11 extension
- All shared libraries needed by the Riva preprocessor
- WFST normalization assets
- Abbreviation assets
- An EasyMagpie-specific preprocessor configuration
- The existing EasyMagpie model and tokenizer files

Library compatibility must be checked for:

- C++ ABI
- libtorch version
- Python version and extension ABI
- pybind11 build version
- glibc and compiler runtime

pybind11 is header-only and does not add a runtime shared-library dependency.
Build the extension against the Python version used in the serving image.

The extension should be imported only in the API-server process. The Stage 0
and Stage 1 model processes do not need Riva preprocessing libraries.

## Observability

Record these per-request timestamps:

```text
request_received
preprocessing_started
preprocessing_completed
engine_request_submitted
first_audio_emitted
request_completed
```

Derived metrics:

```text
preprocessing latency =
  preprocessing_completed - preprocessing_started

admission delay =
  engine_request_submitted - preprocessing_completed

model TTFA =
  first_audio_emitted - engine_request_submitted

end-to-end TTFA =
  first_audio_emitted - request_received
```

Recommended counters and histograms:

- Preprocessing latency by language
- Input and normalized-text byte length
- Normalized HF token count
- Preprocessor queue depth and wait time
- Active preprocessing workers
- Available and checked-out preprocessor instances when using an instance pool
- Normalization failures
- Unsupported SSML requests
- End-to-end TTFA with and without preprocessing

## Validation plan

### Functional validation

- Compare normalized output against Riva preprocessor golden tests.
- Confirm EasyMagpie receives normalized text, not Riva token IDs.
- Test numbers, dates, currencies, abbreviations, punctuation, and whitespace.
- Test `<sub>` SSML.
- Reject unsupported phoneme and prosody tags.
- Verify speaker and language mapping errors.
- Confirm cancellation during preprocessing and generation.
- Compare concurrent results byte-for-byte with a serial golden baseline.
- Verify that one preprocessor instance never receives overlapping calls when
  the instance-pool fallback is active.

### Performance validation

Run fixed-corpus comparisons at concurrency 1, 8, and 16:

1. EasyMagpie without C++ preprocessing.
2. EasyMagpie with C++ preprocessing enabled.
3. Normalization-only microbenchmark.
4. Mixed plain and normalization-heavy text.
5. Shared-instance and instance-pool modes, if both pass correctness tests.

Report:

- RTFX
- TTFA mean and P95
- Preprocessing mean and P95
- Admission delay
- Time to first audio
- Steady-state audio inter-chunk latency
- Playback underruns
- CPU utilization and queue depth
- Preprocessor-instance pool utilization and wait time

The main acceptance condition is that normalization improves frontend
correctness without materially reducing GPU batching efficiency or introducing
playback underruns.

## Implementation phases

### Phase 1: C++ text-only facade

- Add `//riva/tts/preprocessing:text_preprocessor`.
- Add the EasyMagpie text-only preprocessor mode.
- Disable G2P and model-facing Riva token IDs.
- Expose normalized text segments.
- Make the existing pipeline preprocessor consume the reusable library.
- Add C++ unit tests for normalization-only behavior.

### Phase 2: Python extension

- Add the pybind11 extension under
  `riva/tts/bindings/python/riva_tts_preprocessor/`.
- Bind `TextSegment` and `TextPreprocessor`.
- Release the GIL around only the C++ preprocessing call with
  `py::call_guard<py::gil_scoped_release>()`.
- Add any domain-specific C++ exception translators required by the HTTP error
  mapping.
- Package shared libraries and normalization assets.
- Add pybind11 extension tests, including lifetime, conversion, GIL-release, and
  error-path tests.

### Phase 3: Thread safety and ownership

- Stress concurrent preprocessing against a serial golden baseline.
- Run ThreadSanitizer where supported.
- Verify the WFST normalizer and abbreviation assets under mixed concurrent
  inputs.
- Select one shared preprocessor only after validation; otherwise implement a
  bounded instance pool.
- Verify that the text-only path never enters pypinyin or another Python API
  while the GIL is released.

### Phase 4: Standard HTTP serving

- Initialize the selected shared preprocessor or instance pool during startup.
- Add `riva/cbe/python-tts/vllm_extensions/easymagpie/`.
- Integrate preprocessing into `EasyMagpieTTSAdapter.build()`.
- Add bounded CPU execution and backpressure.
- Join normalized sentence segments into one model request.
- Add request timing metrics.

### Phase 5: Concurrency and performance validation

- Benchmark concurrency 1, 8, and 16.
- Verify CPU scaling and vLLM cohort formation.
- Tune executor size and queue limits.
- Confirm that no nested request-level thread pools are active.
- Add admission coalescing only if measurements justify it.

### Phase 6: Incremental input

- Buffer incremental text until `input.done`.
- Add sentence-boundary normalization.
- Validate cross-chunk numbers, abbreviations, and SSML.
- Preserve direct trusted-token input.

## Relevant source locations

### Riva C++ preprocessor

- `/home/siddhartht/riva-speech/riva/tts/pipeline/preprocessor/preprocessor.h`
- `/home/siddhartht/riva-speech/riva/tts/pipeline/preprocessor/preprocessor.cc`
- `/home/siddhartht/riva-speech/riva/tts/pipeline/preprocessor/aggregated_tokenizer.h`
- `/home/siddhartht/riva-speech/riva/tts/pipeline/preprocessor/aggregated_tokenizer.cc`
- `/home/siddhartht/riva-speech/riva/tts/pipeline/preprocessor/BUILD`

### EasyMagpie vLLM-Omni

- `easymagpie_vllm_omni/serving_adapter.py`
- `easymagpie_vllm_omni/serving_stream.py`
- `easymagpie_vllm_omni/easymagpie.py`
- `easymagpie_vllm_omni/pipeline.py`
