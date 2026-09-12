## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

EasyMagpieTTS decomposes into EasyMagpie LM and SpectralCodec-BWE-22kHz:

| Stage | Role |
|-------|------|
| **0 — EasyMagpie LM** | `EasyMagpie_LM_Backbone` (Nemotron-H) + `EasyMagpie_LM_LT` → stacked acoustic codes |
| **1 — SpectralCodec-BWE-22kHz** | Stateful native vLLM codec → 22.05 kHz waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

### Convert a NeMo checkpoint

This step converts the EasyMagpie LM, precomputes the text-embedding lookup,
and saves the tokenizer and optional named-speaker embedding. The Stage-1 codec
decoder is always bundled. By default, the codec encoder and reference-speaker
Transformer are not copied, so the resulting artifact accepts only known
`speaker_id` values.

Pass `--bundle-audio-encoders` only when you intend to package the codec encoder
and reference-speaker Transformer together. This one opt-in enables request-time
raw audio and therefore zero-shot voice cloning; it does not control the decoder.
Run conversion in the **NeMo environment** from the repository root. Prepending
the repository root to `PYTHONPATH` is important when the environment has an
editable NeMo install pointing at a different checkout:

```bash
PYTHONPATH="$PWD" python tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir tools/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

### Setup the serving environment

Serving needs a GPU, matching **vLLM 0.26.0 / vLLM-Omni 0.26.0** versions, and this package.
It does not need NeMo after conversion:

```bash
cd tools/easymagpie_vllm_omni
conda create -n easymagpie-vllm python=3.12 -y
conda activate easymagpie-vllm
pip install -r requirements.txt
pip install -e ".[perth]"
# optionally for notebook
pip install ipykernel
python -m ipykernel install --user \
  --name easymagpie-vllm \
  --display-name "Python (easymagpie-vllm)"
```

The `perth` extra is required for default watermarking. `pip install -r requirements-perth.txt`
is equivalent if you install the package without extras.

Mamba's selective-state-update kernel requires shape- and GPU-specific tuning, so an untuned cache can give
suboptimal performance. Reuse the same Triton/vLLM cache directories across launches so repeated runs accumulate
better kernels; for an explicit sweep, run `python scripts/tune_mamba_ssu.py --model converted_model` and restart.

### Perth watermarking

EasyMagpie audio is watermarked with [Perth](https://github.com/resemble-ai/perth) after native codec decoding and before float
PCM leaves the vLLM-Omni codec stage. Pass `NEMOTRON_TTS_PERTH_WATERMARK=0` to disable it.

### Standalone serving image

Build from the Speech repository root; the image installs this entire package on
the pinned upstream stack, without a nemotron-speech overlay:

```bash
docker build -f tools/easymagpie_vllm_omni/Dockerfile -t easymagpie-omni .
docker run --rm --gpus all --ipc=host -p 8091:8091 \
  -v /absolute/path/to/converted_model:/model:ro easymagpie-omni /model 8091
```

The image builds pinned Cairo bindings with builder-only native prerequisites and aligns `nixl-cu13` with
the base's NIXL 1.3.1 packages; the final install must pass `pip check`. This does not validate optional NIXL
GPU transfers. The inherited system PyGObject binding targets CPython 3.10 and cannot load in CPython 3.12;
it is not required by EasyMagpie's shared-memory pipeline.

The default profile has one LM and one FP32 codec, each with capacity 32. It is
not the tuned H100 BS128 profile. This source accepts converted `.pt` speaker
contexts; do not replace an existing speaker bundle without a separate migration.
The codec retains per-stream state between chunks, so streams waiting for their
next chunk still count toward its capacity. This prevents parked streams from
being re-admitted with already-decoded history as a new chunk.
Container caches and uploaded speaker samples default to writable `/tmp` paths.
Override the cache variables with writable persistent mounts when reusing compilation caches.
The launcher uses `exec` for signal forwarding; graceful shutdown still needs runtime validation.


### Quick start — offline synthesis

See the [`offline_demo.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/offline_demo.ipynb) tutorial to check how
`AsyncOmni` is initialized and used.

### Request-time reference and user audio

Every source EasyMagpie NeMo checkpoint contains the codec encoder and
reference-speaker Transformer. They are omitted from converted artifacts by
default and bundled together only with `--bundle-audio-encoders`; the Stage-1
codec decoder is always bundled. Without this explicit opt-in, requests must
select a precomputed embedding by `speaker_id`.

`arch.audio_input_token_id` in `prompt_token_ids` is an audio marker. Each
marker is paired, in order, with one item from `multi_modal_data["audio"]`;
the processor rejects a marker/item count mismatch. A non-final first marker
selects reference conditioning, while a final marker selects user history.
Each item is encoded once into both representations, and its marker selects
which representation is inserted.

Pass each item as `(waveform, sample_rate)`. It must already be mono and match
`arch.codec_input_sample_rate`; the serving path deliberately does not downmix
or resample it.

Each raw audio item is limited to `arch.max_audio_seconds` (30 seconds by
default). Longer inputs are rejected rather than truncated or chunked; set
`--max-audio-seconds` during conversion to change the startup profiling limit.

For a first turn with raw reference audio and user speech, place the reference
marker before context rows and the user marker last:

```python
prompt = {
    "prompt_token_ids": (
        [0] * task_rows
        + [arch.audio_input_token_id]
        + [0] * len(context_token_ids)
        + [arch.audio_input_token_id]
    ),
    "multi_modal_data": {
        "audio": [
            (reference_waveform, reference_sample_rate),
            (user_waveform, user_sample_rate),
        ],
    },
    "additional_information": {
        "context_text": "[EN]",
        "text": response_text,
        "text_prefill_num": arch.text_prefill_num,
        "temperature": 0.7,
        "top_k": 80,
        "reset_codec_on_segment": True,
    },
}
```

A non-final marker is reference conditioning; a final marker is user history.
For reference-only synthesis, replace the final user marker with
`arch.text_prefill_num` zero rows and provide `prefill_text_tokens`. For a
known voice, set `speaker_id`; then a sole final marker is user history. On
later turns of the same Stage-0 request, submit only that final marker and the
new user waveform—the model retains the earlier raw reference conditioning.

User-history prefill additionally requires a checkpoint configured with
`use_multiturn_dataset`, `condition_on_user_speech`, and the user-speaking
tokens. Reference-only raw audio does not require those multiturn features.

Yielding turns as `StreamingInput` items from one async input generator keeps
the Stage-0 causal/Mamba state. A separate `omni.generate(...)` call starts a
new request and must provide `speaker_id` or raw reference conditioning again.
Set `reset_codec_on_segment=True` for each reply so Stage 1 flushes and resets
its response-local codec state.

The OpenAI-compatible speech endpoint accepts request-time reference audio for
zero-shot TTS. The incremental WebSocket endpoint currently accepts text input
with a known `voice` only; use the direct `AsyncOmni` path for multi-turn raw
user-audio history.

### Serve over HTTP and WebSocket

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091. Two serving
APIs are available:

- `POST /v1/audio/speech` with a complete text input and either a known `voice`
  or one request-time `ref_audio`.
- `WS /v1/audio/speech/stream` with incremental text/token updates and
  asynchronous PCM audio output.

The upstream HTTP `max_new_tokens` field limits Stage 0 only; it does not change
the codec limit or shared sampling defaults. Raw PCM does not expose a backend
finish reason, so a successful response alone cannot prove natural EOS.
WebSocket `input.done` drains queued text and codec output before normal completion;
empty audio payloads are not sent as PCM frames. Intermediate segment boundaries
remain resumable and must not replay consumed audio.
WebSocket `max_new_tokens` is a session-wide Stage 0 budget across text updates
and the acoustic tail. At the limit, queued text is drained through `input.done`
without generating extra tokens; the final codec payload still completes normally.
Codec completion follows the connector's terminal message, even when no audio
remains to flush. A streaming session's final text-input marker does not submit
another codec placeholder, so late input bookkeeping cannot reopen the request.

HTTP `extra_params.context_text` conditions and sizes the same prefill context;
omitted, null or empty values retain `[EN]`. Other value types are rejected before
generation. The upstream WebSocket configuration does not expose this override
and ignores unknown configuration keys; WebSocket context remains `[EN]`.

Converted checkpoints with `enable_phoneme_text_input=true` accept inline IPA
spans such as `Turn <bop>lɛft<eop> here`. The markers are syntax only: ordinary
segments use the exported text tokenizer, while span contents use the bundled
IPA tokenizer and the checkpoint's reserved text-token range.

For delayed-stream checkpoints, the adapter folds the known text-led positions
into the causal prefill. The current `phoneme_delay=3`, `speech_delay=5` model
therefore prefills four target positions: text-only positions 0–2 and position
3 with the known phoneme BOS input. Whole-text HTTP requests satisfy this
automatically. Incremental WebSocket input buffers initial updates until at
least `phoneme_delay + 1` tokens are available. Marker strings and IPA spans may
cross `input.text` messages. An unclosed IPA span is rejected at `input.done`;
`input.tokens` remains an exact tokenization bypass and is accepted only when
there is no incomplete text marker or IPA span.

Query the HTTP endpoint from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

See the [`server_request.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/server_request.ipynb) tutorial for examples
of both serving APIs.

For zero-shot TTS, omit `voice` and send `ref_audio` as an HTTP(S) URL, a base64
data URL, or a `file://` URI. Local files must be under the server's
`--allowed-local-media-path`. EasyMagpie does not require `ref_text`. The audio
is downmixed to mono by the HTTP media loader, but it is not resampled: its
sample rate must equal `codec_input_sample_rate` in the converted model.
The model must have been converted with `--bundle-audio-encoders`.

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input":"This voice is conditioned from request-time reference audio.",
    "ref_audio":"file:///absolute/path/to/reference.wav",
    "response_format":"wav",
    "stream":true,
    "stream_format":"audio"
  }' \
  --output out_zero_shot.wav
```

### Benchmarks

```bash
# Benchmark acoustic token prediction only (no codec).
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32 \
    [--streaming --tokens-per-chunk 5]

# Benchmark the service's HTTP API with a known speaker.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --speaker-id eng

# Benchmark zero-shot synthesis with one request-time reference.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --reference-audio /path/to/reference.wav

# Perturb the reference for every request to bypass audio caches.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32 \
    --reference-audio /path/to/reference.wav \
    --randomize-reference-audio

# Benchmark the service's incremental synthesis via its WebSocket API.
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```
