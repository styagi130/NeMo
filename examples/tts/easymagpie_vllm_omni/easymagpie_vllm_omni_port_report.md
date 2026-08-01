---
title: "EasyMagpieTTS to vLLM-Omni/Triton Port Report"
author: "Codex"
date: "2026-06-24"
---

# EasyMagpieTTS to vLLM-Omni/Triton Port Report

## Summary

This report documents the work completed to continue and validate the EasyMagpieTTS to vLLM-Omni/Triton port in:

```text
/home/siddhartht/tts/speechLM/NeMoDuplexRealtime
```

The main outcome was a working local Triton deployment path for the converted EasyMagpieTTS model. The codec TensorRT plan was built, a default English speaker embedding was baked, Triton loaded both models successfully, and a whole-text end-to-end request returned audio.

## Starting Point

The previous session had already completed the model and codec conversion stages:

- The EasyMagpie `.nemo` checkpoint had been converted to a vLLM-compatible model directory.
- The codec decoder had been exported to ONNX.
- Conversion and ONNX export scripts had been patched for local-checkout execution and lazy NeMo imports.
- EasyMagpie checkpoint restore had been patched to honor CPU `map_location`.
- vLLM plugin import and parity/unit tests had passed.

The remaining missing pieces were:

- A TensorRT codec plan at `examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan`.
- A baked default speaker embedding at `examples/tts/easymagpie_vllm_omni/easymp_vllm_model/speaker_embeddings/eng.pt`.
- Triton startup validation.
- An end-to-end request or benchmark.

## Artifact Layout

Source model artifacts:

```text
/home/siddhartht/tts/speechLM/NeMo_main/models/easy_tts
```

Important generated/runtime artifacts:

```text
examples/tts/easymagpie_vllm_omni/easymp_vllm_model/
examples/tts/easymagpie_vllm_omni/codec.onnx
examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan
examples/tts/easymagpie_vllm_omni/easymp_vllm_model/speaker_embeddings/eng.pt
```

## Docker Runtime Setup

I first verified that Docker and the GPU were available. The local host had an NVIDIA RTX A4500 with about 20 GB VRAM. Docker access was available after the environment changed to unrestricted execution.

I pulled the Triton base image used by the example Dockerfile:

```bash
docker pull nvcr.io/nvidia/tritonserver:26.02-py3
```

Then I inspected the base image. It provided:

- Triton Server 2.66.0.
- TensorRT 10.15.1 `trtexec`.
- GPU visibility through the NVIDIA runtime.

The base image did not include the vLLM/vLLM-Omni Python dependencies, so I built the project-specific runtime image:

```bash
docker build --network=host -t easymp-vllm-omni examples/tts/easymagpie_vllm_omni/
```

The build installed:

- `vllm==0.21.0`
- `vllm_omni==0.21.0rc1`
- the EasyMagpie vLLM-Omni plugin package
- ONNX and Triton client utilities needed by the export and benchmark scripts

I then validated imports inside the image:

```text
torch 2.11.0+cu130
vllm 0.21.0
vllm_omni 0.21.0rc1
EasyMagpie plugin registered successfully
```

## Building the TensorRT Codec Plan

The exported ONNX codec decoder was already present:

```text
examples/tts/easymagpie_vllm_omni/codec.onnx
```

I built the TensorRT plan inside the `easymp-vllm-omni` container:

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace easymp-vllm-omni \
  python3 examples/tts/easymagpie_vllm_omni/scripts/export_codec_decoder_trt.py \
    --onnx-path examples/tts/easymagpie_vllm_omni/codec.onnx \
    --trt-path examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan \
    --batch-profile 1 8 32 \
    --frames-profile 15 15 15 \
    --fp32
```

TensorRT parsed the ONNX successfully, inferred 16 stacked quantizer/codebook channels, generated the engine, and ran its built-in inference check. The generated plan was:

```text
examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan
```

The plan size was about 210 MB.

## First Triton Startup Attempt

I started Triton using the generated model repository:

```bash
docker run --rm --detach --gpus all --shm-size=8g \
  -p 18000:8000 -p 18001:8001 -p 18002:8002 \
  -v "$PWD":/workspace -w /workspace \
  --name easymp-triton-codex \
  easymp-vllm-omni \
  tritonserver --model-repository=examples/tts/easymagpie_vllm_omni/model_repository
```

The codec TensorRT model loaded successfully. The EasyMagpie Python backend also initialized vLLM-Omni successfully after the first-run Torch compile and CUDA graph capture steps. Both models reached `READY`:

```text
codec   READY
easymp  READY
```

The Triton health endpoint returned HTTP 200:

```bash
curl http://localhost:18000/v2/health/ready
```

However, a first request failed because the model repository configured `default_speaker=eng`, but no corresponding speaker embedding existed:

```text
speaker_embeddings/eng.pt
```

The server error was:

```text
EasyMagpieTTS: no speaker embedding .../speaker_embeddings/eng.pt for speaker_id 'eng'
```

This confirmed that Triton and vLLM loading were working, and that the remaining functional blocker was the missing speaker embedding.

## Baking the Default English Speaker Embedding

The repository contained context audio samples under:

```text
context_audios/audio_context_samples/
```

For the default English speaker, I used:

```text
context_audios/audio_context_samples/english_audio_context_samples/Emma_Additional.flac
```

This file was a mono 22.05 kHz FLAC, about 6.98 seconds long. The conversion path uses a 5 second context window by default, so this was suitable.

Rather than rerunning the full model conversion and rewriting the 2.6 GB `model.safetensors`, I reused the conversion script's speaker extraction function directly:

```python
extract_speaker_embedding(model, context_audio_path, context_audio_duration)
```

The host Python environment did not have all NeMo conversion dependencies, so I used a disposable container based on the existing Riva/NeMo image:

```text
gitlab-master.nvidia.com:5005/dl/riva/riva-speech/riva:dev..siddhartht..magpie_may26_2605.54159250-linux-x86_64
```

That image had most of the NeMo stack, but needed temporary in-container installs of:

```text
loguru
causal-conv1d
mamba-ssm
```

Those dependencies were installed only inside the disposable extraction container. They were not installed into the active host NeMo development environment.

The extraction loaded:

- the EasyMagpieTTS `.nemo` checkpoint
- the spectral codec `.nemo`
- the phoneme tokenizer
- the English context audio

It generated:

```text
examples/tts/easymagpie_vllm_omni/easymp_vllm_model/speaker_embeddings/eng.pt
```

The saved speaker embedding had:

```text
shape = (64, 1536)
dtype = float32
source audio = context_audios/audio_context_samples/english_audio_context_samples/Emma_Additional.flac
checkpoint = 2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12
```

I then corrected file ownership so the generated `eng.pt` belonged to the workspace user rather than container root.

## Final Triton Validation

After baking `eng.pt`, I restarted Triton with the same command and waited for the model repository to load.

The logs showed:

```text
codec   READY
easymp  READY
EasyMagpie initialized (default_speaker=eng, codec_noop=False)
```

The health endpoint returned HTTP 200 again.

## End-to-End Request

I ran one whole-text gRPC request through the existing benchmark client:

```bash
docker exec easymp-triton-codex bash -lc \
  'printf "utt1\tHello world.\n" > /tmp/easymp_one.txt && \
   python3 examples/tts/easymagpie_vllm_omni/scripts/benchmark_service.py \
     --text-file /tmp/easymp_one.txt \
     --triton-url localhost:8001 \
     --no-warmup \
     -n 1 \
     -c 1 \
     --verbose \
     --chunk-timeout 120 \
     --output-dir /tmp/easymp_e2e_out'
```

The request succeeded:

```text
1 ok / 0 failed
```

The benchmark reported:

```text
0.88s audio in 25.63s wall time
TTFA 25590.7 ms
ITL 11.0 ms
5 audio chunks
0 underruns
```

The output audio was copied to the host:

```text
/tmp/easymp_e2e_out/utt1.wav
```

Audio properties:

```text
channels: 1
sample rate: 22050 Hz
precision: 16-bit PCM
duration: 0.88 seconds
```

The server log confirmed the request used the baked default speaker:

```text
speaker=eng text='Hello world.'
```

## RTFX Observation

For the cold first request:

```text
RTFX = audio duration / wall time = 0.88 / 25.65 = about 0.03x
```

This cold result was dominated by first-request time to first audio:

```text
TTFA = about 25.59 seconds
```

The streamed chunks after first audio were much faster:

```text
playback/fetch factor = about 18.5x realtime
```

A meaningful steady-state RTFX measurement should be taken with warmup enabled and multiple requests.

## Cleanup

After validation, I stopped the Triton container:

```bash
docker stop easymp-triton-codex
```

GPU memory returned to idle display-only usage.

## Final State

Completed:

- Triton 26.02 base image pulled.
- Custom `easymp-vllm-omni` runtime image built.
- Codec TensorRT plan generated from `codec.onnx`.
- Default English speaker embedding `eng.pt` baked.
- Triton loaded both `codec` and `easymp` successfully.
- End-to-end gRPC request returned valid audio.

Large generated artifacts remain untracked and were not committed:

```text
context_audios/
examples/tts/easymagpie_vllm_omni/codec.onnx
examples/tts/easymagpie_vllm_omni/easymp_vllm_model/
examples/tts/easymagpie_vllm_omni/model_repository/codec/1/
```

No vLLM installation was performed in the active host NeMo development environment.
