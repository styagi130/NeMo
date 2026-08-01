# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton Python backend for EasyMagpieTTS driven by vllm-omni's AsyncOmni engine.

Wraps ``EasyMagpieTTSForConditionalGeneration`` (the vLLM-Omni talker, same model
used by the inference demo / benchmark): it streams stacked codec frames, which we
chunk-decode (overlap-save) through the ``codec`` TensorRT model.

Two request flavours share one engine (``async_chunk=True`` +
``EasyMagpieARAsyncScheduler``, a drop-in for both paths):

* **whole-text** (default) — a request carries the full ``text``; we build a single
  prompt and run ``AsyncOmni.generate(prompt, ...)``.
* **streaming-text** — the client pushes subword ``text_token`` ids across several
  requests sharing a ``stream_id`` (``stream_start`` on the first, ``stream_end`` on
  the last), with however many ids it wants per request. We forward each client
  message verbatim as one ``StreamingInput`` chunk — ``text_token`` is the whole
  ``list[int]`` and ``max_tokens == len(chunk)`` so the engine free-runs that many
  frames off the single message (prefill first, then one chunk per client message,
  then a free-running acoustic tail) into a single
  ``AsyncOmni.generate(<async-gen>, ...)`` call. All audio for the stream is sent
  back on the ``stream_start`` request's response sender; the follow-up requests
  just feed tokens and close with no output.

Both flavours converge on the same accumulator/codec pipeline:
  1. Each engine step yields the *cumulative* ``audio_codes`` ``(T_total, C*S)``
     (prefill rows + one row per decode step); we slice the decoded rows, drop the
     leading ``speech_delay`` warm-up frames, and stop at the audio EOS frame.
  2. The first packet is emitted at ``first_chunk_frames``. Later packets can use a
     bounded accumulation interval and then drain every available frame up to
     ``codec_chunk_size`` (with trimmed ``codec_left_context``) through the ``codec``
     BLS, which unstacks + index-converts + decodes them to 22.05 kHz audio chunks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import yaml
# The Triton image ships a version-matched EasyMagpie vLLM plugin.  Its package
# must remain ahead of the workspace source tree on ``sys.path``; otherwise the
# image's older StagePipelineConfig receives newer plugin-only fields.  The
# microbatch primitive is therefore exposed as a standalone module path by the
# launch scripts, with a package fallback for ordinary Python use.
try:
    from codec_microbatch import CodecMicroBatcher
except ImportError:
    from easymagpie_vllm_omni.codec_microbatch import CodecMicroBatcher

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("easymp_triton")

# Sentinel pushed onto a streaming session's token queue to signal "no more text"
# (``stream_end``): the input generator then appends text-EOS + the mask sentinel.
_STREAM_END = object()

# Sentinel pushed onto a request's codec queue when the engine generator is fully
# drained (clean end): the codec worker flushes the trailing window as final.
_GEN_DONE = object()


@dataclass
class _QueuedCodecResult:
    """A submitted codec window, retained until it can be sent in stream order."""

    future: concurrent.futures.Future
    left_context_frames: int
    pad_frames: int
    final: bool


class _StreamingSession:
    """Per-``stream_id`` state for a streaming-text request.

    ``response_sender`` is the ``stream_start`` request's sender; *all* audio for the
    stream is sent there. ``token_q`` is fed (thread-safely, via the event loop) by
    the follow-up chunk requests and drained by the input async-generator.

    ``pace_q`` is the output->input back-pressure channel: vLLM drains the input
    async-generator eagerly (a background task, decoupled from decode steps), and the
    scheduler *replaces* ``additional_information`` per chunk, so feeding faster than
    the model decodes overwrites not-yet-consumed ``text_token``s. We therefore
    release exactly one chunk per observed decode-step output: ``_drive_codec`` puts a
    token here after each step and ``_stream_inputs`` waits for one before each yield.
    """

    __slots__ = ("stream_id", "request_id", "speaker", "context_text", "response_sender", "token_q", "pace_q")

    def __init__(self, stream_id, request_id, speaker, context_text, response_sender, token_q, pace_q):
        self.stream_id = stream_id
        self.request_id = request_id
        self.speaker = speaker
        self.context_text = context_text
        self.response_sender = response_sender
        self.token_q = token_q
        self.pace_q = pace_q


def _require_param(parameters: dict, key: str) -> str:
    val = parameters.get(key)
    if isinstance(val, dict):
        val = val.get("string_value")
    if val is None:
        raise KeyError(f"Missing required model parameter: {key!r}")
    return str(val)


def _optional_param(parameters: dict, key: str, default: str) -> str:
    val = parameters.get(key)
    if isinstance(val, dict):
        val = val.get("string_value")
    return str(val) if val is not None else default


def _as_bool(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class TritonPythonModel:
    def initialize(self, args):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        self.vllm_model_path = _require_param(params, "vllm_model_path")
        self.default_speaker = _require_param(params, "default_speaker")
        self.default_context_text = _require_param(params, "default_context_text")

        self.max_model_len = int(_require_param(params, "max_model_len"))
        self.max_num_seqs = int(_require_param(params, "max_num_seqs"))
        self.max_num_batched_tokens = int(_require_param(params, "max_num_batched_tokens"))
        self.max_new_tokens = int(_require_param(params, "max_new_tokens"))
        self.gpu_memory_utilization = float(_require_param(params, "gpu_memory_utilization"))

        self.codec_chunk_size = int(_require_param(params, "codec_chunk_size"))
        self.codec_left_context = int(_require_param(params, "codec_left_context"))
        self.first_chunk_frames = int(_require_param(params, "first_chunk_frames"))
        # Preserve the legacy fixed-hop cadence at zero.  A positive value keeps
        # the first audio packet immediate, then waits at most this long for more
        # cumulative codec frames before draining everything already buffered.
        # An environment value intentionally overrides the repository default:
        # it lets an operator tune post-first coalescing without rebuilding the
        # TensorRT repository.  With no override, preserve the model-config
        # value (and therefore the legacy zero-wait behavior).
        dynamic_wait_us = os.environ.get("EASYMAGPIE_CODEC_DYNAMIC_CHUNK_WAIT_US")
        if dynamic_wait_us is None:
            dynamic_wait_us = _optional_param(params, "codec_dynamic_chunk_wait_us", "0")
        self.codec_dynamic_chunk_wait_us = max(0, int(dynamic_wait_us))
        if self.codec_chunk_size <= self.codec_left_context:
            raise ValueError(
                "codec_chunk_size must exceed codec_left_context "
                f"({self.codec_chunk_size} <= {self.codec_left_context})"
            )

        self.lt_temperature = float(_require_param(params, "lt_temperature"))
        self.lt_top_k = int(_require_param(params, "lt_top_k"))

        # Benchmark toggle: when set, skip the codec BLS entirely (no GPU->CPU copy,
        # no codec inference, no audio) so the AR/orchestration path in this model can
        # be measured in isolation against benchmark_model.py. Returns silence chunks.
        self.codec_noop = _as_bool(_optional_param(params, "codec_noop", "false"))
        # Samples emitted per model frame in codec_noop mode, used only to size the
        # silence chunks. One model frame = 2 codec frames @ 12.5 fps -> 24000/12.5 is
        # the codec rate; here 22050/12.5 = 1764 samples per model frame.
        self.codec_noop_spf = int(_optional_param(params, "codec_noop_spf", "1764"))
        # A codec window becomes ready independently in each streaming request.
        # Collect those windows explicitly before one BLS call; Triton's dynamic
        # batcher cannot reliably recover a cohort after the per-request worker
        # threads have drifted apart.
        self.codec_microbatch_wait_us = max(
            0,
            int(
                _optional_param(
                    params,
                    "codec_microbatch_wait_us",
                    # Match the codec model's dynamic-batching queue budget. A
                    # shorter 1.5 ms window leaves later streaming windows too
                    # fragmented at B32 (mean B10.8 in validation); 10 ms
                    # reaches mean B15.3 while adding only bounded TTFA.
                    os.environ.get("EASYMAGPIE_CODEC_MICROBATCH_WAIT_US", "10000"),
                )
            ),
        )
        requested_codec_batch = int(
            _optional_param(
                params,
                "codec_microbatch_max_batch_size",
                os.environ.get("EASYMAGPIE_CODEC_MICROBATCH_MAX_BATCH_SIZE", str(self.max_num_seqs)),
            )
        )
        self.codec_microbatch_max_batch_size = max(1, min(requested_codec_batch, self.max_num_seqs))
        # Keep codec windows and output audio on the GPU across the Python BLS
        # boundary. This avoids synchronizing every stream through NumPy before
        # the explicit microbatch collector has a chance to coalesce it. Set to
        # false to restore the legacy CPU staging path while diagnosing a
        # deployment.
        self.codec_device_io = _as_bool(
            os.environ.get(
                "EASYMAGPIE_CODEC_DEVICE_IO",
                _optional_param(params, "codec_device_io", "false"),
            )
        )
        # The Riva-style queue scheduler submits codec windows to the shared
        # collector without blocking each request worker. The worker keeps
        # consuming vLLM's cumulative outputs while a prior codec cohort runs;
        # rows are still sent strictly in per-stream submission order.
        self.codec_queue_scheduler = _as_bool(
            os.environ.get(
                "EASYMAGPIE_CODEC_QUEUE_SCHEDULER",
                _optional_param(params, "codec_queue_scheduler", "false"),
            )
        )

        self._load_arch_and_tokenizer()
        self._init_sampling_helpers()
        # Inferred from the first codec decode (audio_len / codec_chunk_size).
        self._spf: int | None = None

        # Active streaming-text sessions keyed by stream_id. Touched from the Triton
        # execute() thread (add/lookup) and the asyncio loop thread (removal on
        # completion), so guard it with a lock.
        self._sessions: dict[str, _StreamingSession] = {}
        self._sessions_lock = threading.Lock()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # One thread per in-flight request serializes its codec decode +
        # response_sender.send calls, off the asyncio loop and overlapping with
        # vLLM generation. Codec windows join the explicit bounded batcher below.
        self._codec_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.max_num_seqs),
            thread_name_prefix="easymp_codec",
        )
        self._codec_batcher = None
        if not self.codec_noop and self.codec_microbatch_wait_us > 0:
            self._codec_batcher = CodecMicroBatcher(
                self._decode_codec_batch,
                max_batch_size=self.codec_microbatch_max_batch_size,
                max_wait_s=self.codec_microbatch_wait_us / 1_000_000.0,
            )
        # Queued submission needs the bounded global batcher. Retain the
        # previous blocking implementation if an operator disables batching.
        self.codec_queue_scheduler = self.codec_queue_scheduler and self._codec_batcher is not None

        self._start_omni_engine()
        logger.info(
            "EasyMagpie initialized (default_speaker=%s, codec_noop=%s, codec_device_io=%s, codec_queue_scheduler=%s, codec_microbatch=%s, dynamic_wait=%dus)",
            self.default_speaker,
            self.codec_noop,
            self.codec_device_io,
            self.codec_queue_scheduler,
            (
                f"B{self.codec_microbatch_max_batch_size}/{self.codec_microbatch_wait_us}us"
                if self._codec_batcher is not None
                else "disabled"
            ),
            self.codec_dynamic_chunk_wait_us,
        )
        if self.codec_noop:
            logger.warning("codec_noop=True: codec decode is DISABLED; responses carry silence (benchmark mode).")

    def _load_arch_and_tokenizer(self):
        from transformers import AutoTokenizer

        from easymagpie_vllm_omni.config import EasyMagpieOmniArch
        from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

        config = json.loads((Path(self.vllm_model_path) / "config.json").read_text())
        cfg_obj = type("Cfg", (), config)
        arch = EasyMagpieOmniArch.from_hf_config(cfg_obj)

        self.audio_eos_id = int(arch.audio_eos_id)
        self.speech_delay = int(getattr(arch, "streaming_speech_delay", 0) or 0)
        self.num_stacked_codebooks = int(arch.num_stacked_codebooks)
        self.stop_token_id = EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(cfg_obj)
        # Appended after the client's streamed subword ids to close the text channel
        # before the free-running acoustic tail (matches the demo / benchmark).
        self.text_eos_id = int(config.get("text_vocab_size", config.get("vocab_size", 0))) - 2

        self.tokenizer = AutoTokenizer.from_pretrained(self.vllm_model_path, trust_remote_code=True)
        self._get_prompt_len = EasyMagpieTTSForConditionalGeneration.get_prompt_len

    def _init_sampling_helpers(self):
        """Cache the vLLM sampling/streaming types used by both request flavours."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        try:
            from vllm.engine.protocol import StreamingInput
        except ImportError:
            from vllm.v1.engine.async_llm import StreamingInput

        self._SamplingParams = SamplingParams
        self._RequestOutputKind = RequestOutputKind
        self._StreamingInput = StreamingInput

        # Streaming chunks reuse SamplingParams keyed by max_tokens (== chunk len):
        # one int per distinct chunk size the client sends, instead of cloning per
        # chunk. Shared read-only across requests (the scheduler only reads them).
        self._sp_cache: dict[int, object] = {}

    def _sampling_params(self, max_tokens: int):
        """Return a cached :class:`SamplingParams` for ``max_tokens`` (>=1)."""
        key = max(1, int(max_tokens))
        sp = self._sp_cache.get(key)
        if sp is None:
            sp = self._make_sampling_params(key)
            self._sp_cache[key] = sp
        return sp

    def _make_sampling_params(self, max_tokens: int):
        """SamplingParams shared by both paths (audio sampling happens in the LT).

        ``output_kind=DELTA`` is what makes ``audio_codes`` arrive as a growing list
        of per-step frames during decode; the backbone token sampler is a no-op
        (temperature 0) and stops at the audio-EOS ``stop_token_id``.
        """
        return self._SamplingParams(
            temperature=0.0,
            max_tokens=max(1, int(max_tokens)),
            detokenize=False,
            ignore_eos=True,
            stop_token_ids=[self.stop_token_id],
            output_kind=self._RequestOutputKind.DELTA,
        )

    def _build_stage_config_file(self) -> str:
        stage_cfg = {
            # async_chunk enables the streaming-text feed (one subword per chunk);
            # it is a no-op for the whole-text path, so one engine serves both.
            "async_chunk": True,
            "stage_args": [
                {
                    "stage_id": 0,
                    "stage_type": "llm",
                    "is_comprehension": True,
                    "final_output": True,
                    "final_output_type": "audio",
                    "runtime": {"devices": "0"},
                    "engine_args": {
                        "model_stage": "easymagpie",
                        "max_num_seqs": self.max_num_seqs,
                        "model_arch": "EasyMagpieTTSForConditionalGeneration",
                        "worker_type": "ar",
                        # EasyMagpie-aware scheduler: forwards each chunk's text_token
                        # and the raised acoustic-tail max_tokens for streaming-text,
                        # and is a drop-in equivalent of the stock scheduler for
                        # whole-text.
                        "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler",
                        "enforce_eager": False,
                        "trust_remote_code": True,
                        "async_scheduling": True,
                        "enable_prefix_caching": False,
                        "engine_output_type": "audio",
                        "gpu_memory_utilization": self.gpu_memory_utilization,
                        "distributed_executor_backend": "uni",
                        "max_num_batched_tokens": self.max_num_batched_tokens,
                        "max_model_len": self.max_model_len,
                        # bf16 overflows the Nemotron-H fused-MoE Triton kernel's
                        # fp32 shared memory; fp16 backbone + fp32 mamba cache.
                        "dtype": "float16",
                        "mamba_ssm_cache_dtype": "float32",
                        "attention_backend": "TRITON_ATTN",
                        # We feed prompt_token_ids directly; the model loads the
                        # bundled tokenizer to tokenize context_text + text.
                        "skip_tokenizer_init": True,
                    },
                    "default_sampling_params": {
                        # Backbone token sampler is a no-op (audio is sampled in the
                        # local transformer via additional_information temperature/top_k).
                        "temperature": 0.0,
                        "max_tokens": self.max_new_tokens,
                        "detokenize": False,
                        # Audio EOS lives in the codes; the model emits stop_token_id
                        # on the backbone stream at the EOS frame.
                        "ignore_eos": True,
                        "stop_token_ids": [self.stop_token_id],
                    },
                }
            ],
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="easymp_triton_", delete=False)
        yaml.dump(stage_cfg, tmp, sort_keys=False)
        tmp.close()
        return tmp.name

    def _start_omni_engine(self):
        from vllm_omni import AsyncOmni

        self._stage_cfg_path = self._build_stage_config_file()
        self.omni = AsyncOmni(
            model=self.vllm_model_path,
            stage_configs_path=self._stage_cfg_path,
            log_stats=False,
            stage_init_timeout=300,
        )

    def _prompt_len(self, speaker: str) -> int:
        """Placeholder ``prompt_token_ids`` length for a known speaker.

        Resolves everything from the checkpoint dir (speaker embedding +
        ``has_task_embedding``), so the caller only holds a ``speaker_id`` and
        never loads / ships the embedding itself (the engine sources it from its
        precomputed speaker table via ``speaker_id``).
        """
        return int(
            self._get_prompt_len(
                speaker,
                self.vllm_model_path,
                tokenize=lambda t: self.tokenizer.encode(t),
            )
        )

    def _build_prompt(self, text: str, context_text: str, speaker: str) -> dict:
        prompt_len = self._prompt_len(speaker)
        return {
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": {
                "speaker_id": speaker,
                "context_text": context_text,
                "text": text,
                "temperature": self.lt_temperature,
                "top_k": self.lt_top_k,
            },
        }

    def _decode_codec_batch(self, codes):
        """Run one batched codec BLS request and preserve its batch dimension."""
        if codes.ndim != 3:
            raise ValueError(f"Expected batched codec codes (B, F, Q), got {codes.shape}")

        # vLLM-Omni currently hands audio codes to this backend through its
        # inter-process connector as CPU tensors. Once the central collector
        # has formed a batch, upload that complete batch once and keep its
        # TensorRT output on the device. If a future connector delivers CUDA
        # codes directly, this path simply reuses that storage.
        device_io = self.codec_device_io
        if device_io:
            if not isinstance(codes, torch.Tensor):
                codes = torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64))
            if not codes.is_cuda:
                codes = codes.to(device="cuda", dtype=torch.int64)
            codec_input = pb_utils.Tensor.from_dlpack(
                "audio_codes", torch.utils.dlpack.to_dlpack(codes.contiguous())
            )
            # BLS otherwise permits a system-memory output even when both
            # models run on the same GPU. Prefer GPU so audio stays device-side
            # until it becomes the client response.
            preferred_memory = pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_GPU, 0)
        else:
            codec_input = pb_utils.Tensor("audio_codes", np.ascontiguousarray(codes, dtype=np.int64))
            preferred_memory = None

        request_kwargs = {}
        if preferred_memory is not None:
            request_kwargs["preferred_memory"] = preferred_memory
        response = pb_utils.InferenceRequest(
            model_name="codec",
            requested_output_names=["audio_values"],
            inputs=[codec_input],
            **request_kwargs,
        ).exec()
        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(response, "audio_values")
        if device_io:
            # ``from_dlpack`` is valid for both CPU and GPU tensors. The
            # preferred-memory request above makes this GPU in the normal path,
            # while still allowing a safe Triton fallback on older servers.
            audio = torch.from_dlpack(audio_tensor.to_dlpack())
        else:
            audio = (
                audio_tensor.as_numpy()
                if audio_tensor.is_cpu()
                else torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy()
            )
        if audio.ndim == 1:
            audio = audio[np.newaxis]
        elif audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim != 2 or audio.shape[0] != codes.shape[0]:
            raise RuntimeError(
                f"Codec returned audio shape {tuple(audio.shape)} for input batch {tuple(codes.shape)}"
            )
        return audio.contiguous() if device_io else np.ascontiguousarray(audio)

    def _prepare_codec_window(self, codes: torch.Tensor):
        """Pad one codec window without invoking the decoder.

        Keeping preparation separate from decode lets a request enqueue a window
        into the global codec scheduler and continue receiving vLLM outputs.
        """
        if self.codec_noop:
            raise RuntimeError("codec_noop windows must not be submitted to the codec scheduler")

        if self.codec_device_io and codes.is_cuda:
            # Batch and pad on-device. This deliberately replaces the legacy
            # ``detach().cpu().numpy()`` staging point when the connector
            # already supplies CUDA codes.
            codes_window = codes.detach().to(dtype=torch.int64).contiguous()
            pad = self.codec_chunk_size - codes_window.shape[0]
            if pad > 0:
                codes_window = torch.nn.functional.pad(codes_window, (0, 0, 0, pad))
        else:
            # Cast on the host (codec wants int64): retained for the rollout
            # toggle and for environments without CUDA DLPack support.
            codes_window = codes.detach().cpu().numpy().astype(np.int64, copy=False)
            pad = self.codec_chunk_size - codes_window.shape[0]
            if pad > 0:
                codes_window = np.pad(codes_window, ((0, pad), (0, 0)))
        return codes_window, int(pad)

    def _trim_codec_audio(self, audio, left_context_frames: int, pad_frames: int):
        """Trim overlap-save context and static-plan padding from one codec row."""
        if self._spf is None:
            self._spf = int(audio.shape[-1]) // self.codec_chunk_size
        left = left_context_frames * self._spf
        right = pad_frames * self._spf
        return audio[left:-right] if right > 0 else audio[left:]

    def _enqueue_codec(
        self, codes: torch.Tensor, left_context_frames: int, *, final: bool, stream_key: str
    ) -> _QueuedCodecResult:
        """Queue one fixed-shape window in the global codec collector."""
        codes_window, pad = self._prepare_codec_window(codes)
        assert self._codec_batcher is not None
        return _QueuedCodecResult(
            future=self._codec_batcher.submit(codes_window, stream_key=stream_key),
            left_context_frames=left_context_frames,
            pad_frames=pad,
            final=final,
        )

    def _decode_codec(self, codes: torch.Tensor, left_context_frames: int):
        """Blocking compatibility path: decode one window, then trim it."""
        if self.codec_noop:
            # Benchmark mode: skip the codec BLS entirely; return silence sized
            # to the real (post-left-context) frames.
            spf = self._spf or self.codec_noop_spf
            n_frames = max(0, int(codes.shape[0]) - int(left_context_frames))
            return np.zeros(n_frames * spf, dtype=np.float32)

        codes_window, pad = self._prepare_codec_window(codes)
        if self._codec_batcher is None:
            audio = self._decode_codec_batch(codes_window[None])[0]
        else:
            audio = self._codec_batcher.decode(codes_window)
        return self._trim_codec_audio(audio, left_context_frames, pad)

    def _send_audio(self, response_sender, audio, final: bool):
        if isinstance(audio, torch.Tensor):
            audio = audio.to(dtype=torch.float32).contiguous()
            output = pb_utils.Tensor.from_dlpack("audio", torch.utils.dlpack.to_dlpack(audio))
        else:
            output = pb_utils.Tensor("audio", audio.astype(np.float32))
        response_sender.send(
            pb_utils.InferenceResponse(output_tensors=[output]),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0,
        )

    def _send_error(self, response_sender, err: Exception):
        try:
            response_sender.send(
                pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(err))),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        except Exception:
            pass

    def _codec_worker(self, codec_q: queue.Queue, response_sender, state: dict, head: int, stream_key: str) -> None:
        """Per-request codec pump: runs the whole accumulate -> chunk -> decode -> send
        pipeline on a pool thread so the shared asyncio loop does no per-step tensor
        work at all.

        Queue protocol (pushed by :meth:`_drive_codec`):
          * ``(cum_codes, hit_eos)`` — the latest cumulative ``(T, C*S)`` codes tensor
            and whether the backbone audio-EOS stop token fired on that step (a cheap
            CPU flag, never a tensor scan). We keep the largest cumulative seen.
          * ``_GEN_DONE`` — the engine generator drained cleanly; flush the tail final.
          * ``None`` — error/abort; send an empty final and exit.

        Rows ``[0, head)`` are prefill + ``speech_delay`` warm-up; the first real audio
        frame is at ``head``. When the EOS step is seen its last row is the audio-EOS
        frame, so we never vocode it.
        """
        L = self.codec_left_context
        sent = 0  # real frames already vocoded + sent
        first_pending = True
        post_capacity = self.codec_chunk_size - L
        dynamic_wait_s = self.codec_dynamic_chunk_wait_us / 1_000_000.0
        post_deadline: float | None = None
        cum = None  # largest cumulative codes tensor seen
        saw_eos = False
        finalized = False
        # Riva-style split between production and decode: jobs are submitted to
        # the model-global collector immediately, while this stream continues
        # to accept newer cumulative vLLM outputs. Audio is only emitted from
        # the head of this deque, preserving stream order.
        pending_codec: deque[_QueuedCodecResult] = deque()

        def drain_completed_codec(*, wait: bool) -> None:
            nonlocal finalized
            while pending_codec:
                item = pending_codec[0]
                if not wait and not item.future.done():
                    break
                audio = item.future.result()
                pending_codec.popleft()
                audio = self._trim_codec_audio(audio, item.left_context_frames, item.pad_frames)
                self._send_audio(response_sender, audio, final=item.final)
                if state["t_first_audio"] is None:
                    state["t_first_audio"] = time.perf_counter()
                finalized = finalized or item.final

        def emit_ready(real_count: int, *, final: bool, flush_post: bool) -> None:
            """Vocode ready frames, preserving a low-latency first packet."""
            nonlocal sent, first_pending, finalized, post_deadline
            while sent < real_count:
                remaining = real_count - sent
                if first_pending:
                    if not final and remaining < self.first_chunk_frames:
                        break
                    take = min(self.first_chunk_frames, remaining)
                    first_pending = False
                elif final:
                    take = min(post_capacity, remaining)
                elif remaining >= post_capacity:
                    take = post_capacity
                elif flush_post:
                    # The bounded dynamic deadline elapsed.  Do not wait for a
                    # fixed hop: this is the exact amount of buffered speech.
                    take = remaining
                else:
                    break
                ctx = min(sent, L)
                chunk = cum[head + sent - ctx : head + sent + take]
                sent += take
                is_final = final and sent >= real_count
                if self.codec_queue_scheduler:
                    pending_codec.append(self._enqueue_codec(chunk, ctx, final=is_final, stream_key=stream_key))
                else:
                    audio = self._decode_codec(chunk, ctx)
                    self._send_audio(response_sender, audio, final=is_final)
                    if state["t_first_audio"] is None:
                        state["t_first_audio"] = time.perf_counter()
                    finalized = finalized or is_final
                if not first_pending:
                    # A full post-first drain starts a new interval only after a
                    # later codec frame actually arrives.
                    post_deadline = None

        try:
            done = False
            while True:
                drain_completed_codec(wait=False)
                # Once post-first frames exist, use the queue timeout as the
                # bounded coalescing timer.  The first packet never enters this
                # path, so TTFA remains governed only by first_chunk_frames.
                timeout = None
                timed_out = False
                if pending_codec:
                    # Do not wait indefinitely for a new vLLM output while a
                    # completed codec row is ready to stream to the client.
                    timeout = 0.001
                if cum is not None and not done and not first_pending and dynamic_wait_s > 0:
                    real_avail = max(0, cum.shape[0] - head - (1 if saw_eos else 0))
                    remaining = real_avail - sent
                    if 0 < remaining < post_capacity:
                        now = time.monotonic()
                        if post_deadline is None:
                            post_deadline = now + dynamic_wait_s
                        codec_timeout = max(0.0, post_deadline - now)
                        timeout = codec_timeout if timeout is None else min(timeout, codec_timeout)

                try:
                    # Block for the next item, then drain any backlog so we only
                    # act on the most recent cumulative codes (older snapshots
                    # are subsets of it).
                    first_item = codec_q.get(timeout=timeout) if timeout is not None else codec_q.get()
                    batch = [first_item]
                except queue.Empty:
                    batch = []
                    timed_out = True
                while True:
                    try:
                        batch.append(codec_q.get_nowait())
                    except queue.Empty:
                        break
                for item in batch:
                    if item is _GEN_DONE:
                        done = True
                    elif item is None:
                        self._send_audio(response_sender, np.array([], dtype=np.float32), final=True)
                        finalized = True
                        return
                    else:
                        cum_now, hit_eos = item
                        if cum is None or cum_now.shape[0] > cum.shape[0]:
                            cum = cum_now
                        saw_eos = saw_eos or hit_eos

                if state["error"] is not None:
                    return
                if cum is not None:
                    real_avail = max(0, cum.shape[0] - head - (1 if saw_eos else 0))
                    if done:
                        emit_ready(real_avail, final=True, flush_post=True)
                    elif real_avail > sent:
                        emit_ready(real_avail, final=False, flush_post=False)
                        remaining = real_avail - sent
                        if not first_pending and dynamic_wait_s > 0 and 0 < remaining < post_capacity:
                            if post_deadline is not None and time.monotonic() >= post_deadline:
                                emit_ready(real_avail, final=False, flush_post=True)
                            elif post_deadline is None:
                                post_deadline = time.monotonic() + dynamic_wait_s
                if done:
                    # All final windows have been submitted. Only this terminal
                    # path waits for codec completion, so steady-state vLLM
                    # production never serializes on the BLS call.
                    drain_completed_codec(wait=True)
                    if not finalized:
                        self._send_audio(response_sender, np.array([], dtype=np.float32), final=True)
                        finalized = True
                    return
        except Exception as e:
            state["error"] = e
            if not finalized:
                self._send_error(response_sender, e)

    @staticmethod
    def _cumulative_codes(payload):
        """Reduce one step's ``audio_codes`` payload to the cumulative ``(T, C*S)``.

        DELTA decode surfaces a growing list ``[cum_so_far, new_frame, ...]``, but
        every finished segment consolidates to a single cumulative tensor — and with
        ``max_tokens=1`` (streaming-text) *each* fed token finishes its segment, so
        most steps arrive as a tensor, not a list. Both forms reduce to the full
        cumulative here (cat the list / take the tensor); callers keep the largest.
        """
        if isinstance(payload, list):
            parts = [t for t in payload if isinstance(t, torch.Tensor) and t.numel() > 0]
            return torch.cat(parts, dim=0) if parts else None
        if isinstance(payload, torch.Tensor) and payload.numel() > 0:
            return payload
        return None

    async def _drive_codec(
        self, gen, response_sender, request_id: str, speaker: str, text: str, prompt_len: int, pace_q=None
    ):
        """Drain one omni request's per-step ``audio_codes`` and stream audio out.

        ``gen`` is the ``AsyncOmni.generate(...)`` async iterator (whole-text or
        streaming-text); from here on both flavours are identical. All audio is sent
        on ``response_sender`` (for streaming that is the ``stream_start`` sender).

        This coroutine stays deliberately thin: per step it only reads the cumulative
        ``audio_codes`` reference and a cheap CPU end-of-stream flag, then hands both
        to the per-request :meth:`_codec_worker` thread. All tensor slicing, the
        device->host copy, codec inference and ``response_sender.send`` happen on that
        pool thread, so the single shared event loop never does per-step tensor work
        (and never blocks on a GPU sync) — that is what lets many requests share the
        loop without serializing.

        Each step yields the *cumulative* codes ``(prompt_len prefill rows + decoded
        frames, C*S)`` (as a list to cat or an already-consolidated tensor); the first
        real audio frame is at row ``prompt_len + speech_delay``. End of stream is the
        backbone audio-EOS stop token, surfaced as ``outputs[0].stop_reason`` (a CPU
        attribute) — no per-step scan of the codes tensor is needed.

        For streaming-text, ``pace_q`` carries one token per observed decode-step
        output so the input feeder releases the next chunk only after the previous one
        has been decoded (see :class:`_StreamingSession`); ``None`` for whole-text.
        """
        t_start = time.perf_counter()

        codec_q: queue.Queue = queue.Queue()
        state: dict = {"t_first_audio": None, "error": None}
        head = prompt_len + self.speech_delay  # cumulative row of the first real frame
        codec_future = self._codec_pool.submit(self._codec_worker, codec_q, response_sender, state, head, request_id)
        sent = 0  # forwarded steps (for the log line only)

        try:
            async for out in gen:
                # Release the next input chunk for each decode-step output (one chunk
                # produces one frame); harmless extra puts during the acoustic tail.
                if pace_q is not None:
                    pace_q.put_nowait(True)
                if state["error"] is not None:
                    break
                payload = (getattr(out, "multimodal_output", None) or {}).get("audio_codes")
                cum_now = self._cumulative_codes(payload)
                if cum_now is None:
                    continue
                # The request truly ends at the backbone audio-EOS stop token; vLLM
                # already detected it and reports it as the matched stop_reason. When it
                # fires this step's last row is the EOS frame (which the worker drops).
                co = out.outputs[0] if getattr(out, "outputs", None) else None
                hit_eos = getattr(co, "stop_reason", None) == self.stop_token_id
                sent += 1
                codec_q.put((cum_now, hit_eos))

            if state["error"] is None:
                codec_q.put(_GEN_DONE)
            else:
                codec_q.put(None)

            await asyncio.wrap_future(codec_future)
            if state["error"] is not None:
                raise state["error"]

            t_end = time.perf_counter()
            ttfa_ms = ((state["t_first_audio"] or t_end) - t_start) * 1000
            logger.info(
                "rid=%s ttfa=%.1fms total=%.1fms steps=%d speaker=%s text=%r",
                request_id,
                ttfa_ms,
                (t_end - t_start) * 1000,
                sent,
                speaker,
                text[:120],
            )
        except Exception as e:
            logger.error("rid=%s failed: %s", request_id, e, exc_info=True)
            try:
                await self.omni.abort(request_id)
            except Exception:
                pass
            if not codec_future.done():
                codec_q.put(None)
                try:
                    await asyncio.wrap_future(codec_future)
                except Exception:
                    pass
            self._send_error(response_sender, e)
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

    async def _synthesize_whole_text(self, text: str, context_text: str, speaker: str, response_sender):
        request_id = f"easymp-{uuid.uuid4().hex[:8]}"
        prompt = self._build_prompt(text, context_text, speaker)
        prompt_len = len(prompt["prompt_token_ids"])
        gen = self.omni.generate(
            prompt,
            sampling_params_list=[self._make_sampling_params(self.max_new_tokens)],
            request_id=request_id,
        )
        await self._drive_codec(gen, response_sender, request_id, speaker, text, prompt_len)

    def _text_chunk(self, text_tokens, sampling_params):
        """One ``StreamingInput`` carrying a whole chunk of ids as a ``list[int]``."""
        return self._StreamingInput(
            prompt={
                "prompt_token_ids": [0],
                "additional_information": {"text_token": [int(t) for t in text_tokens]},
            },
            sampling_params=sampling_params,
        )

    async def _stream_inputs(self, session: _StreamingSession):
        """Yield ``StreamingInput`` chunks for a streaming-text session.

        Prefill (speaker + context, no text) emits one frame, then we forward each
        client message verbatim as one chunk: ``text_token`` is the whole
        ``list[int]`` and ``max_tokens == len(chunk)`` so the engine free-runs that
        many frames off the single message (the model appends the ids to its buffer
        and consumes one per frame). Once the client signals ``stream_end`` we append
        the text-EOS id and then a ``[]`` tail chunk whose raised ``max_tokens`` lets
        the model free-run the acoustic tail to audio-EOS.

        Each post-prefill chunk is gated on ``session.pace_q`` — one decode-step
        output per emitted frame, ``prev_frames`` of them — so the eagerly-drained
        feed can't overwrite a chunk's ``text_token`` before the model has consumed
        all of it.
        """
        StreamingInput = self._StreamingInput
        prompt_len = self._prompt_len(session.speaker)
        sp1 = self._sampling_params(1)

        prefill_info = {
            "speaker_id": session.speaker,
            "context_text": session.context_text,
            "temperature": self.lt_temperature,
            "top_k": self.lt_top_k,
        }
        # Prefill is released immediately; its single decode-step output unblocks the
        # first text chunk (mirrors the demo's go_queue handshake).
        yield StreamingInput(
            prompt={"prompt_token_ids": [0] * prompt_len, "additional_information": prefill_info},
            sampling_params=sp1,
        )
        # Frames the last-yielded segment will emit (prefill -> 1); each gates the
        # next chunk on that many pace_q tokens.
        prev_frames = 1

        n_text = 0
        ended = False
        while not ended:
            item = await session.token_q.get()
            if item is _STREAM_END:
                ended = True
                break
            chunk = [int(t) for t in item]
            if not chunk:
                continue
            for _ in range(prev_frames):
                await session.pace_q.get()
            yield self._text_chunk(chunk, self._sampling_params(len(chunk)))
            prev_frames = len(chunk)
            n_text += len(chunk)

        for _ in range(prev_frames):
            await session.pace_q.get()
        yield self._text_chunk([self.text_eos_id], sp1)
        prev_frames = 1
        n_text += 1

        for _ in range(prev_frames):
            await session.pace_q.get()
        tail_budget = self.max_new_tokens - n_text
        yield self._text_chunk([], self._sampling_params(tail_budget))

    async def _synthesize_streaming(self, session: _StreamingSession):
        prompt_len = self._prompt_len(session.speaker)
        inputs_gen = self._stream_inputs(session)
        gen = self.omni.generate(
            inputs_gen, sampling_params_list=[self._sampling_params(1)], request_id=session.request_id
        )
        try:
            await self._drive_codec(
                gen,
                session.response_sender,
                session.request_id,
                session.speaker,
                "<streaming>",
                prompt_len,
                pace_q=session.pace_q,
            )
        finally:
            try:
                await inputs_gen.aclose()
            except Exception:
                pass
            with self._sessions_lock:
                self._sessions.pop(session.stream_id, None)

    @staticmethod
    def _log_future_exception(future: concurrent.futures.Future) -> None:
        try:
            exc = future.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            logger.error("synthesis task crashed: %s", exc, exc_info=exc)

    @staticmethod
    def _read_str(request, name: str, default: str) -> str:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return tensor.as_numpy().flatten()[0].decode("utf-8")

    @staticmethod
    def _read_bool(request, name: str, default: bool) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        return bool(tensor.as_numpy().flatten()[0])

    @staticmethod
    def _read_int_list(request, name: str) -> list:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return []
        return [int(x) for x in tensor.as_numpy().flatten().tolist()]

    def _feed_session(self, session: _StreamingSession, item) -> None:
        """Hand ``item`` (a list of token ids or ``_STREAM_END``) to the session's
        queue on the event-loop thread (asyncio.Queue is not thread-safe)."""
        self._loop.call_soon_threadsafe(session.token_q.put_nowait, item)

    def _launch(self, coro) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(self._log_future_exception)

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                stream_id = self._read_str(request, "stream_id", "")
                if stream_id:
                    self._handle_streaming(request, stream_id, response_sender)
                else:
                    self._handle_whole_text(request, response_sender)
            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                self._send_error(response_sender, e)
        return None

    def _handle_whole_text(self, request, response_sender) -> None:
        text = self._read_str(request, "text", "")
        context_text = self._read_str(request, "context_text", self.default_context_text)
        speaker = self._read_str(request, "speaker", self.default_speaker)
        self._launch(self._synthesize_whole_text(text, context_text, speaker, response_sender))

    def _handle_streaming(self, request, stream_id: str, response_sender) -> None:
        """Route one chunk of a streaming-text request.

        ``stream_start`` opens a session (launching one generate() call whose audio
        streams back on *this* response sender) and ``stream_end`` closes the text
        feed. Follow-up chunks only push tokens, then immediately complete their own
        (output-less) response so the client side of the stream stays tidy.
        """
        start = self._read_bool(request, "stream_start", False)
        end = self._read_bool(request, "stream_end", False)
        tokens = self._read_int_list(request, "text_token")

        if start:
            context_text = self._read_str(request, "context_text", self.default_context_text)
            speaker = self._read_str(request, "speaker", self.default_speaker)
            session = _StreamingSession(
                stream_id=stream_id,
                request_id=f"easymp-stream-{uuid.uuid4().hex[:8]}",
                speaker=speaker,
                context_text=context_text,
                response_sender=response_sender,
                token_q=asyncio.Queue(),
                pace_q=asyncio.Queue(),
            )
            with self._sessions_lock:
                self._sessions[stream_id] = session
            # All audio for the stream is sent on this (stream_start) sender by the
            # coroutine; do NOT complete it here.
            self._launch(self._synthesize_streaming(session))
            if tokens:
                self._feed_session(session, tokens)
            if end:
                self._feed_session(session, _STREAM_END)
            return

        with self._sessions_lock:
            session = self._sessions.get(stream_id)
        if session is not None:
            if tokens:
                self._feed_session(session, tokens)
            if end:
                self._feed_session(session, _STREAM_END)
        else:
            logger.warning("Streaming chunk for unknown/closed stream_id=%s", stream_id)
        # Follow-up chunks produce no audio of their own; close this response.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        if getattr(self, "_codec_batcher", None) is not None:
            stats = self._codec_batcher.stats()
            logger.info(
                "EasyMagpie codec microbatch summary: calls=%d items=%d mean_batch=%.2f histogram=%s",
                stats["calls"],
                stats["items"],
                stats["mean_batch_size"],
                stats["histogram"],
            )
            self._codec_batcher.close()
        if hasattr(self, "omni"):
            try:
                self.omni.shutdown()
            except Exception:
                pass
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_codec_pool"):
            self._codec_pool.shutdown(wait=False)
        if getattr(self, "_stage_cfg_path", None):
            try:
                os.unlink(self._stage_cfg_path)
            except OSError:
                pass
