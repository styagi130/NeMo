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
"""Bounded batching primitives for streaming EasyMagpie codec windows.

Streaming TTS exposes one codec window per request.  Letting every worker submit
that window immediately fragments a nominal B16/B32 codec engine into tiny GPU
launches.  This module provides two complementary fixes:

* :class:`CodecMicroBatcher` explicitly batches concurrent Triton BLS codec calls.
* :func:`install_transfer_microbatch_patch` coalesces and parallelizes the pure
  vLLM-Omni Stage-0 -> Stage-1 handoff, while preserving ordering within a stream.
* :func:`install_scheduler_microbatch_patch` holds a newly-ready Stage-1 codec
  cohort for the same bounded deadline before vLLM schedules its GPU decode.

Both use a small, bounded deadline rather than waiting for a full batch, so a
single low-concurrency request keeps its streaming behaviour.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np
from easymagpie_vllm_omni.profiling import nvtx_profiled, nvtx_range

try:  # Keep the primitive importable in the lightweight CPU unit-test environment.
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:  # pragma: no cover - exercised only without vLLM installed.
    logger = logging.getLogger(__name__)

_STOP = object()
_Task = TypeVar("_Task")


@dataclass
class _CodecWork:
    """One queued codec window and its eventual row of batched output."""

    # ``codes`` and ``output`` are normally NumPy arrays, but Triton's Python
    # backend can also pass CUDA torch tensors through DLPack.  Keeping this
    # primitive framework-neutral lets that backend batch on the device while
    # retaining the CPU-only behaviour used by the standalone tests.
    codes: Any
    future: concurrent.futures.Future
    stream_key: str | None = None


class CodecMicroBatcher:
    """Collect fixed-shape codec windows into explicit bounded batches.

    Args:
        decode_batch: Callable receiving ``(B, F, Q)`` int64 codes and returning
            one waveform row per item.
        max_batch_size: Largest batch passed to ``decode_batch``.
        max_wait_s: Maximum time the oldest request waits for peer windows.

    Each caller blocks only until its own batch is decoded.  The worker keeps
    differently shaped windows separate; normal EasyMagpie serving pads all
    windows to one fixed shape, so a single worker produces B16/B32 batches.
    """

    def __init__(
        self,
        decode_batch: Callable[[np.ndarray], np.ndarray],
        *,
        max_batch_size: int,
        max_wait_s: float,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be non-negative")

        self._decode_batch = decode_batch
        self._max_batch_size = int(max_batch_size)
        self._max_wait_s = float(max_wait_s)
        self._queue: queue.Queue[_CodecWork | object] = queue.Queue()
        self._deferred: deque[_CodecWork] = deque()
        self._closed = False
        self._close_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._batch_hist: Counter[int] = Counter()
        self._batch_calls = 0
        self._batch_items = 0
        self._worker = threading.Thread(target=self._run, name="easymagpie_codec_microbatch", daemon=True)
        self._worker.start()

    def submit(self, codes: Any, *, stream_key: str | None = None) -> concurrent.futures.Future:
        """Queue one window and return a future for its decoded waveform row.

        Unlike :meth:`decode`, this method never waits for the BLS codec call.
        A per-stream producer can therefore queue a later codec window while a
        prior cohort is executing, which is necessary for cohort-level rather
        than request-worker-level batching.
        """
        if isinstance(codes, np.ndarray):
            array = np.ascontiguousarray(codes, dtype=np.int64)
        elif getattr(codes, "is_cuda", False):
            # The caller has already converted the tiny code window to int64
            # on-device. Do not call ``cpu()`` here: that would synchronize all
            # request workers before the collector can form a batch.
            array = codes.detach().contiguous()
        else:
            raise TypeError("CodecMicroBatcher expects a NumPy array or CUDA tensor")
        if array.ndim != 2:
            raise ValueError(f"CodecMicroBatcher expects (frames, codebooks), got {array.shape}")

        with self._close_lock:
            if self._closed:
                raise RuntimeError("CodecMicroBatcher is closed")
            item = _CodecWork(codes=array, future=concurrent.futures.Future(), stream_key=stream_key)
            self._queue.put(item)
        return item.future

    def decode(self, codes: Any) -> Any:
        """Blocking compatibility wrapper around :meth:`submit`."""
        return self.submit(codes).result()

    def stats(self) -> dict[str, Any]:
        """Return a snapshot suitable for a serving diagnostic log."""
        with self._stats_lock:
            calls = self._batch_calls
            items = self._batch_items
            return {
                "calls": calls,
                "items": items,
                "mean_batch_size": items / calls if calls else 0.0,
                "histogram": dict(sorted(self._batch_hist.items())),
            }

    def close(self) -> None:
        """Stop the worker after all queued work has been resolved."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._worker.join(timeout=10)

    def _next_item(self, timeout: float | None = None) -> _CodecWork | object:
        if self._deferred:
            return self._deferred.popleft()
        if timeout is None:
            return self._queue.get()
        return self._queue.get(timeout=timeout)

    def _collect(self, first: _CodecWork) -> list[_CodecWork]:
        """Collect peers of ``first`` until the deadline or batch capacity."""
        items = [first]
        seen_streams = {first.stream_key} if first.stream_key is not None else set()
        # Windows with equal shape but different storage types/devices cannot
        # be stacked together. Normal serving has one CUDA device, while the
        # existing NumPy mode remains supported for unit tests and fallback.
        shape = (type(first.codes), str(getattr(first.codes, "device", "cpu")), tuple(first.codes.shape))
        deadline = time.monotonic() + self._max_wait_s

        while len(items) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                # Only inspect newly queued work while forming this batch.  A
                # differently shaped deferred item is deliberately held for a
                # later batch; repeatedly popping and re-deferring it would
                # otherwise starve compatible work that is already in the
                # queue.
                next_item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_item is _STOP:
                # Reinsert the stop marker so the outer loop exits after this
                # batch has completed and all callers receive their result.
                self._queue.put(_STOP)
                break
            assert isinstance(next_item, _CodecWork)
            next_shape = (
                type(next_item.codes),
                str(getattr(next_item.codes, "device", "cpu")),
                tuple(next_item.codes.shape),
            )
            # The queued scheduler can submit later windows from a fast stream
            # while another stream is still producing its first one. Admit at
            # most one window per stream to a cohort; otherwise that fast
            # stream monopolizes the batch and recreates B1 fragmentation for
            # its peers. The blocking compatibility path has no stream key.
            if next_shape == shape and (next_item.stream_key is None or next_item.stream_key not in seen_streams):
                items.append(next_item)
                if next_item.stream_key is not None:
                    seen_streams.add(next_item.stream_key)
            else:
                self._deferred.append(next_item)
        return items

    def _finish(self, items: list[_CodecWork], outputs: Any | None, error: BaseException | None) -> None:
        if error is None:
            assert outputs is not None
            if outputs.ndim == 0 or int(outputs.shape[0]) != len(items):
                error = RuntimeError(
                    f"Codec microbatch returned shape {tuple(outputs.shape)} for {len(items)} input windows"
                )
        for row, item in enumerate(items):
            if error is None:
                if isinstance(outputs, np.ndarray):
                    # Retain an independent row: the next BLS output may reuse
                    # its backing allocation before this request sends audio.
                    output = np.ascontiguousarray(outputs[row]).copy()
                else:
                    # A tensor view retains the DLPack-owned output storage, so
                    # it stays valid after this method returns without a GPU
                    # device-to-device clone per request.
                    output = outputs[row].contiguous()
                item.future.set_result(output)
            else:
                item.future.set_exception(error)

    def _run(self) -> None:
        while True:
            next_item = self._next_item()
            if next_item is _STOP:
                return
            assert isinstance(next_item, _CodecWork)
            items = self._collect(next_item)
            try:
                if isinstance(items[0].codes, np.ndarray):
                    batch = np.stack([item.codes for item in items], axis=0)
                    # The Triton codec may return a CUDA tensor even when the
                    # collector received CPU windows (one batched H2D upload).
                    # Preserve its storage instead of forcing a GPU->CPU copy.
                    outputs = self._decode_batch(batch)
                else:
                    # Import lazily so this module remains importable in the
                    # lightweight CPU-only unit-test environment.
                    import torch

                    batch = torch.stack([item.codes for item in items], dim=0)
                    outputs = self._decode_batch(batch)
            except BaseException as error:  # Resolve every waiting request before continuing.
                self._finish(items, None, error)
                continue

            with self._stats_lock:
                self._batch_calls += 1
                self._batch_items += len(items)
                self._batch_hist[len(items)] += 1
            self._finish(items, outputs, None)


def group_pending_tasks(tasks: list[_Task]) -> OrderedDict[str, list[_Task]]:
    """Partition pending transfer tasks, preserving order within each stream."""
    groups: OrderedDict[str, list[_Task]] = OrderedDict()
    for task in tasks:
        request = task.get("request") if isinstance(task, dict) else getattr(task, "request", None)
        request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
        if request_id is None:
            request_id = f"anonymous-{id(request)}"
        groups.setdefault(str(request_id), []).append(task)
    return groups


def _connector_extra(adapter: Any) -> dict[str, Any]:
    return _model_extra(getattr(adapter, "config", None))


def _model_extra(model_config: Any) -> dict[str, Any]:
    connector_config = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_config, dict):
        extra = connector_config.get("extra", connector_config)
    else:
        extra = getattr(connector_config, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _transfer_microbatch_settings(adapter: Any) -> tuple[float, int]:
    """Return configured bounded-wait and worker count for a Stage-0 adapter."""
    extra = _connector_extra(adapter)
    try:
        wait_us = int(extra.get("codec_microbatch_wait_us", 0) or 0)
    except (TypeError, ValueError):
        wait_us = 0
    try:
        workers = int(
            os.getenv(
                "EASYMAGPIE_CODEC_TRANSFER_WORKERS",
                str(extra.get("codec_microbatch_parallelism", 0) or 0),
            )
        )
    except (TypeError, ValueError):
        workers = 0
    max_seqs = int(getattr(adapter, "scheduler_max_num_seqs", 1) or 1)
    # Do not impose a hidden global cap here.  The deploy profile already
    # bounds both the requested worker count and Stage-0 admission through
    # ``scheduler_max_num_seqs``.  A fixed B64 ceiling serialized connector
    # publication into two/four waves for matched B128/B256 profiles.
    return max(0.0, wait_us / 1_000_000.0), max(1, min(workers or max_seqs, max_seqs))


def _scheduler_microbatch_settings(scheduler: Any) -> tuple[float, int]:
    """Return bounded wait and target codec batch size for a Stage-1 scheduler."""
    vllm_config = getattr(scheduler, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    extra = _model_extra(model_config)
    try:
        wait_us = int(extra.get("codec_microbatch_wait_us", 0) or 0)
    except (TypeError, ValueError):
        wait_us = 0
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_seqs = int(
        getattr(scheduler, "max_num_running_reqs", 0)
        or getattr(scheduler_config, "max_num_seqs", 0)
        or 1
    )
    try:
        target_batch = int(
            os.getenv(
                "EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE",
                str(extra.get("codec_microbatch_max_batch_size", max_seqs) or max_seqs),
            )
        )
    except (TypeError, ValueError):
        target_batch = max_seqs
    return max(0.0, wait_us / 1_000_000.0), max(1, min(target_batch, max_seqs))


def _as_flat_ints(value: Any) -> list[int]:
    """Return a CPU-owned, one-dimensional sequence of codec token ids.

    Connector payloads are normally Python lists at this point, but accepting a
    CPU tensor keeps the scheduler-side patch compatible with both the shared
    memory and in-process connector implementations.  This function purposely
    avoids importing torch so the microbatch primitives remain CPU-testable.
    """
    if value is None:
        return []
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        return []

    def flatten(items: list | tuple) -> list[int]:
        result: list[int] = []
        for item in items:
            if isinstance(item, (list, tuple)):
                result.extend(flatten(item))
            else:
                result.append(int(item))
        return result

    # Some connector serializers add an outer ``[1, Q*F]`` batch dimension.
    # Flattening that representation is safe; the values remain codebook-major.
    return flatten(value)


def _scalar_int(value: Any, default: int = 0) -> int:
    """Read an integer from connector metadata without a torch dependency."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError):
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def merge_codec_windows(
    current_ids: Any,
    next_ids: Any,
    *,
    num_quantizers: int,
    next_left_context_frames: int,
    max_frames: int,
) -> Any | None:
    """Append a connector window to a contiguous codec buffer.

    The transfer payload is codebook-major ``[Q * F]``.  A later source window
    repeats its left context, so only ``next[left_context:]`` represents new
    audio.  The result retains the current window's left context; Code2Wav then
    trims that overlap exactly once after the one combined decoder call.

    ``None`` means the next window would exceed the fixed codec artifact.  The
    caller must leave that payload for the next decode rather than truncate it.
    """
    q = int(num_quantizers)
    fixed = int(max_frames)
    if q <= 0 or fixed <= 0:
        raise ValueError("num_quantizers and max_frames must be positive")

    device_resident = bool(getattr(current_ids, "is_cuda", False)) or bool(getattr(next_ids, "is_cuda", False))
    if device_resident:
        if not (bool(getattr(current_ids, "is_cuda", False)) and bool(getattr(next_ids, "is_cuda", False))):
            raise ValueError("Cannot merge CUDA and CPU codec windows")

        def as_codebook_major_flat(value: Any) -> Any:
            tensor = value.detach()
            # The Stage-0 frame ring publishes [F, Q] views. Convert only at
            # the point where Stage 1 genuinely needs to merge windows; the
            # normal single-window codec path consumes that layout directly.
            if getattr(tensor, "ndim", 0) == 2 and int(tensor.shape[1]) == q:
                return tensor.transpose(0, 1).reshape(-1).contiguous()
            return tensor.reshape(-1).contiguous()

        current = as_codebook_major_flat(current_ids)
        following = as_codebook_major_flat(next_ids)
    else:
        current = _as_flat_ints(current_ids)
        following = _as_flat_ints(next_ids)
    current_len = int(current.numel()) if device_resident else len(current)
    following_len = int(following.numel()) if device_resident else len(following)
    if current_len == 0 or following_len == 0 or current_len % q or following_len % q:
        raise ValueError("Codec windows must be non-empty and divisible by num_quantizers")

    current_frames = current_len // q
    following_frames = following_len // q
    left = int(next_left_context_frames)
    if left < 0 or left > following_frames or left > current_frames:
        raise ValueError(
            "Invalid codec left context: "
            f"left={left}, current_frames={current_frames}, next_frames={following_frames}"
        )
    new_frames = following_frames - left
    if current_frames + new_frames > fixed:
        return None

    if device_resident:
        import torch

        # Both tensors are codebook-major [Q * F]. This is one small D2D
        # concatenate on Stage 1, not a CPU list materialization or a stream
        # synchronization to inspect every individual codec id.
        return torch.cat(
            (current.reshape(q, current_frames), following.reshape(q, following_frames)[:, left:]), dim=1
        ).reshape(-1).contiguous()

    merged: list[int] = []
    for codebook in range(q):
        current_start = codebook * current_frames
        following_start = codebook * following_frames
        current_codes = current[current_start : current_start + current_frames]
        following_codes = following[following_start : following_start + following_frames]
        if left and current_codes[-left:] != following_codes[:left]:
            raise ValueError("Codec windows are not contiguous at the declared left-context boundary")
        merged.extend(current_codes)
        merged.extend(following_codes[left:])
    return merged


def _codec_dynamic_window_settings(scheduler: Any) -> tuple[int, int, int, float]:
    """Return fixed capacity, codebooks, source hop, and post-first wait."""
    vllm_config = getattr(scheduler, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    extra = _model_extra(model_config)
    fixed = _scalar_int(extra.get("codec_fixed_chunk_frames"), 0)
    hop = _scalar_int(extra.get("codec_chunk_frames"), 0)
    left = _scalar_int(extra.get("codec_left_context_frames"), 0)
    if fixed <= 0:
        fixed = max(0, hop + left)

    hf_config = getattr(model_config, "hf_config", None)
    quantizers = _scalar_int(getattr(hf_config, "num_stacked_codebooks", 0), 0)
    if quantizers <= 0:
        quantizers = _scalar_int(getattr(hf_config, "num_audio_codebooks", 0), 0) * _scalar_int(
            getattr(hf_config, "frame_stacking_factor", 1), 1
        )

    # Prefer microseconds so the pure and Triton paths use the same unit.  A
    # millisecond spelling is accepted for hand-written deployment files.
    wait_override = os.environ.get("EASYMAGPIE_CODEC_DYNAMIC_CHUNK_WAIT_US")
    wait_us = _scalar_int(wait_override, -1) if wait_override is not None else _scalar_int(
        extra.get("codec_dynamic_chunk_wait_us"), -1
    )
    if wait_us < 0:
        wait_us = _scalar_int(extra.get("codec_dynamic_chunk_wait_ms"), 0) * 1_000
    return fixed, quantizers, hop, max(0.0, wait_us / 1_000_000.0)


def _codec_dynamic_enabled(scheduler: Any) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    value = _model_extra(model_config).get("codec_dynamic_chunking", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _codec_dynamic_wait_only_underfilled(scheduler: Any) -> bool:
    """Whether the post-first drain delay is skipped for a full cohort."""
    vllm_config = getattr(scheduler, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    value = os.environ.get(
        "EASYMAGPIE_CODEC_DYNAMIC_WAIT_ONLY_UNDERFILLED",
        _model_extra(model_config).get("codec_dynamic_wait_only_underfilled", False),
    )
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _codec_cohort_state(scheduler: Any) -> dict[str, Any]:
    """Return the non-blocking Stage-1 codec cohort state.

    The vLLM EngineCore keeps calling ``schedule`` while a request is waiting
    for a chunk.  Keeping the ready request ids out of the adapter's ready set
    until a monotonic deadline therefore gives us a bounded coalescing window
    without sleeping the scheduler thread.
    """
    state = getattr(scheduler, "_easymagpie_codec_cohort_state", None)
    if not isinstance(state, dict):
        state = {"ids": set(), "deadline": 0.0, "kind": None}
        scheduler._easymagpie_codec_cohort_state = state
    return state


def _prepare_codec_cohort_for_schedule(scheduler: Any) -> str | None:
    """Hold an underfilled homogeneous cohort until its deadline.

    Returns the cohort kind that is ready to be scheduled (``"initial"`` or
    ``"dynamic"``), or ``None`` when normal scheduling should proceed.  A
    mixed first/post-first ready set deliberately falls back to normal vLLM
    scheduling: withholding one side would otherwise reorder a stream.
    """
    if not _is_easymagpie_codec_scheduler(scheduler):
        return None
    adapter = getattr(scheduler, "chunk_transfer_adapter", None)
    ready_set = getattr(adapter, "_finished_load_reqs", None) if adapter is not None else None
    if not isinstance(ready_set, set):
        return None

    tracked = set(getattr(scheduler, "requests", {}) or {})
    initial_ids = getattr(scheduler, "_easymagpie_codec_initial_ids", None)
    if initial_ids is None:
        initial_ids = set()
        scheduler._easymagpie_codec_initial_ids = initial_ids
    initial_ids.intersection_update(tracked)

    state = _codec_cohort_state(scheduler)
    held = state["ids"]
    held.intersection_update(tracked)
    if not held:
        state.update({"deadline": 0.0, "kind": None})

    ready = set(ready_set).intersection(tracked)
    if not ready and not held:
        return None

    # A request's first one-frame packet is identified before it ever reaches
    # the normal scheduler.  Every later packet belongs to the dynamic path.
    initial_ready = ready - initial_ids
    dynamic_ready = ready & initial_ids
    if state["kind"] is None:
        if initial_ready and dynamic_ready:
            return None
        kind = "initial" if initial_ready else "dynamic"
        cohort = initial_ready if initial_ready else dynamic_ready
        if not cohort:
            return None
        if kind == "initial":
            wait_s, target = _scheduler_microbatch_settings(scheduler)
        else:
            _fixed, _quantizers, _hop, wait_s = _codec_dynamic_window_settings(scheduler)
            _micro_wait, target = _scheduler_microbatch_settings(scheduler)
            if _codec_dynamic_wait_only_underfilled(scheduler) and len(cohort) >= target:
                wait_s = 0.0
        state.update({"ids": set(cohort), "deadline": time.monotonic() + wait_s, "kind": kind})
        held = state["ids"]
    else:
        kind = state["kind"]
        compatible = initial_ready if kind == "initial" else dynamic_ready
        incompatible = dynamic_ready if kind == "initial" else initial_ready
        # Avoid cross-phase reordering. The normal scheduler remains the safe
        # fallback for this rare, mixed-arrival condition.
        if incompatible:
            ready_set.update(held)
            state.update({"ids": set(), "deadline": 0.0, "kind": None})
            return None
        held.update(compatible)

    if not held:
        return None
    if kind == "initial":
        _wait_s, target = _scheduler_microbatch_settings(scheduler)
    else:
        _fixed, _quantizers, _hop, _wait_s = _codec_dynamic_window_settings(scheduler)
        _micro_wait, target = _scheduler_microbatch_settings(scheduler)

    now = time.monotonic()
    if len(held) >= target or now >= float(state["deadline"]):
        ready_set.update(held)
        scheduler._easymagpie_codec_released_cohort_ids = set(held)
        state.update({"ids": set(), "deadline": 0.0, "kind": None})
        return kind

    # These requests stay in WAITING_FOR_CHUNK. EngineCore keeps stepping and
    # will revisit this function without blocking the Python scheduler.
    ready_set.difference_update(held)
    return "held"


def _request_left_context(request: Any) -> int:
    info = getattr(request, "additional_information", None)
    meta = info.get("meta") if isinstance(info, dict) else None
    return max(0, _scalar_int(meta.get("left_context_size") if isinstance(meta, dict) else 0, 0))


def _is_cuda_tensor(value: Any) -> bool:
    return bool(getattr(value, "is_cuda", False))


def _flat_length(value: Any) -> int:
    return int(value.numel()) if _is_cuda_tensor(value) else len(value)


def _request_codec_ids(request: Any) -> tuple[Any, bool]:
    """Return codec ids and whether Code2Wav reads them from payload metadata."""
    info = getattr(request, "additional_information", None)
    codes = info.get("codes") if isinstance(info, dict) else None
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if _is_cuda_tensor(audio) and audio.numel() > 0:
        # Native codec packets arrive time-major [F, Q].  The merge helper
        # expects codebook-major storage, which is also the native codec's
        # documented flattened input representation.
        if audio.ndim == 2:
            return audio.detach().transpose(0, 1).reshape(-1).contiguous(), True
        return audio.detach().reshape(-1).contiguous(), True
    payload_ids = _as_flat_ints(audio)
    if payload_ids:
        return payload_ids, True
    return _as_flat_ints(getattr(request, "prompt_token_ids", None)), False


def _request_codec_end_frame(request: Any) -> int | None:
    """Return the producer's absolute acoustic-frame end, when available.

    EasyMagpie adds ``generated_len`` to every async codec payload.  The
    Stage-1 worker later repurposes that field for its own runtime state, but
    the scheduler-side request retains the connector value.  Older payloads
    have no such marker and continue through the left-context-only fallback.
    """
    info = getattr(request, "additional_information", None)
    value = info.get("generated_len") if isinstance(info, dict) else None
    end = _scalar_int(value, 0)
    return end if end > 0 else None


def _set_merged_codec_request(
    request: Any,
    ids: Any,
    left_context: int,
    *,
    use_payload: bool,
) -> None:
    """Install a merged flat payload while retaining the first window's trim."""
    info = getattr(request, "additional_information", None)
    updated = dict(info) if isinstance(info, dict) else {}
    if use_payload:
        # Tensor connector payloads use one scheduler placeholder token while
        # Code2Wav obtains the real codebook-major stream from ``codes.audio``.
        # Retain that contract rather than turning a B32 codec request into
        # thousands of scheduler tokens.
        request.prompt_token_ids = [0]
        codes = updated.get("codes")
        codes = dict(codes) if isinstance(codes, dict) else {}
        codes["audio"] = ids
        updated["codes"] = codes
    else:
        # List payloads use prompt ids directly.  Remove an old tensor payload
        # so Code2Wav cannot prefer stale pre-merge ids over this new stream.
        request.prompt_token_ids = ids
        updated.pop("codes", None)
    meta = updated.get("meta")
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta["left_context_size"] = int(left_context)
    updated["meta"] = meta
    request.additional_information = updated


def _drain_one_codec_buffer(adapter: Any, request: Any, *, fixed: int, quantizers: int, hop: int) -> int:
    """Drain already-published successor windows for one Stage-1 request.

    The connector delivers ordered, overlapping source windows.  It is safe to
    poll synchronously here because a request is removed from the adapter's
    pending receive deque as soon as its current window becomes ready.  We only
    poll while the maximum source hop fits, so no consumed payload has to be put
    back into shared memory.
    """
    current, current_uses_payload = _request_codec_ids(request)
    if _flat_length(current) == 0 or _flat_length(current) % quantizers:
        return 0
    current_end = _request_codec_end_frame(request)
    left_context = _request_left_context(request)
    merged_count = 0

    while _flat_length(current) // quantizers + hop <= fixed:
        is_done = getattr(adapter, "is_done_receiving_chunks", None)
        if callable(is_done) and is_done(request.request_id):
            break

        previous_ids = current
        previous_prompt_ids = getattr(request, "prompt_token_ids", None)
        previous_info = getattr(request, "additional_information", None)
        # Avoid a stale payload taking precedence over the successor's prompt
        # ids when a connector changes between tensor and list serialization.
        if isinstance(previous_info, dict) and "codes" in previous_info:
            sanitized = dict(previous_info)
            sanitized.pop("codes", None)
            request.additional_information = sanitized

        if not adapter._poll_single_request(request):
            request.prompt_token_ids = previous_prompt_ids
            request.additional_information = previous_info
            break

        following, following_uses_payload = _request_codec_ids(request)
        if _flat_length(following) == 0:
            # A terminal marker may carry no codes.  Preserve its finished state
            # in the adapter, but feed the real buffered audio in this scheduler
            # pass before the normal completion path runs.
            request.prompt_token_ids = previous_prompt_ids
            request.additional_information = previous_info
            break

        following_end = _request_codec_end_frame(request)
        overlap = _request_left_context(request)
        if current_end is not None and following_end is not None:
            current_frames = _flat_length(previous_ids) // quantizers
            following_frames = _flat_length(following) // quantizers
            current_start = current_end - current_frames
            following_start = following_end - following_frames
            if following_end <= current_end:
                # The scheduler can observe a payload that has already been
                # incorporated into a prior dynamic merge.  It is safe to
                # consume only if it is an exact duplicate of a slice of the
                # buffered source stream; otherwise preserve the current window
                # and let the normal connector path handle it.
                offset = following_start - current_start
                contained = 0 <= offset and offset + following_frames <= current_frames
                if _is_cuda_tensor(previous_ids) or _is_cuda_tensor(following):
                    # Connector order makes a contained CUDA payload safe.
                    # Avoid a device-wide synchronization merely to compare a
                    # stale duplicate before releasing it.
                    same_codes = contained
                else:
                    same_codes = contained and all(
                        previous_ids[q * current_frames + offset : q * current_frames + offset + following_frames]
                        == following[q * following_frames : (q + 1) * following_frames]
                        for q in range(quantizers)
                    )
                if same_codes:
                    request.prompt_token_ids = previous_prompt_ids
                    request.additional_information = previous_info
                    continue
                request.prompt_token_ids = previous_prompt_ids
                request.additional_information = previous_info
                logger.warning(
                    "EasyMagpie codec buffer drain found a non-identical stale window: "
                    "request=%s current=[%d,%d) following=[%d,%d)",
                    request.request_id,
                    current_start,
                    current_end,
                    following_start,
                    following_end,
                )
                break
            overlap = current_end - following_start
            if overlap < 0 or overlap > current_frames or overlap > following_frames:
                request.prompt_token_ids = previous_prompt_ids
                request.additional_information = previous_info
                logger.warning(
                    "EasyMagpie codec buffer drain found a gap: request=%s "
                    "current=[%d,%d) following=[%d,%d)",
                    request.request_id,
                    current_start,
                    current_end,
                    following_start,
                    following_end,
                )
                break

        try:
            merged = merge_codec_windows(
                previous_ids,
                following,
                num_quantizers=quantizers,
                next_left_context_frames=overlap,
                max_frames=fixed,
            )
        except ValueError:
            # The connector contract guarantees ordered overlap.  If a custom
            # connector violates it, keep the current valid audio instead of
            # corrupting the stream with an unsafe concatenation.
            current_frames = _flat_length(previous_ids) // quantizers
            following_frames = _flat_length(following) // quantizers
            next_left = _request_left_context(request)
            if _is_cuda_tensor(previous_ids) or _is_cuda_tensor(following):
                current_overlap = following_overlap = "device-resident (comparison skipped)"
            else:
                current_overlap = [
                    previous_ids[codebook * current_frames + current_frames - next_left]
                    for codebook in range(quantizers)
                ] if next_left else []
                following_overlap = [
                    following[codebook * following_frames]
                    for codebook in range(quantizers)
                ] if next_left else []
            logger.warning(
                "EasyMagpie codec buffer drain rejected a non-contiguous window: "
                "request=%s current_frames=%d following_frames=%d left=%d "
                "boundary_current=%s boundary_next=%s",
                request.request_id,
                current_frames,
                following_frames,
                next_left,
                current_overlap,
                following_overlap,
                exc_info=True,
            )
            request.prompt_token_ids = previous_prompt_ids
            request.additional_information = previous_info
            break
        if merged is None:
            # Preflight with ``hop`` above makes this unreachable for the normal
            # processor (whose final window is no larger than one hop).  Preserve
            # the current payload defensively rather than truncating audio.
            request.prompt_token_ids = previous_prompt_ids
            request.additional_information = previous_info
            break

        current = merged
        current_end = following_end if following_end is not None else current_end
        current_uses_payload = current_uses_payload or following_uses_payload
        _set_merged_codec_request(request, current, left_context, use_payload=current_uses_payload)
        merged_count += 1

    return merged_count


@nvtx_profiled("EM.stage1.scheduler.dynamic_drain")
def _drain_dynamic_codec_buffers(scheduler: Any) -> int:
    """Merge post-first codec chunks that are buffered when Stage 1 schedules.

    The first model frame is deliberately exempt so first audio latency remains
    unchanged.  Each later stream gets one bounded coalescing interval, then
    every contiguous window already in its connector buffer is sent to Code2Wav
    as one dynamic-sized decode (capped by the exported fixed frame capacity).
    """
    if not _is_easymagpie_codec_scheduler(scheduler) or not _codec_dynamic_enabled(scheduler):
        return 0
    adapter = getattr(scheduler, "chunk_transfer_adapter", None)
    if adapter is None:
        return 0
    ready = set(getattr(adapter, "_finished_load_reqs", ()) or ())
    if not ready:
        return 0

    initial_ids = getattr(scheduler, "_easymagpie_codec_initial_ids", None)
    if initial_ids is None:
        initial_ids = set()
        scheduler._easymagpie_codec_initial_ids = initial_ids
    tracked = set(getattr(scheduler, "requests", {}) or {})
    initial_ids.intersection_update(tracked)
    first_ready = ready - initial_ids
    initial_ids.update(ready)
    if first_ready:
        # Do not delay the first ~80 ms audio packet for any stream.
        return 0

    fixed, quantizers, hop, _wait_s = _codec_dynamic_window_settings(scheduler)
    if fixed <= 0 or quantizers <= 0 or hop <= 0:
        return 0

    drained = 0
    # The original ready set remains valid; the receive thread only appends a
    # request after it obtains the next payload, and the synchronous polling
    # below consumes those successors in connector order.
    for request_id in ready:
        request = getattr(scheduler, "requests", {}).get(request_id)
        if request is None:
            continue
        with nvtx_range("EM.stage1.scheduler.drain_one_request"):
            drained += _drain_one_codec_buffer(
                adapter,
                request,
                fixed=fixed,
                quantizers=quantizers,
                hop=hop,
            )
    return drained


def _is_easymagpie_codec_scheduler(scheduler: Any) -> bool:
    """Limit the generic scheduler patch to EasyMagpie's Stage-1 codec."""
    vllm_config = getattr(scheduler, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    stage_id = int(getattr(model_config, "stage_id", -1) or -1)
    model_arch = getattr(model_config, "model_arch", "")
    model_stage = getattr(model_config, "model_stage", "")
    return stage_id > 0 and (
        model_arch in {"EasyMagpieCodecForConditionalGeneration", "EasyMagpieCode2Wav"}
        or model_stage in {"easymagpie_codec", "easymagpie_code2wav"}
    )


@nvtx_profiled("EM.stage1.scheduler.cohort_wait")
def _wait_for_codec_cohort(scheduler: Any) -> bool:
    """Give Stage 1 a bounded chance to collect peer codec windows.

    ``OmniGenerationScheduler`` normally schedules the first request whose
    connector chunk appears immediately. That turns a simultaneous B32 AR step
    into a sequence of small codec launches. The connector's ready set is the
    right scheduling boundary: delay only after at least one real chunk has
    arrived, and never wait longer than the configured deadline.
    """
    if not _is_easymagpie_codec_scheduler(scheduler):
        return False
    adapter = getattr(scheduler, "chunk_transfer_adapter", None)
    if adapter is None:
        return False
    ready_requests = getattr(adapter, "_finished_load_reqs", ())
    if not ready_requests:
        return False
    wait_s, target_batch = _scheduler_microbatch_settings(scheduler)
    if wait_s <= 0 or len(ready_requests) >= target_batch:
        return False

    now = time.monotonic()
    deadline = float(getattr(scheduler, "_easymagpie_codec_cohort_deadline", 0.0) or 0.0)
    if deadline <= now:
        deadline = now + wait_s
        scheduler._easymagpie_codec_cohort_deadline = deadline
    remaining = deadline - now
    if remaining > 0:
        if not getattr(scheduler, "_easymagpie_codec_microbatch_logged", False):
            logger.warning(
                "EasyMagpie codec scheduler microbatching enabled: wait=%.2fms target=B%d",
                wait_s * 1000.0,
                target_batch,
            )
            scheduler._easymagpie_codec_microbatch_logged = True
        # Cap the sleep explicitly: float round-off in ``deadline - now``
        # must not extend the configured latency budget.
        with nvtx_range("EM.stage1.scheduler.cohort_sleep"):
            time.sleep(min(remaining, wait_s))
    scheduler._easymagpie_codec_cohort_deadline = 0.0
    return True


def _send_one_stream(adapter: Any, tasks: list[Any]) -> None:
    """Send a stream's queued chunks serially, retaining its connector order."""
    for task in tasks:
        try:
            adapter._send_single_request(task)
        except Exception:  # noqa: BLE001 - mirrors vLLM-Omni's original save loop.
            logger.warning("Error saving EasyMagpie codec chunk", exc_info=True)


def install_transfer_microbatch_patch() -> bool:
    """Patch vLLM-Omni's Stage-0 save loop when the runtime is available.

    The generic adapter wakes on the first request in an AR scheduler step and
    serially writes every request's codec chunk.  This patch gives the cohort a
    configurable short collection window, then sends different streams in
    parallel.  Requests from the same stream remain serialized so connector
    chunk indices cannot be reordered.
    """
    try:
        from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
    except Exception:
        return False

    if getattr(OmniTransferAdapterBase, "_easymagpie_codec_microbatch_patched", False):
        return True

    original_save_loop = OmniTransferAdapterBase.save_loop
    original_shutdown = OmniTransferAdapterBase.shutdown

    def save_loop(self) -> None:
        connector = getattr(self, "connector", None)
        if connector is None or int(getattr(connector, "stage_id", -1)) != 0:
            original_save_loop(self)
            return

        wait_s, parallelism = _transfer_microbatch_settings(self)
        if wait_s <= 0:
            original_save_loop(self)
            return

        if not getattr(self, "_easymagpie_codec_microbatch_logged", False):
            logger.warning(
                "EasyMagpie codec transfer microbatching enabled: wait=%.2fms parallelism=%d",
                wait_s * 1000.0,
                parallelism,
            )
            self._easymagpie_codec_microbatch_logged = True

        pool = getattr(self, "_easymagpie_codec_transfer_pool", None)
        if pool is None:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=parallelism,
                thread_name_prefix="easymagpie_codec_xfer",
            )
            self._easymagpie_codec_transfer_pool = pool

        while not self.stop_event.is_set():
            with self._save_cond:
                if not self._pending_save_reqs:
                    self._save_cond.wait(timeout=0.1)
                if not self._pending_save_reqs:
                    continue

            # Do not let a notify from the next request shorten the cohort
            # collection period. The deadline is deliberately tiny (1-2 ms).
            deadline = time.monotonic() + wait_s
            while not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                with self._save_cond:
                    self._save_cond.wait(timeout=remaining)

            pending: list[Any] = []
            while self._pending_save_reqs:
                pending.append(self._pending_save_reqs.popleft())
            if not pending:
                continue

            groups = group_pending_tasks(pending)
            futures = [pool.submit(_send_one_stream, self, tasks) for tasks in groups.values()]
            for future in futures:
                try:
                    future.result()
                except Exception:  # _send_one_stream handles individual task failures.
                    logger.warning("EasyMagpie codec transfer worker failed", exc_info=True)

    def shutdown(self) -> None:
        pool = getattr(self, "_easymagpie_codec_transfer_pool", None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        original_shutdown(self)

    OmniTransferAdapterBase.save_loop = save_loop
    OmniTransferAdapterBase.shutdown = shutdown
    OmniTransferAdapterBase._easymagpie_codec_microbatch_patched = True
    return True


def install_scheduler_microbatch_patch() -> bool:
    """Patch the Stage-1 generation scheduler with bounded codec cohorting.

    The transfer adapter can make all Stage-0 messages available promptly, but
    the stock generation scheduler still launches the first ready request before
    its peers arrive. Cohorting at this scheduling boundary is what turns those
    messages into one actual batched Code2Wav forward pass.
    """
    try:
        from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
    except Exception:
        return False

    if getattr(OmniGenerationScheduler, "_easymagpie_codec_microbatch_patched", False):
        return True

    original_schedule = OmniGenerationScheduler.schedule

    def schedule(self, *args, **kwargs):
        cohort_state = _prepare_codec_cohort_for_schedule(self)
        if cohort_state == "dynamic":
            _drain_dynamic_codec_buffers(self)
        result = original_schedule(self, *args, **kwargs)
        # Mark the initial packet only after the normal scheduler has admitted
        # it.  This prevents the dynamic drain from accidentally merging it
        # with successors that Stage 0 has already placed in shared memory.
        if cohort_state == "initial":
            initial_ids = getattr(self, "_easymagpie_codec_initial_ids", None)
            if isinstance(initial_ids, set):
                released = set(getattr(self, "_easymagpie_codec_released_cohort_ids", set()) or ())
                self._easymagpie_codec_released_cohort_ids = set()
                scheduled = set()
                for req_data in getattr(result, "scheduled_new_reqs", ()) or ():
                    req_id = getattr(req_data, "req_id", None)
                    if req_id:
                        scheduled.add(req_id)
                cached = getattr(result, "scheduled_cached_reqs", None)
                scheduled.update(getattr(cached, "req_ids", ()) or ())
                initial_ids.update(released.intersection(scheduled))
        return result

    OmniGenerationScheduler.schedule = schedule
    OmniGenerationScheduler._easymagpie_codec_microbatch_patched = True
    return True
