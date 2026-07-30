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
"""Streaming scheduler that propagates EasyMagpie request metadata.

Configure it on a single-stage deployment with::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

import os
import threading
import time
from types import MethodType

import torch
from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter

logger = init_logger(__name__)


def _codec_cohort_target(extra: dict, max_num_running_reqs: int) -> int:
    """Return the steady Stage-1 cohort target, with an experiment override."""
    raw_target = os.getenv(
        "EASYMAGPIE_CODEC_COHORT_MAX_BATCH_SIZE",
        str(extra.get("codec_microbatch_max_batch_size", max_num_running_reqs)),
    )
    try:
        target = int(raw_target)
    except (TypeError, ValueError):
        target = max_num_running_reqs
    return max(1, min(target, max_num_running_reqs))


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Forward each chunk's token limit and additional information.

    This class also works around a bug in vLLM-Omni's async segment-stop
    handling that deadlocks paced streaming sessions. On a resumable segment
    stop, ``OmniARScheduler.update_from_output`` does::

        request.async_tokens_to_discard = 1        # hardcoded
        request.num_output_placeholders = 0

    i.e. it assumes exactly one async token is in flight and, unlike omni's own
    *resume* path, it never rolls ``num_computed_tokens`` back for the tokens it
    is about to discard. Combined with vLLM 0.24's async accounting (a discarded
    token returns early from ``AsyncScheduler._update_request_with_output``
    without decrementing ``num_output_placeholders``) this leaves the re-admitted
    session in an unschedulable state:

    * a leaked placeholder (``placeholders>0`` with ``num_computed==num_tokens``)
      permanently trips the scheduler's async skip-optimisation, or
    * ``num_computed_tokens == num_tokens`` with ``placeholders==0`` yields
      ``num_new_tokens==0``.

    Either way the request is never scheduled again and paced clients hang.

    The fix mirrors omni's resume path: snapshot the *true* number of in-flight
    async tokens at the moment of the stop, then after ``update_from_output`` set
    ``async_tokens_to_discard`` to that count (0 when nothing is in flight, so no
    spurious discard) and roll ``num_computed_tokens`` back by the same amount.

    TODO(upstream): fix ``OmniARScheduler.update_from_output`` directly so the
    segment-stop branch uses ``async_tokens_to_discard = num_output_placeholders``
    and ``num_computed_tokens -= num_output_placeholders`` (matching the resume
    branch), then drop this override.
    """

    def _update_request_with_output(self, request: Request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)
        if stopped:
            # After super() has decremented the placeholder for the stopping
            # token, ``num_output_placeholders`` is the number of *other* async
            # tokens still in flight for this request — the value omni's stop
            # handler should have used but overwrites with a hardcoded 1. Record
            # it so update_from_output can restore the correct accounting. Only
            # tracked while inside update_from_output, so there is no per-step
            # cost beyond the (rare) segment stops themselves.
            pending = getattr(self, "_emp_stopped_this_step", None)
            if pending is not None:
                pending.append((request, request.num_output_placeholders))
        return new_token_ids, stopped

    def update_from_output(self, scheduler_output, model_runner_output):
        self._emp_stopped_this_step = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request, snap in self._emp_stopped_this_step:
                # Only correct resumable stops where omni actually armed a discard.
                if getattr(request, "async_tokens_to_discard", 0) > 0:
                    request.async_tokens_to_discard = snap
                    if snap > 0:
                        request.num_computed_tokens -= snap
        finally:
            self._emp_stopped_this_step = None
        return outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        # The input engine queues ``None`` after the final StreamingInput but
        # leaves the existing session's ``resumable`` flag set. Clear it before
        # the base handler consumes the sentinel so the chunk-transfer adapter
        # emits a true terminal payload and releases request-persistent codec
        # state. An empty queue still means "waiting for more websocket input".
        streaming_queue = getattr(request, "streaming_queue", None)
        if getattr(request, "resumable", False) and streaming_queue and streaming_queue[0] is None:
            request.resumable = False
        return super()._handle_stopped_request(request)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        super()._update_request_as_session(session, update)

        # Upstream hardcodes one discard on resume even when multiple async
        # outputs are outstanding. Its rollback is otherwise correct, so retain
        # it and replace only the discard count with the captured real value.
        if outstanding_async_tokens > 0 and getattr(session, "async_tokens_to_discard", 0) > 0:
            session.async_tokens_to_discard = outstanding_async_tokens

        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info

        # Defensive guard: if a resumed session has every token already computed
        # (``num_computed_tokens >= num_tokens``), the upstream scheduler computes
        # ``num_new_tokens == 0`` and trips ``assert num_new_tokens > 0``. Roll
        # back one token so there is always something to recompute and sample from
        # — the same "recompute the last token" corrective vLLM applies on a full
        # prompt cache hit (see Scheduler._update_waiting_for_remote_kv).
        if session.num_computed_tokens >= session.num_tokens:
            session.num_computed_tokens = session.num_tokens - 1


def _codec_payload_frames(info, num_quantizers: int) -> int:
    """Return the number of time-major acoustic rows in a connector payload."""
    codes = info.get("codes", {}) if isinstance(info, dict) else {}
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return 0
    if audio.ndim == 2:
        return int(audio.shape[0])
    if audio.ndim == 1 and audio.numel() % num_quantizers == 0:
        return int(audio.numel() // num_quantizers)
    raise ValueError(f"invalid native codec payload shape: {tuple(audio.shape)}")


def _poll_native_codec_chunk_unlocked(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Receive a chunk without resetting the vLLM state-cache position."""
    old_num_computed_tokens = request.num_computed_tokens
    # Async-chunk prewarm may install one unscheduled placeholder before the
    # first real payload. Only tokens with materialized state are retained.
    old_prompt = list(request.prompt_token_ids or [])[:old_num_computed_tokens]
    old_all_token_ids = list(request._all_token_ids)[:old_num_computed_tokens]

    received = OmniChunkTransferAdapter._poll_single_request(adapter, request)
    if not received:
        request.prompt_token_ids = old_prompt
        request._all_token_ids[:] = old_all_token_ids
        request.num_prompt_tokens = len(old_prompt)
        request.num_computed_tokens = old_num_computed_tokens
        request.update_block_hashes()
        return False

    frames = _codec_payload_frames(request.additional_information, adapter._easymagpie_num_quantizers)
    placeholders = [0] * frames
    request.prompt_token_ids = old_prompt + placeholders
    request._all_token_ids[:] = old_all_token_ids + placeholders
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.num_computed_tokens = old_num_computed_tokens
    request.update_block_hashes()
    return True


def _poll_native_codec_chunk(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Publish connector readiness only after the request payload is coherent."""
    with adapter._easymagpie_chunk_lock:
        return _poll_native_codec_chunk_unlocked(adapter, request)


def _native_codec_audio(request: Request, num_quantizers: int) -> torch.Tensor | None:
    """Return one time-major CUDA codec window from request metadata."""
    info = getattr(request, "additional_information", None)
    codes = info.get("codes") if isinstance(info, dict) else None
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio, torch.Tensor) or not audio.is_cuda or audio.numel() == 0:
        return None
    if audio.ndim == 1:
        if audio.numel() % num_quantizers:
            return None
        audio = audio.reshape(num_quantizers, -1).transpose(0, 1)
    if audio.ndim != 2 or int(audio.shape[1]) != num_quantizers:
        return None
    return audio.detach()


def _drain_native_codec_request_unlocked(
    adapter: OmniChunkTransferAdapter,
    request: Request,
    *,
    num_quantizers: int,
    hop_frames: int,
    max_frames: int,
) -> int:
    """Merge already-published successors without re-entering the adapter lock.

    The scheduler holds ``_easymagpie_chunk_lock`` while this runs, so the
    background receiver cannot race the same connector key.  Unlike the legacy
    generic drain, this preserves EasyMagpieCodecScheduler's append-only prompt
    and recurrent-cache position.
    """
    current = _native_codec_audio(request, num_quantizers)
    if current is None:
        return 0

    base_computed = int(request.num_computed_tokens)
    base_prompt = list(request.prompt_token_ids or [])[:base_computed]
    base_all_token_ids = list(request._all_token_ids)[:base_computed]
    current_info = request.additional_information
    merged_count = 0

    while int(current.shape[0]) + hop_frames <= max_frames:
        if adapter.is_done_receiving_chunks(request.request_id):
            break

        # Prevent a metadata-only terminal marker from inheriting the current
        # audio leaf through the adapter's incremental metadata merge.
        sanitized = dict(current_info) if isinstance(current_info, dict) else {}
        sanitized.pop("codes", None)
        request.additional_information = sanitized
        if not _poll_native_codec_chunk_unlocked(adapter, request):
            request.additional_information = current_info
            break

        following = _native_codec_audio(request, num_quantizers)
        if following is None:
            # The connector state retains the consumed finish marker. Feed the
            # accumulated real audio before the normal completion path runs.
            request.additional_information = current_info
            break
        if int(following.shape[0]) > hop_frames:
            # The producer contract caps every successor at one source hop.
            # Do not consume beyond the exported fixed codec capacity.
            request.additional_information = current_info
            break

        next_info = request.additional_information
        current = torch.cat((current, following), dim=0).contiguous()
        merged_info = dict(next_info) if isinstance(next_info, dict) else {}
        merged_codes = dict(merged_info.get("codes") or {})
        merged_codes["audio"] = current
        merged_info["codes"] = merged_codes
        current_info = merged_info
        request.additional_information = current_info
        merged_count += 1

    merged_frames = int(current.shape[0])
    placeholders = [0] * merged_frames
    request.prompt_token_ids = base_prompt + placeholders
    request._all_token_ids[:] = base_all_token_ids + placeholders
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.num_computed_tokens = base_computed
    request.update_block_hashes()
    request.additional_information = current_info
    return merged_count


class EasyMagpieCodecScheduler(OmniGenerationScheduler):
    """Keep each Stage-1 stream on one append-only native vLLM request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = self.chunk_transfer_adapter
        if adapter is None:
            raise ValueError("the native EasyMagpie codec requires async_chunk")
        config = self.vllm_config.model_config.hf_config
        num_quantizers = int(getattr(config, "num_stacked_codebooks", 0))
        if num_quantizers <= 0:
            raise ValueError("native EasyMagpie codec config has no stacked codebooks")
        adapter._easymagpie_chunk_lock = threading.Lock()
        adapter._easymagpie_num_quantizers = num_quantizers
        adapter._poll_single_request = MethodType(_poll_native_codec_chunk, adapter)
        self._easymagpie_started_codec_ids: set[str] = set()
        self._easymagpie_drain_successors = 0

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """Resume connector polling without resetting the stateful codec.

        Every incremental text update prewarms downstream stages again. vLLM
        turns the duplicate Stage-1 request into a streaming update, but the
        generation scheduler's default handler replaces its prompt and resets
        ``num_computed_tokens``. For the native codec the update carries no new
        codec input; it only signals that another upstream segment is coming.
        Keep the append-only prompt position and vLLM-managed codec state intact.
        """
        prompt_token_ids = session.prompt_token_ids
        all_token_ids = list(session._all_token_ids)
        num_prompt_tokens = session.num_prompt_tokens
        num_computed_tokens = session.num_computed_tokens
        additional_information = session.additional_information

        super()._update_request_as_session(session, update)

        session.prompt_token_ids = prompt_token_ids
        session._all_token_ids[:] = all_token_ids
        session.num_prompt_tokens = num_prompt_tokens
        session.num_computed_tokens = num_computed_tokens
        session.additional_information = additional_information
        session.update_block_hashes()

    def _handle_stopped_request(self, request: Request) -> bool:
        finished = super()._handle_stopped_request(request)
        stopped_sessions = getattr(self, "_easymagpie_stopped_sessions", None)
        if not finished and stopped_sessions is not None:
            stopped_sessions.append(request)
        return finished

    def _resume_codec_after_segment(self, session: Request) -> None:
        """Keep a resumable codec request on the worker's cached-request path."""
        waiting_for_input = session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        if session in self.waiting:
            self.waiting.remove_requests((session,))
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
        if waiting_for_input:
            self.num_waiting_for_streaming_input -= 1

        session.status = RequestStatus.RUNNING
        if session not in self.running:
            self.running.append(session)
        self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        # A segment finish must reach the output processor, but Stage 1 must not
        # be re-admitted through the generation scheduler's ``scheduled_new``
        # path afterward. That path recreates the worker batch row, losing the
        # codec's recurrent cache even when ``num_computed_tokens`` is retained.
        # Move resumable segment stops back to ``running`` after the base method
        # has emitted the finish and removed them. Their next codec frames are
        # then scheduled as cached tokens against the same state pages.
        self._easymagpie_stopped_sessions = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for session in self._easymagpie_stopped_sessions:
                self._resume_codec_after_segment(session)
        finally:
            self._easymagpie_stopped_sessions = None
        return outputs

    def schedule(self, *args, **kwargs):
        with self.chunk_transfer_adapter._easymagpie_chunk_lock:
            adapter = self.chunk_transfer_adapter
            tracked = set(self.requests)
            self._easymagpie_started_codec_ids.intersection_update(tracked)
            ready = set(adapter._finished_load_reqs).intersection(tracked)
            initial_ready = ready - self._easymagpie_started_codec_ids
            dynamic_ready = ready & self._easymagpie_started_codec_ids
            steady_token_budget: int | None = None
            target = self.max_num_running_reqs

            # Keep first audio immediate. Once every ready request is past its
            # startup packet, allow one bounded collection interval and drain
            # only contiguous audio windows already published under the next
            # connector keys.
            if dynamic_ready and not initial_ready:
                raw_config = getattr(adapter.connector, "config", {}) or {}
                extra = raw_config.get("extra", raw_config) if isinstance(raw_config, dict) else {}
                enabled = extra.get("codec_dynamic_chunking", False)
                if isinstance(enabled, str):
                    enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
                if enabled:
                    wait_us = max(0, int(extra.get("codec_dynamic_chunk_wait_us", 0) or 0))
                    target = _codec_cohort_target(extra, self.max_num_running_reqs)
                    wait_only_underfilled = extra.get("codec_dynamic_wait_only_underfilled", False)
                    if isinstance(wait_only_underfilled, str):
                        wait_only_underfilled = wait_only_underfilled.strip().lower() in {"1", "true", "yes", "on"}
                    if wait_us and (not wait_only_underfilled or len(dynamic_ready) < target):
                        time.sleep(min(wait_us, 5_000) / 1_000_000.0)

                    hop = max(1, int(extra.get("codec_chunk_frames", 1) or 1))
                    if target < self.max_num_running_reqs:
                        # Keep all first packets eligible for immediate
                        # admission, then bound only steady codec work. Every
                        # normal successor contributes ``hop`` frame tokens;
                        # reducing the per-step token budget partitions the
                        # eager FP32 codec without narrowing either stage's
                        # request capacity or Stage-0 transfer parallelism. Do
                        # not merge successor windows in this mode: a merged
                        # 96-frame request would consume an entire B16 token
                        # budget by itself and silently turn the experiment
                        # into B1.
                        steady_token_budget = target * hop
                    else:
                        capacity = max(hop, int(extra.get("codec_fixed_chunk_frames", hop) or hop))
                        drained = 0
                        for request_id in dynamic_ready:
                            request = self.requests.get(request_id)
                            if request is not None:
                                drained += _drain_native_codec_request_unlocked(
                                    adapter,
                                    request,
                                    num_quantizers=adapter._easymagpie_num_quantizers,
                                    hop_frames=hop,
                                    max_frames=capacity,
                                )
                        self._easymagpie_drain_successors += drained
                        if drained and not getattr(self, "_easymagpie_native_drain_logged", False):
                            logger.warning(
                                "EasyMagpie native codec queue drain active: merged %d successor windows in one pass.",
                                drained,
                            )
                            self._easymagpie_native_drain_logged = True

            original_token_budget = self.max_num_scheduled_tokens
            if steady_token_budget is not None:
                self.max_num_scheduled_tokens = min(original_token_budget, steady_token_budget)
            try:
                result = super().schedule(*args, **kwargs)
            finally:
                self.max_num_scheduled_tokens = original_token_budget

            if steady_token_budget is not None:
                scheduled_cached = getattr(result, "scheduled_cached_reqs", None)
                scheduled_count = len(getattr(result, "scheduled_new_reqs", ()) or ())
                scheduled_count += len(getattr(scheduled_cached, "req_ids", ()) or ())
                previous_max = int(getattr(self, "_easymagpie_codec_observed_cohort_max", 0) or 0)
                self._easymagpie_codec_observed_cohort_max = max(previous_max, scheduled_count)
                if not getattr(self, "_easymagpie_codec_cohort_cap_logged", False):
                    logger.warning(
                        "EasyMagpie steady codec cohort cap active: target=B%d token_budget=%d observed_first=B%d.",
                        target,
                        steady_token_budget,
                        scheduled_count,
                    )
                    self._easymagpie_codec_cohort_cap_logged = True
                elif scheduled_count > previous_max:
                    logger.warning(
                        "EasyMagpie steady codec observed cohort maximum increased: B%d (target=B%d).",
                        scheduled_count,
                        target,
                    )
            # ``schedule`` advances computed tokens synchronously. Mark only
            # startup requests that were actually admitted in this pass.
            for request_id in initial_ready:
                request = self.requests.get(request_id)
                if request is not None and request.num_computed_tokens > 0:
                    self._easymagpie_started_codec_ids.add(request_id)
            return result
