# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from collections import Counter
from copy import copy
from types import MethodType

import torch
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Preserve per-chunk limits, metadata and exact async resume accounting.

    vLLM-Omni 0.26 handles segment stops, but still discards only one outstanding
    token on resume. Keep that correction and EasyMagpie session metadata here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_admission_enabled = os.environ.get("EASYMAGPIE_PHASE_ADMISSION", "0") == "1"
        self.phase_admission_phase = "mixed"
        self.phase_admission_force_decode = False
        self.phase_admission_throttled = False
        self.phase_admission_counts = Counter()
        if self.phase_admission_enabled and getattr(self, "_easymagpie_phase_hooks_version", None) != 1:
            raise RuntimeError("phase admission requires the version-1 scheduler eligibility hooks")
        self.admission_decode_fence = None
        if os.environ.get("EASYMAGPIE_ADMISSION_DECODE_FENCE", "0") == "1":
            from .admission_fence import AdmissionDecodeFence

            self.admission_decode_fence = AdmissionDecodeFence(self)
        if self.vllm_config.model_config.stage_id == 0 and self.chunk_transfer_adapter is not None:
            self.chunk_transfer_adapter._send_single_request = MethodType(
                _send_talker_chunk, self.chunk_transfer_adapter
            )
            from .bulk_sender import install_bulk_sender

            install_bulk_sender(self)
        self.empty_prefix_planner = None
        self.q1_no_growth_allocator = None
        self.deadline_decode_policy = None
        if self.vllm_config.model_config.stage_id == 0:
            if _enabled("EASYMAGPIE_EMPTY_PREFIX_PLANNER"):
                from .empty_prefix_planner import Planner

                self.empty_prefix_planner = Planner(
                    self.kv_cache_manager, shadow=_enabled("EASYMAGPIE_EMPTY_PREFIX_PLANNER_SHADOW")
                )
            if _enabled("EASYMAGPIE_Q1_NO_GROWTH_ALLOCATOR"):
                from .q1_no_growth_allocator import NoGrowthAllocator

                self.q1_no_growth_allocator = NoGrowthAllocator(self.kv_cache_manager)
            if _enabled("EASYMAGPIE_DEADLINE_DECODE_MICROBATCH"):
                from .deadline_decode_microbatch import create_policy

                self.deadline_decode_policy = create_policy(self)

    def schedule(self, throttle_prefills: bool = False):
        policy = getattr(self, "deadline_decode_policy", None)
        if policy is None:
            return self._schedule_admission(throttle_prefills)
        policy.clear_plan()
        try:
            output = self._schedule_admission(throttle_prefills)
            policy.observe(self, output)
            return output
        except BaseException:
            policy.force_full = True
            policy.counts["schedule_errors"] += 1
            raise
        finally:
            policy.clear_plan()

    def _schedule_admission(self, throttle_prefills: bool = False):
        if not getattr(self, "phase_admission_enabled", False):
            return super().schedule(throttle_prefills)
        self.phase_admission_throttled = throttle_prefills
        fence = getattr(self, "admission_decode_fence", None)
        try:
            output = super().schedule(throttle_prefills)
        except BaseException:
            if fence is not None:
                fence.clear("schedule_errors")
            raise
        phase = self.phase_admission_phase
        self.phase_admission_counts[f"{phase}_steps"] += 1
        self.phase_admission_counts[f"{phase}_requests"] += len(output.num_scheduled_tokens)
        self.phase_admission_counts[f"{phase}_tokens"] += output.total_num_scheduled_tokens
        if not output.num_scheduled_tokens:
            self.phase_admission_counts[f"{phase}_empty_steps"] += 1
        # An admission attempt cannot indefinitely prevent existing streams from decoding.
        self.phase_admission_force_decode = phase == "prefill"
        if fence is not None:
            fence.after_schedule(self, output)
        return output

    def _should_defer_waiting_admission(self) -> bool:
        result = self._select_admission_phase()
        policy = getattr(self, "deadline_decode_policy", None)
        if policy is not None:
            policy.choose(self, self._phase_running_eligible)
        return result

    def _select_admission_phase(self) -> bool:
        parent_defer = super()._should_defer_waiting_admission()
        if not getattr(self, "phase_admission_enabled", False):
            return parent_defer
        self.phase_admission_phase = "mixed"
        unsupported = (
            self.phase_admission_throttled
            or self.num_spec_tokens != 0
            or self.num_sampled_tokens_per_step != 1
            or self.parallel_config.pipeline_parallel_size != 1
            or self.connector is not None
            or self.ec_connector is not None
            or self.vllm_config.model_config.stage_id != 0
        )
        if unsupported:
            self.phase_admission_counts["unsupported_fallback_steps"] += 1
            fence = getattr(self, "admission_decode_fence", None)
            if fence is not None:
                fence.clear("released_unsupported")
            return parent_defer

        def has_work(request):
            return (
                request.status == RequestStatus.RUNNING
                and self.current_step + 1 >= request.next_decode_eligible_step
                and request.num_tokens_with_spec + request.num_output_placeholders > request.num_computed_tokens
                and not (
                    request.num_output_placeholders > 0
                    and request.num_computed_tokens + 2 - request.num_output_placeholders
                    >= request.num_prompt_tokens + request.max_tokens
                )
            )

        running_prefill = False
        running_decode = False
        for request in self.running:
            if has_work(request):
                if self._phase_request_is_prefill(request):
                    running_prefill = True
                else:
                    running_decode = True
        admission_space = len(self.running) + self.num_waiting_for_streaming_input < self.max_num_running_reqs
        waiting_prefill = admission_space and bool(self.waiting or self.skipped_waiting) and not parent_defer
        fence = getattr(self, "admission_decode_fence", None)
        fence_decode = fence.select(self, has_work, waiting_prefill) if fence is not None else False
        if running_decode and (self.phase_admission_force_decode or fence_decode):
            self.phase_admission_phase = "decode"
        elif running_prefill or waiting_prefill:
            self.phase_admission_phase = "prefill"
        else:
            self.phase_admission_phase = "decode"
        return parent_defer

    def _phase_request_is_prefill(self, request: Request) -> bool:
        pending_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        return request.is_prefill_chunk or pending_tokens > 1

    def _is_running_request_eligible(self, request: Request) -> bool:
        allowed = self._phase_running_eligible(request)
        policy = getattr(self, "deadline_decode_policy", None)
        return allowed if policy is None else policy.allow(self, request, allowed)

    def _phase_running_eligible(self, request: Request) -> bool:
        phase = getattr(self, "phase_admission_phase", "mixed")
        if not getattr(self, "phase_admission_enabled", False) or phase == "mixed":
            return True
        return self._phase_request_is_prefill(request) == (phase == "prefill")

    def _can_schedule_waiting_requests(self) -> bool:
        return not getattr(self, "phase_admission_enabled", False) or self.phase_admission_phase != "decode"

    def phase_admission_status(self) -> dict:
        return {
            "enabled": self.phase_admission_enabled,
            "phase": self.phase_admission_phase,
            "force_decode_next": self.phase_admission_force_decode,
            "counts": dict(self.phase_admission_counts),
            "decode_fence": (
                self.admission_decode_fence.status()
                if getattr(self, "admission_decode_fence", None) is not None
                else {"enabled": False}
            ),
        }

    def shutdown(self):
        error = None
        try:
            return self._shutdown_admission()
        except BaseException as exc:
            error = exc
            raise
        finally:
            # Restore only our own instance delegates after the native sender
            # and scheduler have drained. No global allocator methods change.
            for name in ("deadline_decode_policy", "q1_no_growth_allocator", "empty_prefix_planner"):
                helper = getattr(self, name, None)
                if helper is not None:
                    try:
                        helper.close()
                    except BaseException as exc:
                        if error is not None:
                            error.add_note(f"{name} cleanup failed: {exc!r}")
                        else:
                            error = exc
            if error is not None:
                raise error

    def _shutdown_admission(self):
        fence = getattr(self, "admission_decode_fence", None)
        try:
            result = super().shutdown()
        except BaseException:
            if fence is not None:
                fence.clear("shutdown_error_releases")
                try:
                    fence.write("shutdown_error")
                except Exception:
                    pass  # Preserve the original shutdown failure.
            raise
        if fence is not None:
            fence.clear("shutdown_releases")
            fence.write("shutdown")
        bulk = getattr(self, "bulk_sender", None)
        policy = getattr(bulk, "deadline_policy", None)
        if policy is not None:
            policy.close()
        return result

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if (
            self.vllm_config.model_config.stage_id == 0
            and existing is not None
            and existing.resumable
            and existing.status == RequestStatus.WAITING_FOR_STREAMING_REQ
            and not request.resumable
            and not request.abort_immediately
        ):
            # Unlike cancellation, a late input sentinel must flush the codec
            # tail even though no further model output will drive save_async.
            self.finish_requests(existing.request_id, RequestStatus.FINISHED_STOPPED)
            if self.chunk_transfer_adapter is not None:
                # Pending sender tasks still reference the resumable segment.
                # Only this final callback may flush and release its codec state.
                terminal = copy(existing)
                terminal.resumable = False
                self.chunk_transfer_adapter.save_async(multimodal_output=None, request=terminal)
            return
        super().add_request(request)

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


def _send_talker_chunk(adapter: OmniChunkTransferAdapter, task: dict):
    previous = getattr(adapter, "_easymagpie_send_finish", None)
    adapter._easymagpie_send_finish = (task["is_finished"], task["is_segment_finished"])
    try:
        return OmniChunkTransferAdapter._send_single_request(adapter, task)
    finally:
        adapter._easymagpie_send_finish = previous


def _enabled(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(name + " must be 0 or 1")
    return value == "1"


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


def _set_segment_finished(outputs, request_id: str, value: bool) -> None:
    """Set the streaming-boundary bit on one scheduler output."""
    if not isinstance(outputs, dict):
        return
    for client_outputs in outputs.values():
        for output in getattr(client_outputs, "outputs", ()):
            if getattr(output, "request_id", None) == request_id:
                output.is_segment_finished = value


def _without_consumed_codec_audio(info):
    """Copy request metadata without the previous chunk's audio codes."""
    if not isinstance(info, dict):
        return info
    codes = info.get("codes")
    if not isinstance(codes, dict) or "audio" not in codes:
        return info
    return {**info, "codes": {key: value for key, value in codes.items() if key != "audio"}}


def _poll_native_codec_chunk_unlocked(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Receive a chunk without resetting the vLLM state-cache position."""
    old_num_computed_tokens = request.num_computed_tokens
    old_additional_information = request.additional_information
    # Async-chunk prewarm may install one unscheduled placeholder before the
    # first real payload. Only tokens with materialized state are retained.
    old_prompt = list(request.prompt_token_ids or [])[:old_num_computed_tokens]
    old_all_token_ids = list(request._all_token_ids)[:old_num_computed_tokens]

    # vLLM-Omni merges incoming generation payloads into this object. Remove
    # consumed audio so a control-only boundary cannot inherit and replay it.
    poll_information = _without_consumed_codec_audio(old_additional_information)
    request.additional_information = poll_information
    received = OmniChunkTransferAdapter._poll_single_request(adapter, request)
    if received:
        frames = _codec_payload_frames(request.additional_information, adapter._easymagpie_num_quantizers)
        # Empty terminal controls must wake the request for zero-token completion.
        if frames > 0 or request.request_id in adapter.upstream_exhausted_requests:
            placeholders = [0] * frames
            request.prompt_token_ids = old_prompt + placeholders
            request._all_token_ids[:] = old_all_token_ids + placeholders
            request.num_prompt_tokens = len(request.prompt_token_ids)
            request.num_computed_tokens = old_num_computed_tokens
            request.update_block_hashes()
            return True
        adapter._finished_load_reqs.discard(request.request_id)
    elif request.additional_information is poll_information:
        request.additional_information = old_additional_information

    # Dialogue boundaries must reach the output processor even when the final
    # codec payload is empty. Keep ordinary streaming controls parked, but wake
    # a reset boundary for zero-token completion without replaying prior audio.
    dialogue_boundary = request.request_id in getattr(
        adapter, "segment_finished_requests", ()
    ) and _codec_segment_resets_state(request)
    if dialogue_boundary:
        adapter._finished_load_reqs.add(request.request_id)

    request.prompt_token_ids = old_prompt
    request._all_token_ids[:] = old_all_token_ids
    request.num_prompt_tokens = len(old_prompt)
    request.num_computed_tokens = old_num_computed_tokens
    request.update_block_hashes()
    return dialogue_boundary


def _poll_native_codec_chunk(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Publish connector readiness only after the request payload is coherent."""
    with adapter._easymagpie_chunk_lock:
        return _poll_native_codec_chunk_unlocked(adapter, request)


def _codec_segment_resets_state(request: Request) -> bool:
    """Whether the completed Stage-1 segment is a dialogue utterance boundary."""
    payload = getattr(request, "additional_information", None)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    codec_streaming = meta.get("codec_streaming") if isinstance(meta, dict) else None
    if isinstance(codec_streaming, torch.Tensor):
        return codec_streaming.numel() == 1 and not bool(codec_streaming.item())
    return codec_streaming is False


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
        self._easymagpie_reset_on_update: set[str] = set()
        from .pooled_publication import install_pooled_publication

        install_pooled_publication(adapter, role="receiver")

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """Resume a text chunk, or reset Stage 1 for a new dialogue reply.

        Every incremental text update prewarms downstream stages again. vLLM
        turns the duplicate Stage-1 request into a streaming update, but the
        generation scheduler's default handler replaces its prompt and resets
        ``num_computed_tokens``. For the native codec the update carries no new
        codec input; it only signals that another upstream segment is coming.
        Keep the append-only prompt position and vLLM-managed codec state intact.
        """
        reset_on_update = getattr(self, "_easymagpie_reset_on_update", set())
        if session.request_id in reset_on_update:
            reset_on_update.discard(session.request_id)
            self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
            return super()._update_request_as_session(session, update)

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
        adapter = self.chunk_transfer_adapter
        segment_boundary = request.request_id in adapter.segment_finished_requests
        reset_codec = _codec_segment_resets_state(request)
        if reset_codec:
            # Arm the reset before the base handler consumes a queued update;
            # it calls our _update_request_as_session override synchronously.
            self._easymagpie_reset_on_update.add(request.request_id)

        finished = super()._handle_stopped_request(request)
        if finished:
            self._easymagpie_reset_on_update.discard(request.request_id)
        else:
            segment_requests = getattr(self, "_easymagpie_segment_requests", None)
            if segment_requests is not None:
                segment_requests.add(request.request_id)
            if not reset_codec:
                stopped_sessions = getattr(self, "_easymagpie_stopped_sessions", None)
                if stopped_sessions is not None:
                    stopped_sessions.append(request)

        if segment_boundary:
            # TODO(vLLM-Omni upstream): metadata-only boundaries become ready
            # without scheduling tokens, so the adapter never consumes these
            # readiness entries. Clear them exactly when Stage 1 consumes the
            # boundary; otherwise the next turn schedules its prewarm token
            # against the previous codec payload.
            adapter.segment_finished_requests.discard(request.request_id)
            adapter._finished_load_reqs.discard(request.request_id)
            adapter.requests_with_ready_chunks.discard(request.request_id)
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
        self._easymagpie_segment_requests = set()
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request_id in self._easymagpie_segment_requests:
                _set_segment_finished(outputs, request_id, True)
            for session in self._easymagpie_stopped_sessions:
                self._resume_codec_after_segment(session)
        finally:
            self._easymagpie_stopped_sessions = None
            self._easymagpie_segment_requests = None
        return outputs

    def schedule(self, *args, **kwargs):
        with self.chunk_transfer_adapter._easymagpie_chunk_lock:
            return super().schedule(*args, **kwargs)


def _poll_fenced_codec_chunk(adapter, request):
    """Do not materialize later chunk or terminal metadata during a live step."""
    with adapter._easymagpie_chunk_lock:
        if request.num_in_flight_tokens > 0:
            return False
        return _poll_native_codec_chunk_unlocked(adapter, request)


def _process_fenced_codec_chunks(adapter, waiting_queue, running_queue, scheduler_requests=None):
    adapter._codec_async_process_pending_chunks(waiting_queue, running_queue, scheduler_requests=scheduler_requests)
    # The regular adapter parks requests still awaiting payloads. A request with
    # a final chunk already materialized is otherwise kept visible and could be
    # zero-token-finished by another schedule() before its PCM is delivered.
    # Reuse its existing parking deque: it counts toward retained-slot capacity
    # and restore_queues() restores it after this scheduling decision.
    parked = adapter.waiting_for_chunk_running_requests
    for request in list(running_queue):
        if request.num_in_flight_tokens > 0:
            running_queue.remove(request)
            parked.append(request)


class EasyMagpieAsyncCodecScheduler(EasyMagpieCodecScheduler):
    """Opt-in overlap between different requests with one step per live stream.

    A request's payload, terminal flags and state pages stay stable until its
    preceding output is settled. Other streams can run while PCM copies finish.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = self.chunk_transfer_adapter
        if adapter._active_window:
            raise ValueError("the async native codec fence currently requires active_stream_window=0")
        adapter._codec_async_process_pending_chunks = adapter.process_pending_chunks
        adapter.process_pending_chunks = MethodType(_process_fenced_codec_chunks, adapter)
        adapter._poll_single_request = MethodType(_poll_fenced_codec_chunk, adapter)

    def update_from_output(self, scheduler_output, model_runner_output):
        # Base update decrements the counter before deciding whether to finish.
        # Keep polling blocked until that entire decision, including segment
        # resume bookkeeping, has finished.
        with self.chunk_transfer_adapter._easymagpie_chunk_lock:
            return super().update_from_output(scheduler_output, model_runner_output)
