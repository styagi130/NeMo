# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Group existing codec work by frame count without changing request order.

Each request retains its vLLM-owned cache page and initial-state flag. Grouping
only changes the matrix shape used for this already-scheduled codec invocation;
no frames are padded, delayed, discarded or transferred between requests.
"""
from __future__ import annotations

import copy
import os
from collections import Counter

import torch
from easymagpie_vllm_omni.codec.packed import CodecStateLayer
from vllm.forward_context import get_forward_context, set_forward_context


def uniform_groups_enabled():
    return os.environ.get("EASYMAGPIE_VLLM_CODEC_UNIFORM_GROUPS", "0").lower() in ("1", "true", "yes", "on")


class CodecUniformGroups:
    def __init__(self, codec, vllm_config, graphs=None):
        self.codec = codec
        self.vllm_config = vllm_config
        self.graphs = graphs
        self.layers = tuple(module for module in codec.modules() if isinstance(module, CodecStateLayer))
        self.invocations = 0
        self.group_calls = 0
        self.frames_histogram = Counter()
        self.fallbacks = Counter()

    def status(self):
        return {
            "enabled": True,
            "invocations": self.invocations,
            "group_calls": self.group_calls,
            "frames_histogram": dict(self.frames_histogram),
            "fallbacks_by_reason": dict(self.fallbacks),
            "preserves_request_order": True,
            "pads_frames": False,
        }

    def _decode(self, codes):
        return self.codec(codes) if self.graphs is None else self.graphs.run(codes)

    @staticmethod
    def _compatible(metadata, frame_counts):
        if metadata is None or metadata.num_decode_tokens != metadata.num_decodes:
            return False
        if metadata.num_reqs != len(frame_counts) or metadata.num_decodes + metadata.num_prefills != len(frame_counts):
            return False
        if any(frames != 1 for frames in frame_counts[: metadata.num_decodes]):
            return False
        return sum(frame_counts[metadata.num_decodes :]) == metadata.num_prefill_tokens

    @staticmethod
    def _group_metadata(metadata, members, frames, device):
        parts, initials = [], []
        if metadata.num_decodes:
            indices = metadata.state_indices_tensor_d
            if indices.ndim == 2:
                indices = indices[:, 0]
            parts.append(indices)
            initials.append(torch.ones(metadata.num_decodes, dtype=torch.bool, device=device))
        if metadata.num_prefills:
            parts.append(metadata.state_indices_tensor_p)
            initials.append(metadata.has_initial_states_p)
        select = torch.tensor(members, dtype=torch.long, device=device)
        routing = parts[0] if len(parts) == 1 else torch.cat(parts)
        has_initial = initials[0] if len(initials) == 1 else torch.cat(initials)
        grouped = copy.copy(metadata)
        grouped.num_prefills = grouped.num_reqs = len(members)
        grouped.num_prefill_tokens = len(members) * frames
        grouped.num_decodes = grouped.num_decode_tokens = 0
        grouped.state_indices_tensor_p = routing.index_select(0, select)
        grouped.has_initial_states_p = has_initial.index_select(0, select)
        grouped.query_start_loc_p = torch.arange(len(members) + 1, dtype=torch.int32, device=device) * frames
        grouped.state_indices_tensor_d = None
        grouped.query_start_loc_d = None
        grouped.codec_uniform = grouped.codec_prefill_uniform = True
        grouped.codec_max_query_len = frames
        return grouped

    def run(self, codes, frame_counts):
        raw = get_forward_context().attn_metadata
        if not codes.is_cuda or not isinstance(raw, dict) or not frame_counts or any(n <= 0 for n in frame_counts):
            self.fallbacks["profile_cpu_or_empty"] += 1
            return self._decode(codes)
        if torch.cuda.is_current_stream_capturing():
            self.fallbacks["outer_capture"] += 1
            return self._decode(codes)
        if sum(frame_counts) != codes.shape[0] or any(
            not self._compatible(raw.get(layer.prefix), frame_counts) for layer in self.layers
        ):
            self.fallbacks["metadata_mismatch"] += 1
            return self._decode(codes)
        groups = {}
        offsets = [0]
        for index, frames in enumerate(frame_counts):
            groups.setdefault(frames, []).append(index)
            offsets.append(offsets[-1] + frames)
        self.invocations += 1
        outputs = [None] * len(frame_counts)
        for frames, members in groups.items():
            # Preserve the zero-copy fast path for an already-uniform prefill.
            existing = raw[self.layers[0].prefix]
            if len(groups) == 1 and existing.num_decodes == 0 and getattr(existing, "codec_uniform", False):
                self.group_calls += 1
                self.frames_histogram[frames] += 1
                return self._decode(codes)
            grouped_codes = torch.cat([codes[offsets[i] : offsets[i + 1]] for i in members], dim=0)
            replacements = {}
            grouped_raw = {}
            for layer in self.layers:
                original = raw[layer.prefix]
                identity = id(original)
                if identity not in replacements:
                    replacements[identity] = self._group_metadata(original, members, frames, codes.device)
                grouped_raw[layer.prefix] = replacements[identity]
            with set_forward_context(grouped_raw, self.vllm_config):
                pcm = self._decode(grouped_codes)
            samples = frames * self.codec.config.samples_per_frame
            for row, member in enumerate(members):
                outputs[member] = pcm[row * samples : (row + 1) * samples]
            self.group_calls += 1
            self.frames_histogram[frames] += 1
        return torch.cat(outputs)
