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
"""Bounded startup-only graph islands for cold, exact 71-token Mamba prefills.

The vLLM custom-op split and all decode paths remain intact. No request-time
capture is permitted. Installation is explicit and restricted to an empty
startup runner; the ordinary vLLM allocator continues to own all serving pages.
"""

import functools
import hashlib
import inspect
import os
import time
import weakref
from collections import Counter
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MethodType

import torch

_STAMP = object()
_STAMP_NAME = "_easymagpie_cold71_contract"
_BACKEND_SHA = "4882f993d6124c09ad09283855a278e49fd5030accc91940c8c80d80af58231b"
_MIXER_SHA = "de92464601f291447fa4a4717464a5c8ad7a62f07e254a08c7950e5b0b29d20e"
_BASE_BACKEND_SHA = "b45db062d22f1b6283c58c8ce6dd8b0e2e6ab6a2d4a331a01c96898f24de61d1"
_TOKENS = 71


def stamp_metadata(metadata, common):
    """Verify the partition using existing CPU metadata, never device reads."""
    setattr(metadata, _STAMP_NAME, None)
    count = int(metadata.num_prefills)
    if not (
        1 <= count <= 8
        and metadata.num_reqs == count
        and metadata.num_prefill_tokens == count * _TOKENS
        and metadata.num_decodes == 0
        and metadata.num_decode_tokens == 0
        and not metadata.prep_initial_states
    ):
        return metadata
    starts = common.query_start_loc_cpu
    lengths = common.seq_lens_cpu_upper_bound
    prefilling = common.is_prefilling
    if any(value is None or value.device.type != "cpu" for value in (starts, lengths, prefilling)):
        return metadata
    if (
        starts.tolist() != list(range(0, (count + 1) * _TOKENS, _TOKENS))
        or lengths[:count].tolist() != [_TOKENS] * count
        or prefilling[:count].tolist() != [True] * count
    ):
        return metadata
    setattr(metadata, _STAMP_NAME, (_STAMP, count))
    return metadata


def install_metadata_stamp():
    """One version-bound builder wrapper; allocation/accounting stays upstream."""
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

    cls = Mamba2AttentionMetadataBuilder
    if getattr(cls.build, "_cold71_stamp", False):
        if not getattr(cls.update_block_table, "_cold71_routing_stamp", False):
            raise RuntimeError("Incomplete cold71 metadata hook installation")
        return
    path = Path(inspect.getfile(cls))
    if hashlib.sha256(path.read_bytes()).hexdigest() != _BACKEND_SHA:
        raise RuntimeError("Pinned Mamba2 metadata builder changed")
    base_path = Path(inspect.getfile(BaseMambaAttentionMetadataBuilder))
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != _BASE_BACKEND_SHA:
        raise RuntimeError("Pinned Mamba routing metadata builder changed")
    original = cls.build
    original_update = cls.update_block_table

    @functools.wraps(original)
    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        result = original(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        return stamp_metadata(result, common_attn_metadata)

    @functools.wraps(original_update)
    def update_block_table(self, metadata, block_table, slot_mapping):
        result = original_update(self, metadata, block_table, slot_mapping)
        # The pinned runner reuses metadata across equal cache specs. Upstream
        # dataclasses.replace drops dynamic attributes while refreshing routing.
        # Transfer only a CPU-verified contract whose non-routing fields survive.
        setattr(result, _STAMP_NAME, None)
        stamp = getattr(metadata, _STAMP_NAME, None)
        if (
            stamp is not None
            and stamp[0] is _STAMP
            and self.vllm_config.cache_config.mamba_cache_mode == "none"
            and self.num_spec_tokens == 0
            and type(result) is type(metadata)
            and is_dataclass(metadata)
            and all(
                getattr(result, field.name) is getattr(metadata, field.name)
                for field in fields(metadata)
                if field.name not in ("state_indices_tensor_d", "state_indices_tensor_p")
            )
            and metadata.num_prefills == stamp[1]
            and metadata.num_reqs == stamp[1]
            and 1 <= stamp[1] <= 8
            and metadata.num_prefill_tokens == stamp[1] * _TOKENS
            and metadata.num_decodes == 0
            and metadata.num_decode_tokens == 0
            and not metadata.prep_initial_states
        ):
            setattr(result, _STAMP_NAME, stamp)
        return result

    build._cold71_stamp = True
    update_block_table._cold71_routing_stamp = True
    cls.build = build
    cls.update_block_table = update_block_table


def signature(value):
    """A value signature reads CPU metadata only; GPU values are never read."""
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            tuple(value.flatten().tolist()) if value.device.type == "cpu" else None,
        )
    if is_dataclass(value):
        return (type(value), tuple((field.name, signature(getattr(value, field.name))) for field in fields(value)))
    if isinstance(value, dict):
        return tuple((key, signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(signature(item) for item in value))
    return (type(value), value)


def clone_metadata(value, memo=None):
    """Keep aliases, including nums_dict pointers into the top-level buffers."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif is_dataclass(value):
        result = type(value)(
            **{field.name: clone_metadata(getattr(value, field.name), memo) for field in fields(value)}
        )
    elif isinstance(value, dict):
        result = {key: clone_metadata(item, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [clone_metadata(item, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(clone_metadata(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = result
    return result


def metadata_copy_pairs(destination, source):
    """Collect distinct tensor buffers; preserve aliases in nested metadata."""
    copies, seen = [], set()

    def collect(dst, src):
        if isinstance(dst, torch.Tensor):
            if id(dst) not in seen:
                copies.append((dst, src))
                seen.add(id(dst))
        elif is_dataclass(dst):
            for field in fields(dst):
                collect(getattr(dst, field.name), getattr(src, field.name))
        elif isinstance(dst, dict):
            for key in dst:
                collect(dst[key], src[key])
        elif isinstance(dst, (tuple, list)):
            for left, right in zip(dst, src):
                collect(left, right)

    collect(destination, source)
    return copies


def refresh_metadata(destination, source, *, routing_only=False):
    """Validate everything before mutating any captured buffer.

    The version-bound cold71 builder derives every non-routing value from the
    verified fixed partition and chunk size. Those values were constructed
    before capture. Each layer still refreshes its own physical page indices;
    cache groups are never assumed to share page numbers or storage.
    """
    if signature(destination) != signature(source):
        raise ValueError("Metadata structure/host partition changed")
    if routing_only:
        stamp = getattr(source, _STAMP_NAME, None)
        captured_stamp = getattr(destination, _STAMP_NAME, None)
        if (
            stamp is None
            or stamp[0] is not _STAMP
            or captured_stamp != stamp
            or not 1 <= stamp[1] <= 8
            or source.num_reqs != stamp[1]
            or source.num_prefills != stamp[1]
            or source.num_prefill_tokens != stamp[1] * _TOKENS
            or source.num_decodes != 0
            or source.num_decode_tokens != 0
            or source.prep_initial_states
        ):
            raise ValueError("Routing-only refresh requires matching verified cold71 metadata")
        copies = [(destination.state_indices_tensor_p, source.state_indices_tensor_p)]
    else:
        copies = metadata_copy_pairs(destination, source)
    count = 0
    for dst, src in copies:
        if dst.device.type != "cpu":
            dst.copy_(src, non_blocking=True)
            count += 1
    return count


def cache_identity(layer):
    return tuple(
        (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
        for value in layer.kv_cache
    )


def tensor_layout(value):
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))


def make_metadata(batch, chunk_size, pages, device):
    """Construct the pinned cold-prefill contract with upstream helper logic."""
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    starts_cpu = torch.arange(batch + 1, dtype=torch.int32) * _TOKENS
    chunks, seq_indices, last = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata(
        None, chunk_size, batch, torch.zeros(batch, dtype=torch.int32), starts_cpu
    )
    nums, batch_ptr, offsets = compute_causal_conv1d_metadata(starts_cpu, device=device)
    upload = lambda value: torch.tensor(value, dtype=torch.int32, device=device)
    result = Mamba2AttentionMetadata(
        num_prefills=batch,
        num_prefill_tokens=batch * _TOKENS,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=batch,
        has_initial_states_p=torch.zeros(batch, dtype=torch.bool, device=device),
        query_start_loc_p=starts_cpu.to(device),
        num_computed_tokens_p=None,
        state_indices_tensor_p=upload(pages),
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=upload([_TOKENS] * batch),
        cu_chunk_seqlen_p=upload(chunks),
        last_chunk_indices_p=upload(last),
        nums_dict=nums,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offsets,
        prep_initial_states=False,
        chunk_size=chunk_size,
        seq_idx_p=upload(seq_indices),
    )
    setattr(result, _STAMP_NAME, (_STAMP, batch))
    return result


class LayerIslands:
    def __init__(self, layer, vllm_config):
        self.layer = layer
        self.original = layer.conv_ssm_forward
        self.config = vllm_config
        self.entries = {}
        self.replays = 0
        self.fallbacks = Counter()
        self.frozen = False
        self.cache = cache_identity(layer)
        self.stream = torch.cuda.current_stream().cuda_stream
        self.inference_mode = torch.is_inference_mode_enabled()
        self.precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self.static_metadata = os.environ.get("EASYMAGPIE_MAMBA_STATIC_PREFILL_METADATA") == "1"
        self.metadata_refresh_calls = 0
        self.metadata_copy_ops = 0
        self.metadata_constant_copies_skipped = 0

    def prewarm(self, batches):
        from vllm.forward_context import set_forward_context

        if self.frozen:
            raise RuntimeError("Capture set already frozen")
        layer = self.layer
        device = layer.kv_cache[0].device
        width = layer.tped_intermediate_size + layer.tped_conv_size + layer.tped_dt_size
        dtype = layer.conv_weights.dtype
        capture_stream = torch.cuda.Stream(device=device)
        for batch in batches:
            metadata = make_metadata(
                batch, layer.model_config.get_mamba_chunk_size(), list(range(1, batch + 1)), device
            )
            projected = torch.zeros((batch * _TOKENS, width), dtype=dtype, device=device)
            output = torch.empty((batch * _TOKENS, layer.tped_intermediate_size), dtype=dtype, device=device)
            torch.cuda.synchronize(device)
            # Before serving, pages1..B are verified unowned. Do not use null page0.
            saved = [cache[1 : batch + 1].clone() for cache in layer.kv_cache]
            try:
                capture_stream.wait_stream(torch.cuda.current_stream(device))
                with (
                    torch.cuda.stream(capture_stream),
                    set_forward_context({layer.prefix: metadata}, self.config, num_tokens=batch * _TOKENS),
                ):
                    for _ in range(3):
                        self.original(projected, output)
                    capture_stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        self.original(projected, output)
                torch.cuda.current_stream(device).wait_stream(capture_stream)
                torch.cuda.synchronize(device)
                self.entries[batch] = {
                    "metadata": metadata,
                    "input": projected,
                    "output": output,
                    "graph": graph,
                    "signature": signature(metadata),
                    "metadata_tensor_count": sum(
                        dst.device.type != "cpu" for dst, _ in metadata_copy_pairs(metadata, metadata)
                    ),
                }
            finally:
                torch.cuda.synchronize(device)
                for cache, old in zip(layer.kv_cache, saved):
                    cache[1 : batch + 1].copy_(old)
                torch.cuda.synchronize(device)
                if any(
                    not torch.equal(
                        cache[1 : batch + 1].contiguous().view(torch.uint8), old.contiguous().view(torch.uint8)
                    )
                    for cache, old in zip(layer.kv_cache, saved)
                ):
                    raise RuntimeError("Startup scratch cache restoration failed")
        self.frozen = True

    def run(self, projected_states, output):
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        metadata = context.attn_metadata.get(self.layer.prefix) if isinstance(context.attn_metadata, dict) else None
        stamp = getattr(metadata, _STAMP_NAME, None)
        reason = None
        if stamp is None or stamp[0] is not _STAMP:
            reason = "not_cold_uniform71"
        elif cache_identity(self.layer) != self.cache:
            reason = "cache_rebound"
        elif torch.is_inference_mode_enabled() != self.inference_mode:
            reason = "inference_mode"
        elif (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) != self.precision:
            reason = "precision"
        elif torch.cuda.is_current_stream_capturing():
            reason = "outer_capture"
        elif torch.cuda.current_stream().cuda_stream != self.stream:
            reason = "execution_stream"
        elif stamp[1] not in self.entries:
            reason = "unwarmed_batch"
        else:
            entry = self.entries[stamp[1]]
            rows = stamp[1] * _TOKENS
            # Outer PIECEWISE activation padding is not a Mamba request. The
            # original op also reads/writes only num_actual_tokens rows.
            actual_input = projected_states[:rows]
            actual_output = output[:rows]
            if (
                signature(metadata) != entry["signature"]
                or tensor_layout(actual_input) != tensor_layout(entry["input"])
                or tensor_layout(actual_output) != tensor_layout(entry["output"])
            ):
                reason = "metadata_or_tensor_shape"
        if reason is not None:
            self.fallbacks[reason] += 1
            return self.original(projected_states, output)
        copied = refresh_metadata(entry["metadata"], metadata, routing_only=self.static_metadata)
        self.metadata_refresh_calls += 1
        self.metadata_copy_ops += copied
        if self.static_metadata:
            self.metadata_constant_copies_skipped += entry["metadata_tensor_count"] - copied
        entry["input"].copy_(actual_input)
        entry["graph"].replay()
        actual_output.copy_(entry["output"])
        self.replays += 1
        return None

    def status(self):
        return {
            "layer": self.layer.prefix,
            "captures": len(self.entries),
            "batches": sorted(self.entries),
            "replays": self.replays,
            "fallbacks": dict(self.fallbacks),
            "frozen": self.frozen,
            "execution_stream": self.stream,
            "inference_mode": self.inference_mode,
            "static_metadata": self.static_metadata,
            "metadata_refresh_calls": self.metadata_refresh_calls,
            "metadata_copy_ops": self.metadata_copy_ops,
            "metadata_constant_copies_skipped": self.metadata_constant_copies_skipped,
        }


def prewarm_model(model, runner, batches=tuple(range(1, 9))):
    """Prepare at worker startup; an identical pre-admission second call is a no-op."""
    if os.environ.get("EASYMAGPIE_MAMBA_PREFILL_ISLANDS") != "1":
        if hasattr(model, "_mamba_prefill_islands"):
            raise RuntimeError("Cannot disable Mamba islands after preparation")
        return {"enabled": False}
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    if tuple(batches) != tuple(range(1, 9)):
        raise ValueError("Initial capture contract is exactly batches1..8")
    if runner.input_batch.num_reqs != 0 or getattr(runner, "requests", {}):
        raise RuntimeError("Scratch pages may only be used before request admission")
    config = runner.vllm_config
    if (
        config.cache_config.mamba_cache_mode != "none"
        or config.cache_config.enable_prefix_caching
        or config.num_speculative_tokens != 0
        or config.parallel_config.tensor_parallel_size != 1
        or config.parallel_config.pipeline_parallel_size != 1
        or config.parallel_config.data_parallel_size != 1
    ):
        raise RuntimeError("Unsupported cache/speculation/parallel configuration")
    if hashlib.sha256(Path(inspect.getfile(MambaMixer2)).read_bytes()).hexdigest() != _MIXER_SHA:
        raise RuntimeError("Pinned MambaMixer2 implementation changed")
    layers = [layer for layer in model.backbone.modules() if isinstance(layer, MambaMixer2)]
    if len(layers) != 14:
        raise RuntimeError("Expected14 checkpoint Mamba layers")
    for layer in layers:
        if (
            layer.num_spec
            or len(layer.kv_cache) != 2
            or any(value.device.type != "cuda" or value.shape[0] < 9 for value in layer.kv_cache)
        ):
            raise RuntimeError("Unbound or insufficient scratch cache")
    if hasattr(model, "_mamba_prefill_islands"):
        # The benchmark wrapper may still invoke this after the production
        # worker hook. Never recapture, restore pages or replay on that call.
        islands = model._mamba_prefill_islands
        owner = getattr(model, "_mamba_prefill_runner_ref", lambda: None)
        if owner() is not runner or len(islands) != len(layers):
            raise RuntimeError("Mamba islands belong to a different or incomplete startup")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Startup validation cannot run inside another capture")
        stream = torch.cuda.current_stream().cuda_stream
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        for layer, island in zip(layers, islands, strict=True):
            if (
                island.layer is not layer
                or island.config is not config
                or not island.frozen
                or set(island.entries) != set(batches)
                or island.cache != cache_identity(layer)
                or island.stream != stream
                or not island.inference_mode
                or island.precision != precision
                or island.static_metadata != (os.environ.get("EASYMAGPIE_MAMBA_STATIC_PREFILL_METADATA") == "1")
                or layer.conv_ssm_forward is not getattr(island, "installed_forward", None)
            ):
                raise RuntimeError("Prepared Mamba island contract changed")
        model._mamba_prefill_startup["validated_reuses"] += 1
        return graph_status(model)
    started = time.monotonic()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        islands = [LayerIslands(layer, config) for layer in layers]
        for island in islands:
            island.prewarm(batches)
    install_metadata_stamp()
    for island in islands:

        def forward(self, projected_states, output, _island=island):
            return _island.run(projected_states, output)

        island.installed_forward = MethodType(forward, island.layer)
        island.layer.conv_ssm_forward = island.installed_forward
    model._mamba_prefill_islands = islands
    model._mamba_prefill_runner_ref = weakref.ref(runner)
    model._mamba_prefill_startup = {
        "seconds": time.monotonic() - started,
        "allocated_delta_bytes": torch.cuda.memory_allocated() - before,
        "scratch_pages": list(range(1, 9)),
        "scratch_snapshot_restored": True,
        "ownership_contract": "worker initialization after ordinary warmup, empty input batch, before any admission",
        "metadata_builder_sha256": _BACKEND_SHA,
        "mixer_source_sha256": _MIXER_SHA,
        "routing_metadata_builder_sha256": _BASE_BACKEND_SHA,
        "routing_stamp_propagation": True,
        "static_metadata": all(island.static_metadata for island in islands),
        "metadata_refresh_policy": "per-layer physical routing only" if islands[0].static_metadata else "all tensors",
        "metadata_copy_counter_units": "non-CPU tensor copy_ calls, including zero-element tensors; not CUDA API events",
        "graph_pool_policy": "independent pool per graph; no cross-graph activation aliases",
        "validated_reuses": 0,
    }
    return graph_status(model)


def graph_status(model):
    islands = getattr(model, "_mamba_prefill_islands", ())
    return {
        "enabled": bool(islands),
        "entries": sum(len(item.entries) for item in islands),
        "replays": sum(item.replays for item in islands),
        "frozen": bool(islands) and all(x.frozen for x in islands),
        "layers": [item.status() for item in islands],
        "startup": getattr(model, "_mamba_prefill_startup", None),
    }
