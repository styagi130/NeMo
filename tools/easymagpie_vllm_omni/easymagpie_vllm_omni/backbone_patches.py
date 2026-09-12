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
"""Compatibility fixes for the EasyMagpie backbone on the pinned vLLM 0.26.0."""
from __future__ import annotations

from dataclasses import replace
from functools import wraps

import torch
import vllm.v1.attention.backends.mamba_attn as _mamba_attn
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation, get_act_fn
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder

logger = init_logger(__name__)


def patch_mamba_prefill_initial_states() -> None:
    """Use vLLM 0.26's exact CPU prefill lengths for the initial-state flag."""
    orig = Mamba2AttentionMetadataBuilder.build
    if getattr(orig, "_easymagpie_patched", False):
        return

    @wraps(orig)
    def patched(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        architectures = getattr(self.vllm_config.model_config.hf_config, "architectures", None)
        seq_lens = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
        query_start = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if (
            not isinstance(architectures, (list, tuple))
            or not any(a in ("EasyMagpieTTSForConditionalGeneration", "EasyMagpieTTS") for a in architectures)
            or not isinstance(seq_lens, torch.Tensor)
            or seq_lens.device.type != "cpu"
            or seq_lens.ndim != 1
            or seq_lens.shape[0] < common_attn_metadata.num_reqs
            or not isinstance(query_start, torch.Tensor)
            or query_start.device.type != "cpu"
            or query_start.shape != (common_attn_metadata.num_reqs + 1,)
        ):
            return orig(self, common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs)

        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            prev_last_scheduled_idx=kwargs.get("prev_last_scheduled_idx"),
        )
        prep_initial_states = False
        cu_chunk_seqlen_p = seq_idx_p = last_chunk_indices_p = None
        if common.num_prefills > 0:
            if common.has_initial_states_p is not None:
                # Same prefill-only CPU formula as _build_chunk_metadata_tensors;
                # decode upper bounds may be optimistic during async speculation.
                query_lens = torch.diff(query_start[-common.num_prefills - 1 :])
                computed = seq_lens[common.num_reqs - common.num_prefills : common.num_reqs] - query_lens
                prep_initial_states = torch.any(computed > 0).item()
            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = self._build_chunk_metadata_tensors(
                self.chunk_size, common, common_attn_metadata
            )
        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )

    patched._easymagpie_patched = True
    Mamba2AttentionMetadataBuilder.build = patched
    logger.info("EasyMagpie Mamba prefill CPU initial-state flag installed")


def patch_mamba_streaming_decode() -> None:
    """Classify one-token stream extensions as decodes.

    Decode CUDA graphs require decode metadata and a refreshed Mamba state
    index. Multi-token context inputs remain prefills. Streaming callers must
    generate exactly one new token per step.
    """
    orig = _mamba_attn.split_decodes_and_prefills
    if getattr(orig, "_easymagpie_patched", False):
        return

    def patched(
        common_attn_metadata,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        treat_short_extends_as_decodes: bool = True,
    ):
        return orig(
            common_attn_metadata,
            decode_threshold=decode_threshold,
            require_uniform=require_uniform,
            treat_short_extends_as_decodes=True,
        )

    patched._easymagpie_patched = True
    _mamba_attn.split_decodes_and_prefills = patched
    logger.info("Mamba streaming-decode classification patch installed")


def patch_shared_expert_activation(backbone) -> int:
    """Make shared experts honor ``mlp_hidden_act`` from the model config.

    vLLM 0.26's ``NemotronHMLP`` hard-codes ReLU² even though routed experts
    read ``mlp_hidden_act``. NeMo uses the configured activation for both.
    """
    activation_name = getattr(getattr(backbone, "config", None), "mlp_hidden_act", None)
    if not isinstance(activation_name, str) or not activation_name:
        raise ValueError("Nemotron-H config must provide a non-empty mlp_hidden_act")

    expected_type = type(get_act_fn(activation_name))
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        se = getattr(mixer, "shared_experts", None)
        if se is None:
            continue
        if isinstance(se.act_fn, expected_type):
            continue
        if not isinstance(se.act_fn, ReLUSquaredActivation):
            raise RuntimeError(
                "vLLM Nemotron-H shared-expert activation implementation changed; "
                "review the compatibility patch before replacing it"
            )
        se.act_fn = get_act_fn(activation_name)
        patched += 1
    logger.info("%s shared-expert activation fix installed on %d layers", activation_name, patched)
    return patched


def patch_moe_routed_scale(backbone) -> int:
    """Apply the routed scaling factor omitted from Nemotron-H FP16 outputs."""
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        scale = float(getattr(mixer, "routed_scaling_factor", 1.0))
        if scale == 1.0:
            continue

        def _scale_output(_mod, _inp, out, _scale=scale):
            # FusedMoE only defers the scale in FP16; leave other dtypes alone.
            if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
                return out * _scale
            return out

        mixer.register_forward_hook(_scale_output)
        patched += 1
    logger.info("FP16 MoE routed-scale fix installed on %d layers", patched)
    return patched
