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
"""Register EasyMagpieTTS models and pipeline with vLLM-Omni."""

_TARGET = "easymagpie_vllm_omni.easymagpie:EasyMagpieTTSForConditionalGeneration"
_ARCHS = ("EasyMagpieTTS", "EasyMagpieTTSForConditionalGeneration")
_CODEC_ARCH = "EasyMagpieCodecForConditionalGeneration"
_CODEC_TARGET = "easymagpie_vllm_omni.codec.model:EasyMagpieCodecForConditionalGeneration"


def register() -> None:
    """Register model architectures in both vLLM registries."""
    from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    try:
        AutoConfig.register(EasyMagpieCodecConfig.model_type, EasyMagpieCodecConfig)
    except ValueError:
        # Plugin reloads may encounter the same model type already registered.
        pass
    MODELS_CONFIG_MAP.setdefault(_CODEC_ARCH, MambaModelConfig)

    registries = [ModelRegistry]
    omni_available = False
    try:
        from vllm_omni.model_executor.models import OmniModelRegistry

        registries.append(OmniModelRegistry)
        omni_available = True
    except Exception:
        # vllm_omni not installed — stock vLLM registration is enough.
        pass

    for registry in registries:
        for arch in _ARCHS:
            if arch not in registry.get_supported_archs():
                registry.register_model(arch, _TARGET)
        if _CODEC_ARCH not in registry.get_supported_archs():
            registry.register_model(_CODEC_ARCH, _CODEC_TARGET)

    if omni_available:
        # These hooks optimize the native two-stage pipeline only.  They leave
        # the codec as a vLLM model: no TensorRT plan or Triton codec backend is
        # registered here.
        import os
        if os.environ.get("EASYMAGPIE_NATIVE_OPTIMIZATIONS", "0") == "1":
            _install_cuda_ipc_connector()
            _install_stage0_cuda_payload_hook()
            _install_stage0_inprocess_burst()
            # The transfer half is independent of the legacy scheduler
            # monkeypatch: it replaces Stage 0's serial connector save loop
            # with ordered-per-stream, parallel-across-stream publication.
            if os.environ.get("EASYMAGPIE_CODEC_TRANSFER_PARALLEL", "0") == "1":
                _install_codec_transfer_parallelism()
            # The fresh vLLM-Omni scheduler changed its callback lifecycle.
            # Keep the legacy scheduler monkeypatch opt-in until its native
            # API adaptation is validated; the stage processor still performs
            # dynamic queue draining without it.
            if os.environ.get("EASYMAGPIE_CODEC_MICROBATCH", "0") == "1":
                _install_codec_microbatching()
        _register_pipeline()
        _register_serving_adapter()


def _register_serving_adapter() -> None:
    """Install optional ``/v1/audio/speech`` support."""
    import logging

    try:
        from easymagpie_vllm_omni.serving_adapter import apply_serving_patches

        apply_serving_patches()
    except Exception:  # pragma: no cover - serving support is best-effort
        logging.getLogger(__name__).exception(
            "EasyMagpie: /v1/audio/speech serving support could not be installed "
            "(model + pipeline registration still succeeded)."
        )


def _install_codec_microbatching() -> None:
    """Install bounded Stage-0 transfer and Stage-1 cohorting hooks."""
    import logging

    try:
        from easymagpie_vllm_omni.codec_microbatch import (
            install_scheduler_microbatch_patch,
            install_transfer_microbatch_patch,
        )

        install_transfer_microbatch_patch()
        install_scheduler_microbatch_patch()
    except Exception:  # pragma: no cover - retain stock serving on API drift.
        logging.getLogger(__name__).exception("EasyMagpie: native codec microbatch hooks could not be installed.")


def _install_codec_transfer_parallelism() -> None:
    """Install only Stage-0 parallel transfer for the current native scheduler."""
    import logging

    try:
        from easymagpie_vllm_omni.codec_microbatch import install_transfer_microbatch_patch

        install_transfer_microbatch_patch()
    except Exception:  # pragma: no cover - retain stock serving on API drift.
        logging.getLogger(__name__).exception("EasyMagpie: parallel codec transfer hook could not be installed.")


def _install_cuda_ipc_connector() -> None:
    """Register the optional same-GPU CUDA-IPC connector."""
    import logging

    try:
        from easymagpie_vllm_omni.cuda_ipc_connector import install_cuda_ipc_connector

        install_cuda_ipc_connector()
    except Exception:  # pragma: no cover - shared-memory remains a safe fallback.
        logging.getLogger(__name__).exception("EasyMagpie: CUDA-IPC connector could not be installed.")


def _install_stage0_cuda_payload_hook() -> None:
    """Keep Stage-0 acoustic codes resident until the connector consumes them."""
    import logging

    try:
        from easymagpie_vllm_omni.runner import install_stage0_cuda_payload_patch

        install_stage0_cuda_payload_patch()
    except Exception:  # pragma: no cover - retain the generic vLLM payload path.
        logging.getLogger(__name__).exception("EasyMagpie: CUDA Stage-0 payload hook could not be installed.")


def _install_stage0_inprocess_burst() -> None:
    """Install the opt-in Stage-0 AR cadence hook."""
    import logging

    try:
        from easymagpie_vllm_omni.stage0_burst import install_stage0_inprocess_burst

        install_stage0_inprocess_burst()
    except Exception:  # pragma: no cover - stock EngineCore is a safe fallback.
        logging.getLogger(__name__).exception("EasyMagpie: Stage-0 burst hook could not be installed.")


def _register_pipeline() -> None:
    """Register the two-stage and talker-only pipelines."""
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE, EASYMAGPIE_TALKER_ONLY_PIPELINE
    try:
        from vllm_omni.config.pipeline_registry import register_pipeline
    except ImportError:  # vLLM-Omni 0.24+ moved the registration helper.
        from vllm_omni.config.stage_config import register_pipeline

    register_pipeline(EASYMAGPIE_PIPELINE)
    register_pipeline(EASYMAGPIE_TALKER_ONLY_PIPELINE)
