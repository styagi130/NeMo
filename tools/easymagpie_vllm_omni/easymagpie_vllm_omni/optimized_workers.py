# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Startup policy for the opt-in, measured H100 deployment profile."""

import gc
import os

import torch

from .runner import EasyMagpieCodecGPUGenerationWorker, EasyMagpieGPUARWorker


class H100OptimizedStartup:
    """Set the measured precision policy and finish graph capture before admission."""

    is_codec = False

    def init_device(self):
        _set_precision()
        return super().init_device()

    def load_model(self, *args, **kwargs):
        _set_precision()
        return super().load_model(*args, **kwargs)

    def compile_or_warm_up_model(self):
        _set_precision()
        result = super().compile_or_warm_up_model()
        if self.is_codec and os.environ.get("EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS") == "1":
            # Match the measured profile: 192 exact shapes, frozen before traffic.
            # Capacity is 128; larger codec batches retain the eager fallback.
            with torch.inference_mode():
                self.model_runner.model.prewarm_codec_graphs(range(1, 65), frame_sizes=(2, 6, 8))
            self.codec_prewarm_completed = True
            gc.freeze()
        return result


class H100OptimizedTalkerWorker(H100OptimizedStartup, EasyMagpieGPUARWorker):
    """Warm the native talker and its Mamba prefill graphs before traffic."""


class H100OptimizedCodecWorker(H100OptimizedStartup, EasyMagpieCodecGPUGenerationWorker):
    """Warm the FP32 codec's exact-shape graphs before traffic."""

    is_codec = True


def _set_precision():
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        raise RuntimeError("Unset NVIDIA_TF32_OVERRIDE to use the measured H100 precision policy")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
