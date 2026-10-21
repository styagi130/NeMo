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
import sys

import torch

from .runner import EasyMagpieCodecGPUGenerationWorker, EasyMagpieGPUARWorker


class H100OptimizedStartup:
    """Set the measured precision policy and finish graph capture before admission."""

    is_codec = False
    cudnn_tf32 = True
    codec_graph_max_batch = 64
    codec_graph_descending = False
    talker_switch_interval = None
    fp16_codec = False

    def init_device(self):
        self._set_precision()
        if not self.is_codec and self.talker_switch_interval is not None:
            sys.setswitchinterval(self.talker_switch_interval)
        return super().init_device()

    def load_model(self, *args, **kwargs):
        self._set_precision()
        return super().load_model(*args, **kwargs)

    def compile_or_warm_up_model(self):
        self._set_precision()
        result = super().compile_or_warm_up_model()
        if self.is_codec and os.environ.get("EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS") == "1":
            # Finish exact-shape capture before admission; larger groups keep
            # the eager fallback. The C128 and C256 profiles retain their own
            # capture order and coverage.
            options = {'frame_sizes': (2, 6, 8)}
            if self.codec_graph_descending:
                options['descending'] = True
            with torch.inference_mode():
                self.model_runner.model.prewarm_codec_graphs(range(1, self.codec_graph_max_batch + 1), **options)
            self.codec_prewarm_completed = True
            gc.freeze()
        return result

    def _set_precision(self):
        _set_precision(cudnn_tf32=self.cudnn_tf32)
        if self.is_codec and self.fp16_codec:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False


class H100OptimizedTalkerWorker(H100OptimizedStartup, EasyMagpieGPUARWorker):
    """Warm the native talker and its Mamba prefill graphs before traffic."""


class H100OptimizedCodecWorker(H100OptimizedStartup, EasyMagpieCodecGPUGenerationWorker):
    """Warm the FP32 codec's exact-shape graphs before traffic."""

    is_codec = True


class H100C256Startup(H100OptimizedStartup):
    """Retained C256 precision and graph coverage, independent of C128 defaults."""

    cudnn_tf32 = False
    codec_graph_max_batch = 128
    codec_graph_descending = True
    talker_switch_interval = 0.001
    fp16_codec = True


class H100C256TalkerWorker(H100C256Startup, EasyMagpieGPUARWorker):
    """C256 talker with FP32 SSM state and the retained startup policy."""

    def load_model(self, *args, **kwargs):
        blocking = os.environ.get('EASYMAGPIE_TALKER_BLOCKING_COPY_EVENT', '0')
        if blocking not in ('0', '1'):
            raise ValueError('EASYMAGPIE_TALKER_BLOCKING_COPY_EVENT must be 0 or 1')
        result = super().load_model(*args, **kwargs)
        self.model_runner._omni_snapshot_blocking_event = blocking == '1'
        return result


class H100C256CodecWorker(H100C256Startup, EasyMagpieCodecGPUGenerationWorker):
    """C256 FP16 codec with FP32 accumulation and 384 prewarmed graph shapes."""

    is_codec = True


def _set_precision(*, cudnn_tf32=True):
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        raise RuntimeError("Unset NVIDIA_TF32_OVERRIDE to use the measured H100 precision policy")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
