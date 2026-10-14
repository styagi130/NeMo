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
"""Run this checkout's serving CPU suite in the pinned vLLM-Omni environment."""
import os
import sys
from pathlib import Path


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('LOGNAME', 'easymagpie-tests')
    import pytest
    import torch
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    vllm.platforms._current_platform = CpuPlatform()
    directory = Path(__file__).resolve().parent
    args = sys.argv[1:] or [str(directory)]
    result = pytest.main(['-q', '-o', 'addopts=', '-p', 'no:cacheprovider', '--confcutdir=' + str(directory), *args])
    assert not torch.cuda.is_initialized()
    return int(result)


if __name__ == '__main__':
    raise SystemExit(main())
