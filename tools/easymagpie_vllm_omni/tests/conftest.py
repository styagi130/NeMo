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
"""CPU regression boundary for the separately installed vLLM serving stack."""
import importlib.util
import sys
from pathlib import Path

import pytest

EASYMAGPIE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EASYMAGPIE_ROOT))
SERVING_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'vllm', 'vllm_omni'))
CPU_ONLY_FILES = {'test_dds_routing_pool.py', 'test_query_dispatch_pool.py', 'test_optimized_workers.py'}
collect_ignore = (
    []
    if SERVING_AVAILABLE
    else [path.name for path in Path(__file__).parent.glob('test_*.py') if path.name not in CPU_ONLY_FILES]
)


@pytest.fixture(autouse=True)
def cuda_remains_uninitialized():
    if not SERVING_AVAILABLE:
        yield
        return
    import torch

    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
