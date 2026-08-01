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
"""Focused regression coverage for Triton's dynamic codec packet drain."""
from __future__ import annotations

import queue
import sys
import threading
import time
import types

import torch

# The regular test image does not load Triton's Python backend extension.  The
# worker under test needs it only when it sends a real response, which this test
# replaces with a small in-memory recorder.
sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("triton_python_backend_utils"))

from triton_backend.model import _GEN_DONE, TritonPythonModel


def test_dynamic_codec_worker_drains_all_post_first_frames_on_deadline():
    """The first packet is small; the next one is the whole buffered window."""
    model = TritonPythonModel.__new__(TritonPythonModel)
    model.codec_left_context = 2
    model.codec_chunk_size = 8
    model.first_chunk_frames = 1
    model.codec_dynamic_chunk_wait_us = 10_000
    model.codec_noop = True
    model.codec_noop_spf = 1
    model._spf = None

    sent: list[tuple[int, bool]] = []
    first_sent = threading.Event()
    post_sent = threading.Event()

    def send_audio(_sender, audio, final):
        sent.append((int(audio.size), bool(final)))
        if len(sent) == 1:
            first_sent.set()
        elif len(sent) == 2:
            post_sent.set()

    model._send_audio = send_audio
    codec_q: queue.Queue = queue.Queue()
    state = {"t_first_audio": None, "error": None}
    worker = threading.Thread(target=model._codec_worker, args=(codec_q, object(), state, 0))
    worker.start()

    # One first frame is emitted immediately, independently of the dynamic
    # deadline.  The next cumulative snapshot adds four frames (< capacity 6).
    codec_q.put((torch.zeros((1, 16), dtype=torch.int64), False))
    assert first_sent.wait(timeout=1)
    codec_q.put((torch.zeros((5, 16), dtype=torch.int64), False))

    # The worker must emit all four buffered post-first frames in one packet,
    # rather than reverting to a fixed one-frame cadence.
    assert post_sent.wait(timeout=1)
    codec_q.put(_GEN_DONE)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert state["error"] is None
    assert sent == [(1, False), (4, False), (0, True)]
