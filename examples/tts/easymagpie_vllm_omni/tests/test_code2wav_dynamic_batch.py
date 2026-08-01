# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression coverage for dynamic Code2Wav eager batching."""
from __future__ import annotations

import torch

from easymagpie_vllm_omni.code2wav import _group_codec_jobs_for_decode, _stack_codec_jobs


def _job(index: int, frames: int) -> tuple[int, torch.Tensor, int, int]:
    codes = torch.arange(frames * 2, dtype=torch.long).reshape(frames, 2) + (index * 100)
    return index, codes, frames, 0


def test_dynamic_codec_batches_nearby_frame_lengths_up_to_export_limit():
    jobs = [_job(0, 1), _job(1, 3), _job(2, 4), _job(3, 5), _job(4, 6), _job(5, 200)]

    batches = _group_codec_jobs_for_decode(
        jobs,
        max_batch_size=3,
        dynamic_frames=True,
        max_frames=200,
    )

    assert [[job[2] for job in batch] for batch in batches] == [[1, 3, 4], [5, 6], [200]]


def test_dynamic_codec_batch_padding_repeats_tail_and_keeps_target_length():
    jobs = [_job(0, 1), _job(1, 3), _job(2, 4)]

    batch, target_frames = _stack_codec_jobs(jobs)

    assert target_frames == 4
    assert tuple(batch.shape) == (3, 4, 2)
    assert torch.equal(batch[0, 1:], jobs[0][1][-1:].expand(3, -1))
    assert torch.equal(batch[1, 3:], jobs[1][1][-1:])
    assert torch.equal(batch[2], jobs[2][1])
