#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export rolling metadata from existing converted context tensors, without NeMo."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def export(contexts, output, duration, tail_tokens=125):
    """Record immutable saved-context provenance and the generated-tail budget."""
    import torch

    model = json.loads((output / 'config.json').read_text())
    codec = json.loads((output / 'codec_native/config.json').read_text())
    rates = codec['pre_upsample_rates'] + codec['resblock_upsample_rates']
    if not rates or any(type(rate) is not int or rate <= 0 for rate in rates):
        raise ValueError('invalid codec upsampling rates')
    fps = codec['output_sample_rate'] / math.prod(rates)
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('context duration must be positive and finite')
    total = round(duration * fps)
    if type(tail_tokens) is not int or not 0 < tail_tokens < total:
        raise ValueError('generated tail must be smaller than the total context budget')
    initial, flags = {}, {}
    for entry in contexts:
        voice = entry['speaker_id']
        if voice in initial:
            raise ValueError(f'duplicate saved context {voice!r}')
        saved = torch.load(output / 'speaker_embeddings' / f'{voice}.pt', map_location='cpu', weights_only=True)
        saved = saved['speaker_encoding'] if isinstance(saved, dict) else saved
        if (
            not isinstance(saved, torch.Tensor)
            or saved.ndim != 2
            or saved.shape[0] < 2
            or saved.shape[1] != model['embedding_dim']
            or not bool(torch.isfinite(saved).all())
        ):
            raise ValueError(f'invalid complete saved context for {voice!r}')
        flag = entry.get('original_has_audio', saved.shape[0] != 2)
        if type(flag) is not bool or (not flag and saved.shape[0] != 2):
            raise ValueError('text-only original must have two native boundary rows and a Boolean audio flag')
        flags[voice] = flag
        initial[voice] = dict(
            rows=saved.shape[0], sha256=hashlib.sha256(saved.float().contiguous().numpy().tobytes()).hexdigest()
        )
    if not initial:
        raise ValueError('rolling context requires at least one saved context')
    config = dict(
        schema_version=5,
        embedding_dim=model['embedding_dim'],
        original_has_audio=flags,
        initial_contexts=initial,
        max_generated_frames=tail_tokens,
        total_frames=total,
        frames_per_second=fps,
        total_duration_seconds=total / fps,
    )
    destination = output / 'rolling_context'
    destination.mkdir()
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-audio-duration', type=float, required=True)
    parser.add_argument('--context-tail-tokens', type=int, default=125)
    args = parser.parse_args()
    export(json.loads(args.contexts.read_text()), args.output, args.context_audio_duration, args.context_tail_tokens)


if __name__ == '__main__':
    main()
