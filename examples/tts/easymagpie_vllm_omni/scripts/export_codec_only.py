#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Export an EasyMagpie codec decoder without restoring the Mamba talker.

The full EasyMagpie checkpoint needs the Mamba extension, while codec export only
needs the standalone NeMo codec and the TTS vector-quantizer configuration.  This
small entry point creates the model facade expected by :mod:`codec_export`, making
it suitable for producing larger serving batch artifacts in a codec-capable NeMo
environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from codec_export import export_codec_decoder


class _CodecOnlyEasyMagpie:
    """The minimal EasyMagpie surface consumed by ``build_codec_decoder``."""

    def __init__(
        self,
        codec_model,
        model_cfg,
        *,
        frame_stacking_factor: int,
        num_audio_codebooks: int,
        codebook_size: int,
    ) -> None:
        self._codec_model = codec_model
        self.cfg = model_cfg
        self.frame_stacking_factor = int(frame_stacking_factor)
        self.num_audio_codebooks = int(num_audio_codebooks)
        self.codebook_size = int(codebook_size)


def _restore_codec(codec_path: Path):
    from nemo.collections.tts.models import AudioCodecModel

    cfg = AudioCodecModel.restore_from(str(codec_path), return_config=True)
    if "use_scl_loss" in cfg:
        cfg.use_scl_loss = False
    return AudioCodecModel.restore_from(str(codec_path), strict=False, override_config_path=cfg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-model", required=True, help="Standalone NeMo codec checkpoint.")
    parser.add_argument(
        "--model-config",
        required=True,
        help="EasyMagpie NeMo model_config.yaml; supplies the vector_quantizer config.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for codec_decoder.pt2 and metadata.")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Fixed model-frame chunk size (default: 15 unless dynamic bounds are supplied).",
    )
    parser.add_argument("--min-frames", type=int, default=None, help="Minimum frame count for a dynamic export.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frame count for a dynamic export.")
    parser.add_argument("--max-batch-size", type=int, required=True, help="Maximum dynamic batch accepted by export.")
    parser.add_argument("--frame-stacking-factor", type=int, default=2)
    parser.add_argument("--num-audio-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")
    dynamic_frames = args.min_frames is not None or args.max_frames is not None
    if dynamic_frames and (args.min_frames is None or args.max_frames is None):
        raise ValueError("Dynamic codec export requires both --min-frames and --max-frames.")
    if dynamic_frames and args.frames is not None:
        raise ValueError("Use either --frames or --min-frames/--max-frames, not both.")
    fixed_frames = None if dynamic_frames else int(args.frames or 15)

    codec_path = Path(args.codec_model)
    config_path = Path(args.model_config)
    if not codec_path.is_file():
        raise FileNotFoundError(codec_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    device = torch.device(args.device)
    codec = _restore_codec(codec_path)
    facade = _CodecOnlyEasyMagpie(
        codec,
        OmegaConf.load(config_path),
        frame_stacking_factor=args.frame_stacking_factor,
        num_audio_codebooks=args.num_audio_codebooks,
        codebook_size=args.codebook_size,
    )

    outdir = Path(args.output_dir)
    program_path = outdir / "codec_decoder.pt2"
    metadata_path = outdir / "codec_export.json"
    metadata = export_codec_decoder(
        facade,
        program_path,
        metadata_path,
        frames=fixed_frames,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        max_batch_size=args.max_batch_size,
        device=device,
        atol=args.atol,
    )

    # Validate the actual serving boundaries, not only the exporter example
    # batch. This catches invalid dynamic batch or frame guards immediately.
    decoder = torch.export.load(program_path).module().to(device=device)
    q = int(metadata["num_stacked_codebooks"])
    min_frames = int(metadata.get("min_frames", metadata["frames"]))
    max_frames = int(metadata.get("max_frames", metadata["frames"]))
    frame_sizes = sorted({min_frames, (min_frames + max_frames) // 2, max_frames})
    for batch in sorted({1, min(16, args.max_batch_size), args.max_batch_size}):
        for frames in frame_sizes:
            codes = torch.randint(
                0,
                int(metadata["codebook_size"]),
                (batch, frames, q),
                device=device,
                dtype=torch.long,
            )
            audio = decoder(codes)
            expected = (batch, frames * int(metadata["samples_per_frame"]))
            if tuple(audio.shape) != expected:
                raise RuntimeError(f"batch {batch}, frames {frames}: got {tuple(audio.shape)}, expected {expected}")
            print(f"validated batch={batch} frames={frames}: audio={tuple(audio.shape)}")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
