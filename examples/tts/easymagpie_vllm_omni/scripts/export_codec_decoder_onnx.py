#!/usr/bin/env python3
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
"""Export the standalone EasyMagpie codec decoder to bounded-shape ONNX.

The resulting graph accepts the raw stacked EasyMagpie codes:
``audio_codes: int64 [batch, model_frames, 16]`` and returns fp32 waveform
samples. Batch is dynamic; optionally the model-frame axis is dynamic too,
which lets a TensorRT codec plan drain a variable amount of queued audio after
the low-latency first packet.

This shares the exact codec wrapper used by ``export_codec_only.py`` and the
pure-vLLM ``torch.export`` artifact, avoiding a full Mamba-talker's restore.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from codec_export import build_codec_decoder
from export_codec_only import _CodecOnlyEasyMagpie, _restore_codec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-model", required=True, help="Standalone NeMo codec checkpoint.")
    parser.add_argument(
        "--model-config",
        required=True,
        help="EasyMagpie NeMo model_config.yaml; supplies the model vector quantizer.",
    )
    parser.add_argument("--onnx-path", required=True, help="Destination ONNX path.")
    parser.add_argument("--frames", type=int, required=True, help="TensorRT profile maximum model-frame capacity.")
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help="Export a symbolic model-frame axis bounded later by the TensorRT profile.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Export/parity example batch size.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=2e-3)
    return parser.parse_args()


def _build_wrapper(args: argparse.Namespace) -> tuple[torch.nn.Module, dict]:
    codec_path = Path(args.codec_model)
    config_path = Path(args.model_config)
    if not codec_path.is_file():
        raise FileNotFoundError(codec_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.frames <= 0 or args.batch_size <= 0:
        raise ValueError("--frames and --batch-size must be positive")

    device = torch.device(args.device)
    codec = _restore_codec(codec_path)
    facade = _CodecOnlyEasyMagpie(
        codec,
        OmegaConf.load(config_path),
        frame_stacking_factor=2,
        num_audio_codebooks=8,
        codebook_size=1024,
    )
    return build_codec_decoder(facade, device)


def _verify_parity(
    wrapper: torch.nn.Module,
    onnx_path: Path,
    example: torch.Tensor,
    *,
    atol: float,
) -> None:
    """Run an optional ONNX Runtime parity check when the package is present."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime unavailable; skipped ONNX parity check")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if example.is_cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    with torch.inference_mode():
        reference = wrapper(example).detach().cpu().float().numpy()
    actual = session.run(["audio_values"], {"audio_codes": example.cpu().numpy()})[0]
    diff = float(np.max(np.abs(reference - actual)))
    print(f"ONNX parity provider={session.get_providers()[0]} max_abs_diff={diff:.7g}")
    if diff > atol:
        raise RuntimeError(f"ONNX parity failed: max_abs_diff={diff:.7g}, atol={atol:.7g}")


def main() -> None:
    args = _parse_args()
    # Match ONNX Runtime's full-fp32 matmul behavior for a meaningful parity
    # comparison on Ampere-class GPUs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    wrapper, info = _build_wrapper(args)
    device = torch.device(args.device)
    quantizers = int(info["num_stacked_codebooks"])
    codebook_size = int(info["codebook_size"])
    example = torch.randint(
        0,
        codebook_size,
        (args.batch_size, args.frames, quantizers),
        dtype=torch.long,
        device=device,
    )

    output_path = Path(args.onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(output_path),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch", **({1: "frames"} if args.dynamic_frames else {})},
                "audio_values": {0: "batch", **({1: "samples"} if args.dynamic_frames else {})},
            },
        )

    import onnx

    onnx.checker.check_model(str(output_path))
    _verify_parity(wrapper, output_path, example, atol=args.atol)
    print(
        "Exported "
        f"{output_path}: batch=[1,*], frames={'[dynamic,0..' if args.dynamic_frames else ''}{args.frames}{']' if args.dynamic_frames else ''}, quantizers={quantizers}, "
        f"samples_per_frame={info['output_sample_rate'] // 25 * 2}"
    )


if __name__ == "__main__":
    main()
