#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build a dynamic-batch TensorRT codec engine from EasyMagpie ``codec.onnx``."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import onnx


def _infer_num_quantizers(onnx_path: Path) -> int:
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name != "audio_codes":
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) >= 3 and dims[2].dim_value > 0:
            return int(dims[2].dim_value)
    raise RuntimeError(f"Could not infer static audio_codes channel count from {onnx_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--trt-path", required=True)
    parser.add_argument("--trtexec-bin", default="/usr/bin/trtexec")
    parser.add_argument("--batch-profile", nargs=3, type=int, required=True, metavar=("MIN", "OPT", "MAX"))
    parser.add_argument("--frames-profile", nargs=3, type=int, default=[15, 15, 15], metavar=("MIN", "OPT", "MAX"))
    parser.add_argument("--fp32", action="store_true", help="Build an FP32 engine (default is FP16).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    onnx_path = Path(args.onnx_path)
    trt_path = Path(args.trt_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    trtexec = shutil.which(args.trtexec_bin) if "/" not in args.trtexec_bin else args.trtexec_bin
    if trtexec is None or not Path(trtexec).is_file():
        raise FileNotFoundError(f"trtexec not found: {args.trtexec_bin}")

    nq = _infer_num_quantizers(onnx_path)
    batch = tuple(args.batch_profile)
    frames = tuple(args.frames_profile)
    if min(*batch, *frames) <= 0 or not (batch[0] <= batch[1] <= batch[2]):
        raise ValueError(f"Invalid batch profile: {batch}")

    def shape(b: int, f: int) -> str:
        return f"{b}x{f}x{nq}"

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--minShapes=audio_codes:{shape(batch[0], frames[0])}",
        f"--optShapes=audio_codes:{shape(batch[1], frames[1])}",
        f"--maxShapes=audio_codes:{shape(batch[2], frames[2])}",
    ]
    if not args.fp32:
        command.append("--fp16")
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"TensorRT engine saved to {trt_path}")


if __name__ == "__main__":
    main()
