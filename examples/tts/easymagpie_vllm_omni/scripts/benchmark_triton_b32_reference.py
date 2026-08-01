#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Repeatable EasyMagpie Triton B32 reference benchmark.

Run this *inside* the Triton container (where ``tritonclient.grpc`` is present):

  docker exec easymagpie-triton-dynamic-bs32 python3 \
    /workspace/examples/tts/easymagpie_vllm_omni/scripts/benchmark_triton_b32_reference.py

All requests in a cohort use the same text.  One warm-up B32 cohort initializes
the B32 CUDA-graph/TensorRT specializations; the reported runs are steady-state.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import numpy as np
import tritonclient.grpc as grpcclient


REFERENCE_TEXT = "A small bird flew over the quiet garden while the afternoon sun warmed the stone path."


def run_cohort(url: str, model: str, text: str, concurrency: int, timeout_s: float) -> dict[str, float]:
    """Issue one same-prompt decoupled cohort and return streaming metrics."""
    completed = threading.Event()
    lock = threading.Lock()
    pending = {str(i) for i in range(concurrency)}
    first: dict[str, float] = {}
    final: dict[str, float] = {}
    samples: dict[str, int] = {}
    errors: list[str] = []
    t0 = time.perf_counter()

    def callback(result, error) -> None:
        request_id = result.get_response().id if result is not None else None
        with lock:
            if error is not None:
                errors.append(str(error))
                pending.discard(request_id)
            else:
                audio = result.as_numpy("audio")
                now = time.perf_counter()
                if audio is not None and audio.size:
                    first.setdefault(request_id, now)
                    samples[request_id] = samples.get(request_id, 0) + int(audio.size)
                marker = getattr(result.get_response(), "parameters", {}).get("triton_final_response")
                if marker is not None and getattr(marker, "bool_param", False):
                    final[request_id] = now
                    pending.discard(request_id)
            if not pending:
                completed.set()

    client = grpcclient.InferenceServerClient(url, verbose=False)
    client.start_stream(callback=callback)
    try:
        for request_id in list(pending):
            request = grpcclient.InferInput("text", [1, 1], "BYTES")
            request.set_data_from_numpy(np.asarray([[text.encode("utf-8")]], dtype=object))
            client.async_stream_infer(
                model,
                [request],
                request_id=request_id,
                outputs=[grpcclient.InferRequestedOutput("audio")],
                enable_empty_final_response=True,
            )
        if not completed.wait(timeout_s):
            raise TimeoutError(f"Timed out with {len(pending)} incomplete requests")
    finally:
        client.stop_stream()

    wall_s = time.perf_counter() - t0
    if errors:
        raise RuntimeError(errors[0])
    if len(samples) != concurrency or len(final) != concurrency:
        raise RuntimeError(f"Expected {concurrency} audio/final responses, got {len(samples)}/{len(final)}")

    ttfa_ms = [(first[key] - t0) * 1000 for key in samples]
    total_ms = [(final[key] - t0) * 1000 for key in samples]
    audio_s = [samples[key] / 22050.0 for key in samples]
    return {
        "ttfa_mean_ms": statistics.mean(ttfa_ms),
        "ttfa_p95_ms": sorted(ttfa_ms)[int(0.95 * (concurrency - 1))],
        "total_mean_ms": statistics.mean(total_ms),
        "per_request_rtfx": statistics.mean(audio / (latency / 1000) for audio, latency in zip(audio_s, total_ms)),
        "aggregate_rtfx": sum(audio_s) / wall_s,
        "audio_s_each": statistics.mean(audio_s),
        "wall_ms": wall_s * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--model", default="easymp")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3, help="Measured steady-state cohorts")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--text", default=REFERENCE_TEXT)
    args = parser.parse_args()

    for index in range(args.warmup_runs):
        run_cohort(args.url, args.model, args.text, args.concurrency, args.timeout)
        print(f"warmup B{args.concurrency} cohort {index + 1}/{args.warmup_runs}: complete")

    results = []
    for index in range(args.runs):
        result = run_cohort(args.url, args.model, args.text, args.concurrency, args.timeout)
        results.append(result)
        print(
            f"run {index + 1}: TTFA {result['ttfa_mean_ms']:.1f} ms "
            f"(p95 {result['ttfa_p95_ms']:.1f}), total {result['total_mean_ms']:.1f} ms, "
            f"RTFX/request {result['per_request_rtfx']:.3f}, aggregate {result['aggregate_rtfx']:.3f}"
        )

    keys = ("ttfa_mean_ms", "total_mean_ms", "per_request_rtfx", "aggregate_rtfx")
    mean = {key: statistics.mean(result[key] for result in results) for key in keys}
    print(
        "reference steady-state mean: "
        f"TTFA {mean['ttfa_mean_ms']:.1f} ms, total {mean['total_mean_ms']:.1f} ms, "
        f"RTFX/request {mean['per_request_rtfx']:.3f}, aggregate {mean['aggregate_rtfx']:.3f}"
    )


if __name__ == "__main__":
    main()
