#!/usr/bin/env python3
"""Micro-tune the EasyMagpie routed-expert Triton kernel on H100.

This uses the production Stage-0 MoE dimensions (24 experts, hidden size 1536,
intermediate size 768, top-k 4) and reports median kernel latency for each
candidate.  It does not mutate the production tile JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
from vllm.model_executor.layers.fused_moe import fused_experts, override_config


def benchmark_candidate(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict[str, int],
) -> float:
    with override_config(config):
        # Force compilation before the timed region.
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
        torch.cuda.synchronize()
        return float(
            triton.testing.do_bench(
                lambda: fused_experts(hidden_states, w1, w2, topk_weights, topk_ids),
                warmup=25,
                rep=100,
                return_mode="median",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[24, 32, 48, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260729)
    device = torch.device("cuda")
    dtype = torch.float16
    experts, hidden, intermediate, topk = 24, 1536, 768, 4

    # vLLM packs gate and up projections along w1's output dimension.
    w1 = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype)
    w2 = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype)

    candidates = [
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for block_m, block_n, block_k, group_m, num_warps, num_stages in itertools.product(
            (16, 32),
            (64, 128),
            (64, 128, 256),
            (1, 16),
            (4,),
            (2, 3, 4, 5),
        )
    ]

    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "experts": experts,
            "hidden": hidden,
            "intermediate": intermediate,
            "topk": topk,
            "dtype": str(dtype),
        },
        "batches": {},
    }

    for batch in args.batches:
        hidden_states = torch.randn(batch, hidden, device=device, dtype=dtype)
        logits = torch.randn(batch, experts, device=device, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
        topk_weights = topk_weights.to(dtype)
        topk_ids = topk_ids.to(torch.int32)

        results: list[dict[str, object]] = []
        for config in candidates:
            try:
                latency_ms = benchmark_candidate(
                    hidden_states, w1, w2, topk_weights, topk_ids, config
                )
            except Exception as error:
                results.append({"config": config, "error": repr(error)})
                continue
            results.append({"config": config, "latency_ms": latency_ms})

        valid = [result for result in results if "latency_ms" in result]
        valid.sort(key=lambda result: float(result["latency_ms"]))
        report["batches"][str(batch)] = {
            "winner": valid[0] if valid else None,
            "top10": valid[:10],
            "failures": len(results) - len(valid),
        }
        winner = valid[0]
        print(f"B{batch}: {winner['latency_ms']:.6f} ms {winner['config']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
