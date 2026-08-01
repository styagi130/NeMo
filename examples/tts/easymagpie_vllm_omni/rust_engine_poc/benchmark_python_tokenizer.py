#!/usr/bin/env python3
"""Measure the exact Python AutoTokenizer path used by EasyMagpie Stage 0."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def load_texts(path: Path | None, text: str) -> list[str]:
    if path is None:
        return [text]
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            _, value = line.split("\t", 1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} must contain '<id>\\t<text>'") from error
        texts.append(value.strip())
    if not texts:
        raise ValueError(f"{path} contains no texts")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--text", default="Hello from the Python tokenizer benchmark.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")

    from transformers import AutoTokenizer

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_seconds = time.perf_counter() - load_started
    texts = load_texts(args.text_file, args.text)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be greater than zero")
        texts = [texts[index % len(texts)] for index in range(args.batch_size)]

    def tokenize_pass() -> int:
        if args.batch:
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            return sum(len(token_ids) + 1 for token_ids in encoded)
        return sum(
            len(tokenizer.encode(text, add_special_tokens=False)) + 1
            for text in texts
        )

    for _ in range(args.warmup_iterations):
        tokenize_pass()

    total_target_tokens = 0
    started = time.perf_counter()
    for _ in range(args.iterations):
        total_target_tokens += tokenize_pass()
    elapsed = time.perf_counter() - started
    total_texts = len(texts) * args.iterations

    print("tokenizer_mode=python_auto_tokenizer")
    print(f"tokenizer_load_ms={load_seconds * 1000.0:.3f}")
    print(f"tokenize_strategy={'batch' if args.batch else 'sequential'}")
    print(f"tokenize_iterations={args.iterations}")
    print(f"texts_per_iteration={len(texts)}")
    print(f"tokenized_texts={total_texts}")
    print(f"total_target_tokens={total_target_tokens}")
    print(f"tokenization_seconds={elapsed:.6f}")
    print(f"texts_per_second={total_texts / elapsed:.3f}")
    print(f"mean_text_us={elapsed * 1_000_000.0 / total_texts:.3f}")


if __name__ == "__main__":
    main()
