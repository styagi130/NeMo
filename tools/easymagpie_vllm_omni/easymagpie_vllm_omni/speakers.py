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
"""Bounded, optional startup loading for the existing known-voice .pt format."""

import logging
import pickle
import struct
from pathlib import Path

import torch

MAX_PRELOAD_ENTRIES = 64
MAX_PRELOAD_BYTES = 16 * 1024 * 1024
MAX_PRELOAD_FILE_BYTES = 16 * 1024 * 1024
logger = logging.getLogger(__name__)


def load_speaker_embedding(path, *, width=None, dtype=torch.float32, max_bytes=None):
    """Validate and compact a known voice on CPU before any device transfer."""
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        EOFError,
        RuntimeError,
        ValueError,
        IndexError,
        KeyError,
        pickle.UnpicklingError,
        struct.error,
    ) as error:
        raise ValueError(f"Cannot load speaker embedding {path}: {error}") from error
    embedding = loaded.get("speaker_encoding") if isinstance(loaded, dict) else loaded
    if (
        type(embedding) is not torch.Tensor
        or embedding.device.type != "cpu"
        or embedding.layout != torch.strided
        or embedding.is_quantized
        or embedding.is_complex()
        or embedding.ndim != 2
        or not all(embedding.shape)
        or (width is not None and embedding.shape[1] != width)
    ):
        raise ValueError(f"Invalid speaker embedding {path}: expected a real 2-D tensor with width {width}")
    size = embedding.numel() * torch.empty(0, dtype=dtype, device="cpu").element_size()
    if max_bytes is not None and (size > max_bytes or embedding.untyped_storage().nbytes() > MAX_PRELOAD_FILE_BYTES):
        raise ValueError(f"Skipped speaker embedding {path}: startup storage budget exceeded")
    if not torch.isfinite(embedding).all().item():
        raise ValueError(f"Invalid speaker embedding {path}: non-finite values")
    embedding = embedding.detach().to(device="cpu", dtype=dtype, copy=True, memory_format=torch.contiguous_format)
    if not torch.isfinite(embedding).all().item():
        raise ValueError(f"Invalid speaker embedding {path}: values overflow {dtype}")
    return embedding


def speaker_fingerprint(path):
    """Metadata-only check on existing context-cache misses, not on cache hits."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def preload_speakers(model_path, *, width=None, dtype=torch.float32):
    """Yield valid startup voices; skipped/late voices retain lazy request loading."""
    try:
        paths = sorted((Path(model_path) / "speaker_embeddings").glob("*.pt"))[:MAX_PRELOAD_ENTRIES]
    except OSError as error:
        logger.warning("Skipping speaker preload: %s", error)
        return
    remaining = MAX_PRELOAD_BYTES
    for path in paths:
        fingerprint = speaker_fingerprint(path)
        if fingerprint is None or fingerprint[2] > MAX_PRELOAD_FILE_BYTES or remaining == 0:
            continue
        try:
            embedding = load_speaker_embedding(path, width=width, dtype=dtype, max_bytes=remaining)
            if speaker_fingerprint(path) != fingerprint:
                raise ValueError(f"Speaker embedding changed while loading: {path}")
        except ValueError as error:
            logger.warning("Skipping optional speaker preload: %s", error)
            continue
        remaining -= embedding.untyped_storage().nbytes()
        yield path.stem, fingerprint, embedding
