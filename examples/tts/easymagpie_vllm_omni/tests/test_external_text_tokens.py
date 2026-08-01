# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for token IDs supplied by the native Rust frontend."""

from __future__ import annotations

import pytest
import torch

from easymagpie_vllm_omni.easymagpie import _coerce_external_token_ids


def test_external_context_token_ids_accept_lists_and_tensors() -> None:
    assert _coerce_external_token_ids([11, 12], "context_token_ids") == [11, 12]
    assert _coerce_external_token_ids(
        torch.tensor([[21, 22]], dtype=torch.int64), "context_token_ids"
    ) == [21, 22]


def test_external_context_token_ids_preserve_explicit_empty_list() -> None:
    assert _coerce_external_token_ids([], "context_token_ids") == []
    assert _coerce_external_token_ids(None, "context_token_ids") is None


def test_external_context_token_ids_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="list of integer token IDs"):
        _coerce_external_token_ids("11,12", "context_token_ids")
    with pytest.raises(ValueError, match="negative"):
        _coerce_external_token_ids([11, -1], "context_token_ids")
