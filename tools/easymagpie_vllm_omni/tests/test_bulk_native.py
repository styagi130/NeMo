# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build portability and native chunk ownership, tails and generation reuse."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from easymagpie_vllm_omni import bulk_native
from easymagpie_vllm_omni.bulk_sender import PythonChunkAccumulator


def test_cache_build_is_atomic_and_reused(tmp_path, monkeypatch):
    library = bulk_native.build_native_library(tmp_path)
    assert library.is_file() and library.parent.parent == tmp_path
    monkeypatch.setattr(bulk_native.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected rebuild"))
    assert bulk_native.build_native_library(tmp_path) == library
    assert not list(tmp_path.rglob("build-*"))
    assert not Path(bulk_native.__file__).with_name("libchunk_builder.so").exists()


def test_compiler_failure_leaves_no_partial_library(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bulk_native.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="injected compiler failure"),
    )
    with pytest.raises(RuntimeError, match="injected compiler failure"):
        bulk_native.build_native_library(tmp_path)
    assert not list(tmp_path.rglob("*.so")) and not list(tmp_path.rglob("build-*"))


def test_cxx_uses_argument_vector_not_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("CXX", "c++ '-DSTRING=literal;$(not-a-command)'")
    observed = []

    def compile_command(command, **kwargs):
        observed.append((command, kwargs))
        Path(command[-1]).write_bytes(b"test library")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(bulk_native.subprocess, "run", compile_command)
    bulk_native.build_native_library(tmp_path)
    command, kwargs = observed[0]
    assert command[1] == "-DSTRING=literal;$(not-a-command)"
    assert not kwargs.get("shell", False)


def test_native_matches_reference_and_keeps_returned_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYMAGPIE_NATIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EASYMAGPIE_BULK_FRAME_VIEWLESS", "1")
    native, reference = bulk_native.BulkChunkAccumulator(2), PythonChunkAccumulator(2)
    retained = []
    try:
        for builder in (native, reference):
            builder.reset(0, 1, 0)
        for index in range(19):
            row = torch.arange(16).reshape(1, 16) + index
            flags = bulk_native.HAS_FRAME | (bulk_native.TERMINAL if index == 18 else 0)
            actual = native.append([row], [0], [1], [flags])
            expected = reference.append([row], [0], [1], [flags])
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                assert left.kind == right.kind and left.frames == right.frames
                assert torch.equal(left.codes, right.codes)
                retained.append((left.codes, left.codes.clone()))
        assert [tensor.shape[0] for tensor, _ in retained] == [2, 6, 8, 8]
        native.release(0, 1)
        native.reset(0, 2, 0)
        stale = native.append([None], [0], [1], [bulk_native.TERMINAL])
        assert stale[0].kind == "stale"
        native.append(
            [torch.ones((1, 16), dtype=torch.long)], [0], [2], [bulk_native.HAS_FRAME | bulk_native.TERMINAL]
        )
        for tensor, saved in retained:
            assert torch.equal(tensor, saved)
    finally:
        native.close()
