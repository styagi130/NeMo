# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The API's optional switch interval does not leak into forked workers."""
import sys
from types import SimpleNamespace

import pytest

from easymagpie_vllm_omni import api_switch


@pytest.fixture(autouse=True)
def isolated_switch(monkeypatch):
    before = sys.getswitchinterval()
    monkeypatch.setattr(api_switch, "_STATE", None)
    monkeypatch.setattr(api_switch, "_FORK_REGISTERED", False)
    monkeypatch.setattr(api_switch.os, "register_at_fork", lambda **kwargs: None)
    yield
    sys.setswitchinterval(before)


def test_default_off_and_invalid_never_change_interval(monkeypatch):
    before = sys.getswitchinterval()
    monkeypatch.delenv("EASYMAGPIE_API_SWITCH_INTERVAL_MS", raising=False)
    assert not api_switch.configure_api_switch(SimpleNamespace())["enabled"]
    monkeypatch.setenv("EASYMAGPIE_API_SWITCH_INTERVAL_MS", "2")
    with pytest.raises(ValueError):
        api_switch.configure_api_switch(SimpleNamespace())
    assert sys.getswitchinterval() == before


def test_api_configuration_is_once_and_fork_restores_original(monkeypatch):
    before = sys.getswitchinterval()
    monkeypatch.setenv("EASYMAGPIE_API_SWITCH_INTERVAL_MS", "1")
    with pytest.raises(RuntimeError, match="engine context"):
        api_switch.configure_api_switch(SimpleNamespace())
    context = SimpleNamespace(engine_client=object())
    first = api_switch.configure_api_switch(context)
    assert api_switch.configure_api_switch(context) == first
    assert first["configuration"]["configuration_count"] == 1
    assert sys.getswitchinterval() == 0.001
    parent = api_switch.os.getpid()
    monkeypatch.setattr(api_switch.os, "getpid", lambda: parent + 1)
    api_switch._after_fork_child()
    assert not api_switch.api_switch_status()["enabled"]
    assert sys.getswitchinterval() == before
