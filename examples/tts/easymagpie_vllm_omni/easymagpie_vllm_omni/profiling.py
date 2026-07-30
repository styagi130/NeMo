"""Low-overhead, opt-in NVTX ranges for EasyMagpie serving diagnostics."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
_cuda_profile_lock = threading.Lock()
_cuda_profile_started = False
_cuda_profile_stopped = False


def _enabled() -> bool:
    return os.environ.get("EASYMAGPIE_NVTX", "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range only when an explicit diagnostic capture enables it."""
    if not _enabled():
        yield
        return
    try:
        import torch

        torch.cuda.nvtx.range_push(name)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_profiled(name: str) -> Callable[[_F], _F]:
    """Decorate a synchronous serving boundary with :func:`nvtx_range`."""
    def decorate(fn: _F) -> _F:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with nvtx_range(name):
                return fn(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorate


def maybe_start_cuda_profile() -> bool:
    """Start one Nsight capture when an explicit arm file appears.

    This keeps cold initialization and warm-up outside the trace. A background
    watcher stops the capture from the same CUDA process after the controller
    creates the configured stop file.
    """
    global _cuda_profile_started, _cuda_profile_stopped
    arm_file = os.environ.get("EASYMAGPIE_CUDA_PROFILE_ARM_FILE")
    stop_file = os.environ.get("EASYMAGPIE_CUDA_PROFILE_STOP_FILE")
    if not arm_file or not stop_file or _cuda_profile_started or not os.path.exists(arm_file):
        return False
    with _cuda_profile_lock:
        if _cuda_profile_started or not os.path.exists(arm_file):
            return False
        import torch

        device = torch.cuda.current_device()
        start_result = torch.cuda.cudart().cudaProfilerStart()
        if start_result != 0:
            raise RuntimeError(f"cudaProfilerStart failed with CUDA error {start_result}")
        _cuda_profile_started = True
        _cuda_profile_stopped = False
        try:
            os.unlink(arm_file)
        except OSError:
            pass

        def _stop_watcher() -> None:
            global _cuda_profile_stopped
            while not os.path.exists(stop_file):
                time.sleep(0.01)
            with _cuda_profile_lock:
                if _cuda_profile_stopped:
                    return
                # CUDA's current device is thread-local. Restore the Stage-0
                # device before stopping from this background watcher.
                torch.cuda.set_device(device)
                stop_result = torch.cuda.cudart().cudaProfilerStop()
                if stop_result != 0:
                    raise RuntimeError(f"cudaProfilerStop failed with CUDA error {stop_result}")
                _cuda_profile_stopped = True

        threading.Thread(target=_stop_watcher, name="easymagpie_cuda_profile_stop", daemon=True).start()
        return True
