"""In-process steady-decode bursts for the EasyMagpie Stage-0 engine.

The stock vLLM EngineCore returns to its ZMQ/busy-loop boundary after every
autoregressive frame.  EasyMagpie needs that frame-by-frame dependency, but it
does *not* need to re-enter the outer process loop while a homogeneous Stage-0
decode batch is already running.  This patch executes a small bounded number
of ordinary ``step_fn`` iterations in the same EngineCore turn.  It deliberately
does not fuse model forwards: every frame still uses vLLM's scheduler, sampler,
KV/Mamba update, EOS handling, and output path.

The first iteration is always the stock EngineCore path and its output is put
on the queue before the burst begins, preserving first-audio latency.
"""
from __future__ import annotations

import os
from typing import Any

from easymagpie_vllm_omni.profiling import nvtx_range
from vllm.logger import init_logger

logger = init_logger(__name__)


def _burst_steps(core: Any) -> int:
    """Return extra steady-state steps, restricted to EasyMagpie Stage-0."""
    model_config = getattr(getattr(core, "vllm_config", None), "model_config", None)
    if int(getattr(model_config, "stage_id", -1) or -1) != 0:
        return 0
    if getattr(model_config, "model_arch", "") not in {
        "EasyMagpieTTS",
        "EasyMagpieTTSForConditionalGeneration",
    }:
        return 0
    try:
        # Four total frames per engine turn is conservative: it lowers
        # scheduler/ZMQ cadence without materially delaying abort processing.
        # Disabled by default until an Nsight-validated configuration proves a
        # throughput gain. Enable explicitly with EASYMAGPIE_STAGE0_BURST_STEPS.
        return max(0, min(int(os.getenv("EASYMAGPIE_STAGE0_BURST_STEPS", "0")), 7))
    except ValueError:
        return 0


def _steady_decode_only(core: Any) -> bool:
    """Do not burst across request admission, prefill, cancellation, or idle."""
    scheduler = getattr(core, "scheduler", None)
    if scheduler is None or not getattr(scheduler, "running", None):
        return False
    if getattr(scheduler, "waiting", None):
        return False
    input_queue = getattr(core, "input_queue", None)
    return input_queue is None or input_queue.empty()


def install_stage0_inprocess_burst() -> bool:
    """Patch EngineCore only once per process, failing closed on API drift."""
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except Exception:
        return False
    if getattr(EngineCoreProc, "_easymagpie_stage0_burst_patched", False):
        return True

    original = EngineCoreProc._process_engine_step

    def _process_engine_step(self: Any) -> bool:
        # Preserve stock handling, including immediate publication of the first
        # frame. This is what keeps TTFA unchanged.
        model_executed = original(self)
        extra_steps = _burst_steps(self)
        if not model_executed or extra_steps <= 0 or not _steady_decode_only(self):
            return model_executed

        for _ in range(extra_steps):
            # Do not hide a newly-arrived request/cancel behind a burst.
            if not _steady_decode_only(self):
                break
            with nvtx_range("EM.stage0.scheduler.inprocess_burst"):
                outputs, executed = self.step_fn()
                for output in outputs.items() if outputs else ():
                    self.output_queue.put_nowait(output)
                self.post_step(executed)
            if not executed:
                break
            model_executed = True
        return model_executed

    EngineCoreProc._process_engine_step = _process_engine_step
    EngineCoreProc._easymagpie_stage0_burst_patched = True
    logger.info("EasyMagpie Stage-0 in-process AR burst hook installed (extra steps=%s).", os.getenv("EASYMAGPIE_STAGE0_BURST_STEPS", "0"))
    return True
