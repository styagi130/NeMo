# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Let the EasyMagpie membership watcher finish before orchestrator cleanup."""

from functools import wraps


def install_membership_shutdown() -> None:
    """Keep the pinned upstream cleanup order without its watcher wait cycle."""
    from vllm_omni.engine.orchestrator import Orchestrator

    original = Orchestrator._request_handler
    if getattr(original, "_easymagpie_shutdown", False) is True:
        return

    @wraps(original)
    async def request_handler(self):
        try:
            return await original(self)
        finally:
            if self._shutdown_event.is_set() and self._membership is not None and _is_easymagpie(self.stage_pools):
                # Stop only the watcher. run() still drains pending membership
                # tasks before closing the hub and shutting down stage clients.
                self._membership._shutdown_event.set()

    request_handler._easymagpie_shutdown = True
    Orchestrator._request_handler = request_handler


def _is_easymagpie(pools):
    for pool in pools:
        model_config = getattr(pool.stage_vllm_config, "model_config", None)
        config = getattr(model_config, "hf_config", None)
        if getattr(config, "model_type", None) == "easymagpie_codec" or {
            "EasyMagpieTTS",
            "EasyMagpieTTSForConditionalGeneration",
        }.intersection(getattr(config, "architectures", None) or ()):
            return True
    return False
