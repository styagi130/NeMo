# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch both EasyMagpie engine stages for the Rust full-pipeline POC.

This process owns configuration and subprocess lifecycle only. Requests and
responses travel directly between Rust and the two StageEngineCoreProc
instances; the engines exchange acoustic chunks through their configured
vLLM-Omni connector.
"""

from __future__ import annotations

import argparse
import signal
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--stage0-handshake", default="tcp://127.0.0.1:62100")
    parser.add_argument("--stage1-handshake", default="tcp://127.0.0.1:62101")
    parser.add_argument("--log-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import vllm_plugin_easymagpie_omni

    vllm_plugin_easymagpie_omni.register()

    from vllm_omni.distributed.omni_connectors.utils.initialization import resolve_omni_kv_config_for_stage
    from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager
    from vllm_omni.engine.stage_init_utils import (
        build_engine_args_dict,
        build_vllm_config,
        get_stage_connector_spec,
        inject_omni_kv_connector_config,
        load_omni_transfer_config_for_model,
        prepare_engine_environment,
    )
    from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

    config_path, stage_configs = load_and_resolve_stage_configs(
        args.model,
        None,
        {},
        deploy_config_path=args.deploy_config,
    )
    stages = {stage.stage_id: stage for stage in stage_configs}
    missing = {0, 1} - stages.keys()
    if missing:
        raise ValueError(f"deploy config is missing stages: {sorted(missing)}")
    if any(stages[index].stage_type == "diffusion" for index in (0, 1)):
        raise ValueError("the Rust EasyMagpie POC requires two LLM stages")

    prepare_engine_environment()
    transfer_config = load_omni_transfer_config_for_model(args.model, config_path)
    handshakes = {
        0: args.stage0_handshake,
        1: args.stage1_handshake,
    }
    managers = []
    for stage_id in (0, 1):
        stage_config = stages[stage_id]
        connector_config = resolve_omni_kv_config_for_stage(transfer_config, stage_id)
        connector_spec = get_stage_connector_spec(
            omni_transfer_config=transfer_config,
            stage_id=stage_id,
            async_chunk=True,
        )
        engine_args = build_engine_args_dict(
            stage_config,
            args.model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=None,
        )
        inject_omni_kv_connector_config(engine_args, connector_config, stage_id)
        vllm_config, executor_class = build_vllm_config(
            stage_config,
            args.model,
            stage_connector_spec=connector_spec,
            engine_args_dict=engine_args,
            headless=True,
        )
        managers.append(
            StageEngineCoreProcManager(
                local_engine_count=1,
                start_index=0,
                local_start_index=0,
                vllm_config=vllm_config,
                local_client=True,
                handshake_address=handshakes[stage_id],
                executor_class=executor_class,
                log_stats=args.log_stats,
                omni_stage_id=stage_id,
            )
        )

    stopping = False

    def stop(_signum: int, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopping:
            if any(manager.finished_procs() for manager in managers):
                raise RuntimeError("an EasyMagpie StageEngineCoreProc exited unexpectedly")
            time.sleep(0.25)
    finally:
        for manager in reversed(managers):
            manager.shutdown()


if __name__ == "__main__":
    main()
