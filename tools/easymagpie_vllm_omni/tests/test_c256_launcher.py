"""Exercise the shell launcher without starting vLLM or compiling native code."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, script, overrides=None):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    python = bin_dir / 'python3'
    python.write_text(
        '#!/bin/bash\n'
        'if [[ "$1" == -c ]]; then\n'
        f'    exec "{sys.executable}" "$@"\n'
        'fi\n'
        'echo "$*" >> "$LAUNCH_CALLS"\n'
    )
    vllm = bin_dir / 'vllm'
    vllm.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'print(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}))\n'
    )
    python.chmod(0o755)
    vllm.chmod(0o755)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('EASYMAGPIE_')}
    environment.update(PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], LAUNCH_CALLS=str(tmp_path / 'calls'))
    environment.update(overrides or {})
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts' / script), '/model with spaces', '31017'],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1]), (tmp_path / 'calls').read_text()


def test_c256_launch_selects_profile_and_builds_before_serving(tmp_path):
    result, calls = launch(tmp_path, 'run_c256_server.sh')
    args, environment = result['args'], result['env']
    assert args[:4] == ['serve', '/model with spaces', '--served-model-name', 'easymagpie']
    assert environment['EASYMAGPIE_DEPLOY_CONFIG'].endswith('/deploy/h100_optimized256.yaml')
    assert environment['EASYMAGPIE_VLLM_CODEC_GRAPH_MAX_ENTRIES'] == '384'
    assert environment['EASYMAGPIE_TALKER_BLOCKING_COPY_EVENT'] == '1'
    assert environment['EASYMAGPIE_PREFILL_GRAPHS'] == '1'
    assert environment['EASYMAGPIE_VLLM_CODEC_BATCHED_PAYLOAD_VALIDITY'] == '1'
    assert len(environment['EASYMAGPIE_PUBLICATION_NAMESPACE']) == 32
    assert '-m easymagpie_vllm_omni.bulk_native' in calls
    assert 'apply_runtime_patches.py --verify' in calls


def test_c256_honors_controlled_overrides(tmp_path):
    result, calls = launch(
        tmp_path,
        'run_c256_server.sh',
        {
            'EASYMAGPIE_DEPLOY_CONFIG': '/custom config.yaml',
            'EASYMAGPIE_BULK_SENDER': '0',
            'EASYMAGPIE_SERVED_MODEL_NAME': 'custom model',
            'EASYMAGPIE_PUBLICATION_NAMESPACE': 'shared-instance',
        },
    )
    assert result['args'][3] == 'custom model'
    assert result['env']['EASYMAGPIE_DEPLOY_CONFIG'] == '/custom config.yaml'
    assert result['env']['EASYMAGPIE_PUBLICATION_NAMESPACE'] == 'shared-instance'
    assert 'bulk_native' not in calls


def test_c128_keeps_original_profile_and_model_name(tmp_path):
    result, calls = launch(tmp_path, 'run_optimized_server.sh')
    assert result['args'][:2] == ['serve', '/model with spaces']
    assert '--served-model-name' not in result['args']
    assert result['env']['EASYMAGPIE_DEPLOY_CONFIG'].endswith('/deploy/h100_optimized128.yaml')
    assert 'EASYMAGPIE_VLLM_CODEC_BATCHED_PAYLOAD_VALIDITY' not in result['env']
    assert 'bulk_native' not in calls
