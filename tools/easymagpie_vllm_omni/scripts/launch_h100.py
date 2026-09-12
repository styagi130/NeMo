#!/usr/bin/env python3
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
"""Supervise the optional single-H100 profile using upstream vLLM-Omni."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Converted model with codec_native/.')
    parser.add_argument('--gpu', required=True, help='One full GPU UUID, not a remapped device index.')
    parser.add_argument('--output', type=Path, required=True, help='New directory for logs, metadata and caches.')
    parser.add_argument('--config', type=Path, default=PACKAGE / 'deploy' / 'easymagpie_h100.yaml')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--api-port', type=_port, default=8091)
    parser.add_argument('--master-port', type=_port, default=29600)
    parser.add_argument('--startup-timeout', type=_positive_seconds, default=1800)
    parser.add_argument('--shutdown-timeout', type=_positive_seconds, default=30)
    parser.add_argument('--private-mps', action='store_true', help='Start and own a private foreground MPS daemon.')
    parser.add_argument('--tuned-kernels', action='store_true', help='Enable compatibility-checked kernel tables.')
    parser.add_argument(
        '--plan-only', action='store_true', help='Print commands without starting or writing anything.'
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu):
        parser.error('--gpu must be a full GPU UUID')
    return args


def build_commands(args):
    timeout = str(math.ceil(args.startup_timeout))
    common = [
        'vllm',
        'serve',
        str(args.model),
        '--omni',
        '--deploy-config',
        str(args.config),
        '--trust-remote-code',
        '--omni-master-address',
        '127.0.0.1',
        '--omni-master-port',
        str(args.master_port),
        '--stage-init-timeout',
        timeout,
        '--init-timeout',
        timeout,
        '--disable-log-stats',
    ]
    head = common + [
        '--stage-id',
        '1',
        '--omni-dp-size-local',
        '1',
        '--omni-lb-policy',
        'round-robin',
        '--stage-overrides',
        '{"0":{"devices":"0,0"}}',
        '--host',
        args.host,
        '--port',
        str(args.api_port),
        '--disable-uvicorn-access-log',
        '--uvicorn-log-level',
        'warning',
    ]
    lm = common + [
        '--headless',
        '--stage-id',
        '0',
        '--omni-dp-size-local',
        '1',
        '--omni-replica-address',
        '127.0.0.1',
        '--stage-overrides',
        '{"0":{"devices":"0"}}',
    ]
    return [('api-codec', '0', head), ('lm0', '1', lm), ('lm1', '1', lm)]


def run(args):
    import yaml

    _validate_args(args)
    args.model, args.config, args.output = args.model.resolve(), args.config.resolve(), args.output.resolve()
    profile = yaml.safe_load(args.config.read_text())
    _validate_profile(profile)
    for path in (args.model / 'config.json', args.model / 'codec_native' / 'config.json'):
        if not path.is_file():
            raise ValueError(f'Missing converted model file: {path}')
    if not args.private_mps:
        _validate_plain_mps(os.environ)
    _check_ports(args)
    tuning = _checked_tuning(args, profile) if args.tuned_kernels else None
    args.output.mkdir(parents=False, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith('CUDA_MPS_')}
    env.pop('VLLM_TUNED_CONFIG_FOLDER', None)
    env.pop('VLLM_LOGGING_CONFIG_PATH', None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, VLLM_PLUGINS='easymagpie_omni', VLLM_DISABLE_SHARED_EXPERTS_STREAM='1')
    env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONUNBUFFERED='1')
    env.update(VLLM_LOGGING_LEVEL='INFO', VLLM_CONFIGURE_LOGGING='1')  # Startup gates use upstream INFO markers.
    for key, directory in (
        ('VLLM_CACHE_ROOT', 'vllm'),
        ('TRITON_CACHE_DIR', 'triton'),
        ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
        ('FLASHINFER_WORKSPACE_BASE', 'flashinfer'),
        ('SPEAKER_SAMPLES_DIR', 'speaker-samples'),
    ):
        path = args.output / directory
        path.mkdir()
        env[key] = str(path)
    if tuning is not None:
        env['VLLM_TUNED_CONFIG_FOLDER'] = str(tuning)
    report = {
        'model': str(args.model),
        'model_config_sha256': _sha256(args.model / 'config.json'),
        'config': str(args.config),
        'config_sha256': _sha256(args.config),
        'gpu': args.gpu,
        'tuned_kernels': str(tuning) if tuning else None,
        'shared_experts_same_stream': True,
    }
    stopped = threading.Event()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, lambda received, frame: _handle_signal(stopped, report, received))
    try:
        context = _private_mps(args, env, stopped, report) if args.private_mps else nullcontext((env, None))
        with context as (client_env, daemon):
            _supervise(args, client_env, stopped, report, daemon)
    except InterruptedError:
        return 128 + int(report.get('signal', 0))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return 1 if report.get('cleanup', {}).get('forced') else 0


def main():
    args = parse_args()
    _validate_args(args)
    if args.plan_only:
        print(json.dumps(build_commands(args), indent=2))
        return 0
    return run(args)


def _port(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError('Port must be in [1, 65535]')
    return number


def _positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Timeout must be finite and positive')
    return number


def _validate_args(args):
    if args.api_port == args.master_port:
        raise ValueError('API and master ports must differ')
    if not args.host:
        raise ValueError('HTTP host must not be empty')


def _check_ports(args):
    for host, port in ((args.host, args.api_port), ('127.0.0.1', args.master_port)):
        try:
            for family, kind, protocol, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
                with socket.socket(family, kind, protocol) as probe:
                    probe.bind(address)
        except OSError as error:
            raise ValueError(f'Bind address is unavailable: {host}:{port}') from error


def _validate_profile(profile):
    if not isinstance(profile, dict) or profile.get('pipeline') != 'easymagpie':
        raise ValueError('Expected an EasyMagpie deployment profile')
    stages = profile.get('stages', [])
    if [stage.get('stage_id') for stage in stages] != [0, 1]:
        raise ValueError('Expected exactly Stage0 and Stage1')
    for stage, replicas, devices, worker in zip(
        stages,
        (2, 1),
        (('0', '0,0'), ('0',)),
        ('EasyMagpieGPUARWorker', 'EasyMagpieCodecGPUGenerationWorker'),
        strict=True,
    ):
        if (
            stage.get('num_replicas') != replicas
            or str(stage.get('devices')) not in devices
            or stage.get('engine_extras', {}).get('worker_cls') != f'easymagpie_vllm_omni.runner.{worker}'
        ):
            raise ValueError('Profile must contain two same-GPU LM replicas and the complete codec worker')
    codec = stages[1]
    if codec.get('engine_extras', {}).get('dtype', codec.get('dtype')) != 'float32':
        raise ValueError('The codec must remain float32')


def _validate_tuning(manifest, config, profile, gpu_name, versions):
    lm = profile['stages'][0]
    if gpu_name != manifest['gpu_name'] or versions != manifest['versions']:
        raise ValueError('Tuning tables do not match the GPU or installed runtime versions')
    if any(config.get(key) != value for key, value in manifest['model_shape'].items()):
        raise ValueError('Tuning tables do not match the model dimensions')
    extra = lm.get('engine_extras', {})
    containers = (profile, lm, extra)
    if any(container.get('hf_overrides') for container in containers):
        raise ValueError('Tuning tables require the unmodified model configuration')
    if (
        config.get('quantization_config') is not None
        or any(container.get('quantization') is not None for container in containers)
        or extra.get('dtype', lm.get('dtype', profile.get('dtype'))) != 'float16'
        or extra.get('mamba_ssm_cache_dtype', lm.get('mamba_ssm_cache_dtype')) != 'float32'
        or any(container.get('tensor_parallel_size', 1) != 1 for container in containers)
    ):
        raise ValueError('Tuning tables require unquantized FP16, FP32 Mamba cache and TP=1')


def _checked_tuning(args, profile):
    directory = PACKAGE / 'deploy' / 'h100'
    manifest = json.loads((directory / 'manifest.json').read_text())
    versions = {name: importlib.metadata.version(name) for name in manifest['versions']}
    gpu_name = subprocess.check_output(
        ['nvidia-smi', '--id', args.gpu, '--query-gpu=name', '--format=csv,noheader'], text=True
    ).strip()
    config = json.loads((args.model / 'config.json').read_text())
    _validate_tuning(manifest, config, profile, gpu_name, versions)
    distribution = importlib.metadata.distribution('vllm')
    files = [(directory / 'kernels' / name, digest) for name, digest in manifest['tables'].items()]
    files.extend((distribution.locate_file(name), digest) for name, digest in manifest['vllm_sources'].items())
    for path, expected in files:
        if _sha256(path) != expected:
            raise ValueError(f'Tuning provenance mismatch: {path}')
    return directory / 'kernels'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _handle_signal(stopped, report, signum):
    report['signal'] = signum
    stopped.set()


def _validate_plain_mps(env):
    if any(key.startswith('CUDA_MPS_') for key in env) or Path('/tmp/nvidia-mps/nvidia-cuda-mps-control.pid').exists():
        raise ValueError(
            'Plain launch refuses pre-existing MPS settings/state; use a clean environment or --private-mps'
        )


def _check_children(processes):
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(f'Owned process {process.pid} exited: {process.returncode}')


def _wait_until(ready, processes, stopped, timeout, label):
    deadline = time.monotonic() + timeout
    while True:
        if stopped.is_set():
            raise InterruptedError('Shutdown requested')
        _check_children(processes)
        if ready():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Startup timed out: {label}')
        stopped.wait(0.1)


def _group_alive(process):
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    try:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                return True
    except (OSError, ValueError, IndexError):
        return True  # Unreadable process state is not proof of a drained group.
    return False


def _stop_processes(processes, grace):
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    while True:
        for process in processes:
            process.poll()
        alive = [process for process in processes if _group_alive(process)]
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    forced = False
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for process in alive:
            try:
                os.killpg(process.pid, signum)
                forced |= signum == signal.SIGKILL
            except ProcessLookupError:
                pass
        if alive and signum == signal.SIGTERM:
            time.sleep(0.1)
            for process in alive:
                process.poll()
            alive = [process for process in alive if _group_alive(process)]
    for process in processes:
        process.wait(timeout=5)
    return {'forced': forced, 'exit_codes': {str(process.pid): process.returncode for process in processes}}


@contextmanager
def _private_mps(args, base_env, stopped, report):
    private = Path(tempfile.mkdtemp(prefix='easymagpie-mps-'))
    if private.stat().st_uid != os.getuid():
        raise RuntimeError('Private MPS directory must belong to this user')
    for name in ('pipe', 'log'):
        (private / name).mkdir(mode=0o700)
    env = dict(base_env, CUDA_MPS_PIPE_DIRECTORY=str(private / 'pipe'), CUDA_MPS_LOG_DIRECTORY=str(private / 'log'))
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    report['mps'] = {'directory': str(private)}
    with (args.output / 'mps.log').open('x') as log:
        daemon = subprocess.Popen(
            ['nvidia-cuda-mps-control', '-f'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        report['mps']['pid'] = daemon.pid
        try:
            pid_file = private / 'pipe' / 'nvidia-cuda-mps-control.pid'
            _wait_until(
                lambda: pid_file.is_file() and pid_file.read_text().strip() == str(daemon.pid),
                [daemon],
                stopped,
                min(args.startup_timeout, 30),
                'private MPS',
            )
            clients = dict(env)
            clients.pop('CUDA_VISIBLE_DEVICES', None)  # The private daemon exposes its selected GPU as device0.
            yield clients, daemon
        finally:
            report['mps']['quit_succeeded'] = False
            try:
                subprocess.run(
                    ['nvidia-cuda-mps-control'],
                    input='quit\n',
                    text=True,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=True,
                )
                if daemon.wait(timeout=10) != 0:
                    raise RuntimeError('Private MPS daemon exited unsuccessfully')
                report['mps']['quit_succeeded'] = True
            finally:
                report['mps']['cleanup'] = _stop_processes([daemon], 0)


def _supervise(args, env, stopped, report, daemon):
    processes = []
    report['launches'] = []
    deadline = time.monotonic() + args.startup_timeout
    with ExitStack() as logs:
        try:
            for index, (name, priority, command) in enumerate(build_commands(args)):
                if stopped.is_set():
                    raise InterruptedError('Shutdown requested')
                log = logs.enter_context((args.output / f'{name}.log').open('x'))
                process = subprocess.Popen(
                    command,
                    env=dict(env, CUDA_MPS_CLIENT_PRIORITY=priority),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(process)
                report['launches'].append(
                    {'name': name, 'pid': process.pid, 'priority_before_exec': priority, 'argv': command}
                )
                (args.output / 'launches.json').write_text(json.dumps(report['launches'], indent=2) + '\n')
                marker = (
                    '[StageRuntime] Stage 1 initialized'
                    if index == 0
                    else f'Remote LLM replica attached stage=0 replica={index - 1}'
                )
                _wait_until(
                    lambda: marker in (args.output / 'api-codec.log').read_text(errors='replace'),
                    processes + ([daemon] if daemon else []),
                    stopped,
                    max(0, deadline - time.monotonic()),
                    name,
                )
            watched = processes + ([daemon] if daemon else [])
            _wait_until(lambda: _healthy(args), watched, stopped, max(0, deadline - time.monotonic()), 'HTTP health')
            (args.output / 'ready.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'EasyMagpie ready; deployment record: {args.output}', flush=True)
            while not stopped.wait(1):
                _check_children(watched)
            raise InterruptedError('Shutdown requested')
        finally:
            report['cleanup'] = _stop_processes(processes, args.shutdown_timeout)


def _healthy(args):
    host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(args.host, args.host)
    if ':' in host and not host.startswith('['):
        host = f'[{host}]'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://{host}:{args.api_port}/health', timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


if __name__ == '__main__':
    raise SystemExit(main())
