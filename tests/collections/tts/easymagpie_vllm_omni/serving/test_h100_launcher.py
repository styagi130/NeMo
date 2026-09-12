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
"""Standalone, standard-library tests for H100 process and MPS ownership."""

import ctypes
import importlib.util
import io
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

PACKAGE = Path(__file__).resolve().parents[5] / 'tools' / 'easymagpie_vllm_omni'
SPEC = importlib.util.spec_from_file_location('launch_h100', PACKAGE / 'scripts' / 'launch_h100.py')
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)
GPU = 'GPU-00000000-0000-0000-0000-000000000001'


def arguments(**kwargs):
    args = launch.parse_args(['--model', '/models/tts', '--gpu', GPU, '--output', '/tmp/test-output'])
    vars(args).update(kwargs)
    return args


def profile():
    return {
        'pipeline': 'easymagpie',
        'dtype': 'float16',
        'stages': [
            {
                'stage_id': 0,
                'num_replicas': 2,
                'devices': '0,0',
                'mamba_ssm_cache_dtype': 'float32',
                'engine_extras': {'worker_cls': 'easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker'},
            },
            {
                'stage_id': 1,
                'num_replicas': 1,
                'devices': '0',
                'dtype': 'float32',
                'engine_extras': {'worker_cls': 'easymagpie_vllm_omni.runner.EasyMagpieCodecGPUGenerationWorker'},
            },
        ],
    }


class LauncherTests(unittest.TestCase):
    def test_commands_use_upstream_api_and_two_headless_lms(self):
        commands = launch.build_commands(arguments())
        self.assertEqual(
            [(name, priority) for name, priority, _ in commands], [('api-codec', '0'), ('lm0', '1'), ('lm1', '1')]
        )
        for name, _, command in commands:
            self.assertEqual(command[:3], ['vllm', 'serve', '/models/tts'])
            self.assertIn('--omni', command)
            self.assertEqual('--headless' in command, name != 'api-codec')
            self.assertNotIn('nemotron-speech', ' '.join(command))
        self.assertTrue(arguments().config.name == 'easymagpie_h100.yaml')

    def test_commands_place_both_remote_lms_on_selected_gpu(self):
        for name, _, command in launch.build_commands(arguments()):
            overrides = json.loads(command[command.index('--stage-overrides') + 1])
            self.assertEqual(overrides, {'0': {'devices': '0,0' if name == 'api-codec' else '0'}})
            self.assertEqual(command[command.index('--omni-dp-size-local') + 1], '1')

    def test_profile_accepts_legacy_single_device_template(self):
        candidate = profile()
        candidate['stages'][0]['devices'] = '0'
        launch._validate_profile(candidate)
        self.assertEqual(candidate['stages'][0]['devices'], '0')

    def test_profile_rejects_other_device_placements(self):
        for stage, devices in ((0, '0,1'), (0, '1'), (0, ''), (1, '0,0'), (1, '1')):
            with self.subTest(stage=stage, devices=devices), self.assertRaises(ValueError):
                candidate = profile()
                candidate['stages'][stage]['devices'] = devices
                launch._validate_profile(candidate)

    def test_invalid_arguments_fail_before_start(self):
        for extra in (['--gpu', '0'], ['--api-port', '0'], ['--startup-timeout', 'nan'], ['--shutdown-timeout', '-1']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.parse_args(['--model', '/models/tts', '--gpu', GPU, '--output', '/tmp/results', *extra])
        with self.assertRaises(ValueError):
            launch._validate_args(arguments(api_port=8091, master_port=8091))

    def test_upstream_timeouts_are_integer_arguments(self):
        command = launch.build_commands(arguments(startup_timeout=2.5))[0][2]
        self.assertEqual(command[command.index('--stage-init-timeout') + 1], '3')
        self.assertEqual(command[command.index('--init-timeout') + 1], '3')

    def test_occupied_api_or_master_port_is_rejected(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            busy = occupied.getsockname()[1]
            with socket.socket() as unused:
                unused.bind(('127.0.0.1', 0))
                free = unused.getsockname()[1]
            for api, master in ((busy, free), (free, busy)):
                with self.subTest(api=api, master=master), self.assertRaises(ValueError):
                    launch._check_ports(arguments(api_port=api, master_port=master))
        launch._check_ports(arguments(api_port=busy, master_port=free))

    def test_plan_only_does_not_probe_network_or_launch(self):
        with (
            patch.object(launch, 'parse_args', return_value=arguments(plan_only=True)),
            patch.object(launch, '_check_ports') as check,
            patch.object(launch, 'run') as run,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(launch.main(), 0)
        check.assert_not_called()
        run.assert_not_called()

    def test_profile_rejects_wrong_topology_or_missing_worker(self):
        launch._validate_profile(profile())
        bad = profile()
        bad['stages'][0]['num_replicas'] = 1
        with self.assertRaises(ValueError):
            launch._validate_profile(bad)
        bad = profile()
        bad['stages'][1]['engine_extras'] = {}
        with self.assertRaises(ValueError):
            launch._validate_profile(bad)
        bad = profile()
        bad['stages'][1]['engine_extras']['dtype'] = 'float16'
        with self.assertRaises(ValueError):
            launch._validate_profile(bad)

    def test_tuning_compatibility_is_explicit_and_exact(self):
        manifest = json.loads((PACKAGE / 'deploy' / 'h100' / 'manifest.json').read_text())
        config = manifest['model_shape']
        launch._validate_tuning(manifest, config, profile(), manifest['gpu_name'], manifest['versions'])
        for changed in ({**config, 'n_routed_experts': 32}, {**config, 'quantization_config': {'method': 'fp8'}}):
            with self.assertRaises(ValueError):
                launch._validate_tuning(manifest, changed, profile(), manifest['gpu_name'], manifest['versions'])
        with self.assertRaises(ValueError):
            launch._validate_tuning(manifest, config, profile(), 'NVIDIA A100', manifest['versions'])
        with self.assertRaises(ValueError):
            launch._validate_tuning(
                manifest, config, profile(), manifest['gpu_name'], {**manifest['versions'], 'triton': '0'}
            )
        bad = profile()
        bad['stages'][0]['engine_extras']['mamba_ssm_cache_dtype'] = 'float16'
        with self.assertRaises(ValueError):
            launch._validate_tuning(manifest, config, bad, manifest['gpu_name'], manifest['versions'])

    def test_tuning_files_retain_original_bytes(self):
        directory = PACKAGE / 'deploy' / 'h100'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(len(manifest['tables']), 2)
        for filename, digest in manifest['tables'].items():
            self.assertEqual(launch._sha256(directory / 'kernels' / filename), digest)
        self.assertEqual(manifest['mamba_measured_effective_batches'], [64, 128, 256, 512, 1024])
        self.assertEqual(manifest['mamba_table_entries'], 16)

    def test_tuning_rejects_model_and_global_runtime_overrides(self):
        manifest = json.loads((PACKAGE / 'deploy' / 'h100' / 'manifest.json').read_text())
        for location, key, value in (
            ('extra', 'hf_overrides', {'n_routed_experts': 32}),
            ('stage', 'hf_overrides', {'hidden_size': 2048}),
            ('global', 'quantization', 'fp8'),
            ('global', 'tensor_parallel_size', 2),
        ):
            with self.subTest(location=location, key=key):
                candidate = profile()
                container = candidate if location == 'global' else candidate['stages'][0]
                if location == 'extra':
                    container = container['engine_extras']
                container[key] = value
                with self.assertRaises(ValueError):
                    launch._validate_tuning(
                        manifest, manifest['model_shape'], candidate, manifest['gpu_name'], manifest['versions']
                    )

    def test_plain_mode_refuses_preexisting_mps_state(self):
        with patch.object(launch.Path, 'exists', return_value=False):
            launch._validate_plain_mps({})
            with self.assertRaises(ValueError):
                launch._validate_plain_mps({'CUDA_MPS_PIPE_DIRECTORY': '/another/deployment'})
        with patch.object(launch.Path, 'exists', return_value=True), self.assertRaises(ValueError):
            launch._validate_plain_mps({})

    def test_run_enforces_logging_needed_by_startup_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codec = root / 'model' / 'codec_native'
            codec.mkdir(parents=True)
            for path in (codec / 'config.json', codec.parent / 'config.json', root / 'profile.yaml'):
                path.write_text('{}')
            args = arguments(model=codec.parent, config=root / 'profile.yaml', output=root / 'run')
            received = {}

            def stop(args, env, stopped, report, daemon):
                received.update(env)
                launch._handle_signal(stopped, report, signal.SIGTERM)
                raise InterruptedError

            with (
                patch.dict(sys.modules, {'yaml': SimpleNamespace(safe_load=lambda text: profile())}),
                patch.dict(os.environ, {'VLLM_LOGGING_LEVEL': 'WARNING', 'VLLM_LOGGING_CONFIG_PATH': '/bad/config'}),
                patch.object(launch, '_supervise', side_effect=stop),
                patch.object(launch, '_validate_plain_mps'),
                patch.object(launch, '_check_ports'),
            ):
                self.assertEqual(launch.run(args), 128 + signal.SIGTERM)
            self.assertEqual(received['VLLM_LOGGING_LEVEL'], 'INFO')
            self.assertEqual(received['VLLM_CONFIGURE_LOGGING'], '1')
            self.assertNotIn('VLLM_LOGGING_CONFIG_PATH', received)

    def test_health_uses_local_ipv4_ipv6_and_no_proxy(self):
        for host, expected in (('0.0.0.0', '127.0.0.1'), ('::', '[::1]'), ('::1', '[::1]')):
            with (
                self.subTest(host=host),
                patch.object(launch.urllib.request, '_opener', None),
                patch.object(launch.urllib.request, 'build_opener') as build,
            ):
                build.return_value.open.return_value.__enter__.return_value.status = 200
                self.assertTrue(launch._healthy(arguments(host=host)))
                self.assertEqual(build.return_value.open.call_args.args[0], f'http://{expected}:8091/health')
                self.assertEqual(build.call_args.args[0].proxies, {})

    def test_stop_only_owned_process_group(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(launch, '_group_alive', return_value=True), patch.object(launch.os, 'killpg') as kill:
            result = launch._stop_processes([process], 0)
        process.terminate.assert_called_once_with()
        self.assertTrue(result['forced'])
        self.assertTrue(all(call.args[0] == process.pid for call in kill.call_args_list))
        self.assertEqual(kill.call_args_list[-1].args[1], signal.SIGKILL)

    def test_wait_is_interrupted_before_readiness(self):
        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(InterruptedError):
            launch._wait_until(lambda: True, [], stopped, 1, 'test')

    def test_dead_child_fails_startup(self):
        child = Mock(pid=123)
        child.poll.return_value = 2
        with self.assertRaises(RuntimeError):
            launch._wait_until(lambda: True, [child], threading.Event(), 1, 'test')

    def test_signal_sets_stop_and_exit_status(self):
        stopped, report = threading.Event(), {}
        launch._handle_signal(stopped, report, signal.SIGTERM)
        self.assertTrue(stopped.is_set())
        self.assertEqual(report['signal'], signal.SIGTERM)

    def test_lm_launch_failure_cleans_up_already_started_codec(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(output=Path(temporary))
            child = Mock(pid=111)
            with (
                patch.object(launch.subprocess, 'Popen', side_effect=[child, OSError('LM launch failed')]) as popen,
                patch.object(launch, '_wait_until'),
                patch.object(launch, '_stop_processes', return_value={'forced': False}) as stop,
            ):
                with self.assertRaises(OSError):
                    launch._supervise(args, {}, threading.Event(), {}, None)
            stop.assert_called_once_with([child], args.shutdown_timeout)
            self.assertEqual(popen.call_args_list[0].kwargs['env']['CUDA_MPS_CLIENT_PRIORITY'], '0')
            self.assertEqual(popen.call_args_list[1].kwargs['env']['CUDA_MPS_CLIENT_PRIORITY'], '1')

    def test_all_priorities_are_set_before_exec_and_stop_cleans_all_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(output=Path(temporary))
            stopped = threading.Event()
            children = [Mock(pid=index) for index in (111, 112, 113)]
            waits = []

            def ready(*args):
                waits.append(args)
                if len(waits) == 4:
                    stopped.set()

            with (
                patch.object(launch.subprocess, 'Popen', side_effect=children) as popen,
                patch.object(launch, '_wait_until', side_effect=ready),
                patch.object(launch, '_stop_processes', return_value={'forced': False}) as stop,
            ):
                with self.assertRaises(InterruptedError):
                    launch._supervise(args, {}, stopped, {}, None)
            self.assertEqual(
                [call.kwargs['env']['CUDA_MPS_CLIENT_PRIORITY'] for call in popen.call_args_list], ['0', '1', '1']
            )
            self.assertTrue(all(call.kwargs['start_new_session'] for call in popen.call_args_list))
            stop.assert_called_once_with(children, args.shutdown_timeout)

    def test_private_mps_has_owned_foreground_daemon_and_private_quit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            private = output / 'mps'
            private.mkdir()
            daemon = Mock(pid=23456)
            daemon.poll.return_value = None
            daemon.wait.return_value = 0
            report = {}
            env = {'CUDA_MPS_PIPE_DIRECTORY': '/do-not-touch/production', 'CUDA_VISIBLE_DEVICES': 'other'}
            with (
                patch.object(launch.tempfile, 'mkdtemp', return_value=str(private)),
                patch.object(launch.subprocess, 'Popen', return_value=daemon) as popen,
                patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as control,
                patch.object(launch, '_wait_until'),
                patch.object(launch, '_stop_processes', return_value={'forced': False}),
            ):
                with launch._private_mps(arguments(output=output), env, threading.Event(), report) as result:
                    client_env, child = result
                    self.assertIs(child, daemon)
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', client_env)
                    self.assertEqual(client_env['CUDA_MPS_PIPE_DIRECTORY'], str(private / 'pipe'))
                self.assertEqual(popen.call_args.args[0], ['nvidia-cuda-mps-control', '-f'])
                self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], GPU)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertEqual(control.call_args.kwargs['input'], 'quit\n')
                self.assertEqual(control.call_args.kwargs['env']['CUDA_MPS_PIPE_DIRECTORY'], str(private / 'pipe'))
            self.assertEqual(env['CUDA_MPS_PIPE_DIRECTORY'], '/do-not-touch/production')

    def test_failed_mps_start_still_quits_only_owned_daemon(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            private = output / 'mps'
            private.mkdir()
            daemon = Mock(pid=23456)
            daemon.poll.return_value = None
            daemon.wait.return_value = 0
            with (
                patch.object(launch.tempfile, 'mkdtemp', return_value=str(private)),
                patch.object(launch.subprocess, 'Popen', return_value=daemon),
                patch.object(launch.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as control,
                patch.object(launch, '_wait_until', side_effect=TimeoutError('startup')),
                patch.object(launch, '_stop_processes', return_value={'forced': False}) as stop,
            ):
                with self.assertRaises(TimeoutError):
                    with launch._private_mps(arguments(output=output), {}, threading.Event(), {}):
                        self.fail('Failed startup must not yield a client environment')
                control.assert_called_once()
                stop.assert_called_once_with([daemon], 0)

    def test_failed_private_quit_is_not_reported_as_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            private = output / 'mps'
            private.mkdir()
            daemon, report = Mock(pid=23456), {}
            with (
                patch.object(launch.tempfile, 'mkdtemp', return_value=str(private)),
                patch.object(launch.subprocess, 'Popen', return_value=daemon),
                patch.object(launch.subprocess, 'run', side_effect=subprocess.TimeoutExpired('quit', 10)),
                patch.object(launch, '_wait_until'),
                patch.object(launch, '_stop_processes', return_value={'forced': True}) as stop,
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    with launch._private_mps(arguments(output=output), {}, threading.Event(), report):
                        pass
            self.assertFalse(report['mps']['quit_succeeded'])
            self.assertTrue(report['mps']['cleanup']['forced'])
            stop.assert_called_once_with([daemon], 0)

    def test_natural_child_shutdown_is_reaped(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            result = launch._stop_processes([child], 1)
            self.assertFalse(result['forced'])
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process groups and child subreaper required')
    def test_exited_leader_does_not_hide_live_owned_child(self):
        with _orphaned_group(zombie=False) as (leader, child_pid, unrelated):
            self.assertEqual(leader.returncode, 0)
            self.assertNotEqual(_process_state(child_pid)[0], 'Z')
            with patch.object(launch.os, 'killpg', wraps=os.killpg) as kill:
                launch._stop_processes([leader], 0)
            self.assertEqual(_process_state(child_pid)[0], 'Z')
            self.assertIsNone(unrelated.poll())
            self.assertTrue(any(call.args == (leader.pid, signal.SIGTERM) for call in kill.call_args_list))
            self.assertTrue(all(call.args[0] == leader.pid for call in kill.call_args_list))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process groups and child subreaper required')
    def test_zombie_only_owned_group_does_not_report_forced_cleanup(self):
        with _orphaned_group(zombie=True) as (leader, child_pid, unrelated):
            self.assertEqual(_process_state(child_pid)[0], 'Z')
            with patch.object(launch.os, 'killpg', wraps=os.killpg) as kill:
                result = launch._stop_processes([leader], 0)
            self.assertIsNone(unrelated.poll())
            self.assertTrue(all(call.args[0] == leader.pid for call in kill.call_args_list))
            self.assertFalse(result['forced'], 'A zombie cannot be killed and is not a live worker')
            self.assertFalse(any(call.args[1] != 0 for call in kill.call_args_list))

    def test_group_liveness_matches_exact_pgid_and_final_parenthesis(self):
        for state in ('Z', 'X', 'S'):
            with self.subTest(state=state):
                entries = [
                    _proc_entry('self', PermissionError()),
                    _proc_entry(51, '51 (foreign) S 1 54321 54321'),
                    _proc_entry(52, f'52 (worker (with spaces)) {state} 1 12345 12345'),
                ]
                with patch.object(launch, 'Path') as proc, patch.object(launch.os, 'killpg') as kill:
                    proc.return_value.iterdir.return_value = entries
                    self.assertEqual(launch._group_alive(Mock(pid=12345)), state == 'S')
                kill.assert_called_once_with(12345, 0)

    def test_group_liveness_skips_vanished_proc_entries(self):
        with patch.object(launch, 'Path') as proc, patch.object(launch.os, 'killpg'):
            proc.return_value.iterdir.return_value = [
                _proc_entry(51, FileNotFoundError()),
                _proc_entry(53, ProcessLookupError()),
                _proc_entry(52, '52 (dead) Z 1 12345 12345'),
            ]
            self.assertFalse(launch._group_alive(Mock(pid=12345)))

    def test_missing_group_does_not_read_procfs(self):
        with (
            patch.object(launch, 'Path') as proc,
            patch.object(launch.os, 'killpg', side_effect=ProcessLookupError()),
        ):
            self.assertFalse(launch._group_alive(Mock(pid=12345)))
        proc.assert_not_called()

    def test_unreadable_procfs_or_stat_is_conservatively_alive(self):
        for location in ('directory', 'stat'):
            with self.subTest(location=location):
                with patch.object(launch, 'Path') as proc, patch.object(launch.os, 'killpg'):
                    if location == 'directory':
                        proc.return_value.iterdir.side_effect = PermissionError()
                    else:
                        proc.return_value.iterdir.return_value = [_proc_entry(51, PermissionError())]
                    self.assertTrue(launch._group_alive(Mock(pid=12345)))

    def test_malformed_proc_stat_is_conservatively_alive(self):
        for content in ('truncated', '51 (worker) S 1 invalid'):
            with self.subTest(content=content):
                with patch.object(launch, 'Path') as proc, patch.object(launch.os, 'killpg'):
                    proc.return_value.iterdir.return_value = [_proc_entry(51, content)]
                    self.assertTrue(launch._group_alive(Mock(pid=12345)))


def _proc_entry(pid, content):
    entry = MagicMock()
    entry.name = str(pid)
    reader = (entry / 'stat').read_text
    if isinstance(content, Exception):
        reader.side_effect = content
    else:
        reader.return_value = content
    return entry


@contextmanager
def _orphaned_group(zombie):
    # Adopt and reap the grandchild ourselves, never leave it to container PID 1.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    leader = unrelated = None
    try:
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        code = '''import os, signal, sys
child = os.fork()
if child == 0:
    if sys.argv[1] == 'zombie':
        os._exit(0)
    while True:
        signal.pause()
print(child, flush=True)
os._exit(0)
'''
        leader = subprocess.Popen(
            [sys.executable, '-c', code, 'zombie' if zombie else 'live'],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        if not select.select([leader.stdout], [], [], 3)[0]:
            raise TimeoutError('Fixture leader did not report its child')
        child_pid = int(leader.stdout.readline())
        leader.wait(timeout=3)
        deadline = time.monotonic() + 3
        while True:
            state, parent, group = _process_state(child_pid)
            if parent == os.getpid() and group == leader.pid and (state == 'Z') == zombie:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Fixture child was not adopted in the expected state')
            time.sleep(0.01)
        yield leader, child_pid, unrelated
    finally:
        errors = []
        for process in (leader, unrelated):
            if process is not None:
                try:
                    _reap_group(process)
                except Exception as error:
                    errors.append(error)
        if libc.prctl(36, previous.value, 0, 0, 0):
            errors.append(OSError(ctypes.get_errno(), 'Cannot restore child subreaper'))
        if errors:
            raise RuntimeError('Fixture cleanup failed') from errors[0]


def _process_state(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields[0], int(fields[1]), int(fields[2])


def _reap_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)
    if process.stdout is not None:
        process.stdout.close()
    deadline = time.monotonic() + 3
    while True:
        try:
            reaped, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            raise AssertionError(f'Fixture process group {process.pid} still exists')
        if not reaped:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Fixture process group {process.pid} did not drain')
            time.sleep(0.01)


if __name__ == '__main__':
    unittest.main()
