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
"""CPU-only installer tests; optionally exercise exact retained upstream files.

Run ``python -m unittest discover -s runtime_patches/tests -v`` from the tool
directory. Set EASYMAGPIE_RUNTIME_ORIGINALS to an unmodified package root to also
test the real bundled patch. That root is only read; all installs use temporary
copies. No installed runtime packages, torch imports or GPU access are needed.
"""

import copy
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOL = Path(__file__).resolve().parents[2]
SCRIPT = TOOL / 'scripts' / 'apply_runtime_patches.py'
SPEC = importlib.util.spec_from_file_location('runtime_patch_installer', SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
REAL_BUNDLE = installer.BUNDLE
MANIFEST = json.loads((REAL_BUNDLE / 'manifest.json').read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def metadata(root):
    for name, version in MANIFEST['distributions'].items():
        directory = root / f'{name.replace("-", "_")}-{version}.dist-info'
        directory.mkdir(parents=True)
        (directory / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.site = root / 'site'
        self.site.mkdir()
        metadata(self.site)
        self.bundle = root / 'bundle'
        self.bundle.mkdir()
        self.manifest = copy.deepcopy(MANIFEST)
        self.originals, self.updated = {}, {}
        patch = ''
        for index, row in enumerate(self.manifest['files']):
            name = row['path']
            before = f'# fixture {index}\nvalue = 1\n# end\n'
            after = f'# fixture {index}\nvalue = 2\nhook = True\n# end\n'
            self.originals[name], self.updated[name] = before.encode(), after.encode()
            path = self.site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before.encode())
            row['original_sha256'] = digest(before.encode())
            row['patched_sha256'] = digest(after.encode())
            patch += f'diff --git a/{name} b/{name}\n'
            patch += ''.join(
                difflib.unified_diff(before.splitlines(True), after.splitlines(True), f'a/{name}', f'b/{name}')
            )
        (self.bundle / 'runtime.patch').write_text(patch)
        self.manifest['patch_sha256'] = digest(patch.encode())
        self.save_manifest()
        patcher = mock.patch.object(installer, 'BUNDLE', self.bundle)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_manifest(self):
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))

    def contents(self):
        return {name: (self.site / name).read_bytes() for name in self.originals}

    def assert_original(self):
        self.assertEqual(self.contents(), self.originals)
        self.assertFalse(list(self.site.rglob('*.easymagpie-*')))

    def test_check_and_verify_never_write(self):
        before = {p: p.stat().st_mtime_ns for p in self.site.rglob('*')}
        result = installer.apply_runtime_patches(self.site, check=True)
        self.assertEqual(result['status'], 'compatible_original')
        with self.assertRaisesRegex(installer.PatchError, 'not installed'):
            installer.apply_runtime_patches(self.site, verify=True)
        self.assert_original()
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.site.rglob('*')})

    def test_apply_exact_outputs_and_idempotence(self):
        self.assertEqual(installer.apply_runtime_patches(self.site)['status'], 'applied')
        self.assertEqual(self.contents(), self.updated)
        with mock.patch.object(installer, '_commit', side_effect=AssertionError('must not write')):
            for flags in ({}, {'check': True}, {'verify': True}):
                self.assertEqual(installer.apply_runtime_patches(self.site, **flags)['status'], 'verified_patched')

    def test_unknown_second_file_rejects_before_first_write(self):
        name = list(self.originals)[1]
        (self.site / name).write_bytes(b'unrecognized runtime\n')
        before = self.contents()
        with self.assertRaisesRegex(installer.PatchError, 'Incompatible'):
            installer.apply_runtime_patches(self.site)
        self.assertEqual(self.contents(), before)

    def test_partial_install_rejected_in_both_orders(self):
        for patched in self.originals:
            with self.subTest(patched=patched):
                for name in self.originals:
                    (self.site / name).write_bytes(self.updated[name] if name == patched else self.originals[name])
                before = self.contents()
                with self.assertRaisesRegex(installer.PatchError, 'Partially patched'):
                    installer.apply_runtime_patches(self.site)
                self.assertEqual(self.contents(), before)

    def test_previous_two_hook_installation_rejected_without_writes(self):
        for name in list(self.originals)[:2]:
            (self.site / name).write_bytes(self.updated[name])
        before = self.contents()
        for flags in ({}, {'check': True}, {'verify': True}):
            with self.assertRaisesRegex(installer.PatchError, 'Partially patched'):
                installer.apply_runtime_patches(self.site, **flags)
            self.assertEqual(self.contents(), before)

    def test_missing_file_does_not_touch_other(self):
        name = list(self.originals)[1]
        (self.site / name).unlink()
        with self.assertRaisesRegex(installer.PatchError, 'Missing'):
            installer.apply_runtime_patches(self.site)
        first = next(iter(self.originals))
        self.assertEqual((self.site / first).read_bytes(), self.originals[first])

    def test_wrong_missing_and_duplicate_version_rejected(self):
        path = next(self.site.glob('vllm-*.dist-info/METADATA'))
        original = path.read_text()
        path.write_text(original.replace('Version: 0.26.0', 'Version: 0.26.1'))
        with self.assertRaisesRegex(installer.PatchError, 'Requires exactly one'):
            installer.apply_runtime_patches(self.site)
        path.write_text(original)
        duplicate = self.site / 'duplicate.dist-info'
        duplicate.mkdir()
        (duplicate / 'METADATA').write_text(original)
        with self.assertRaisesRegex(installer.PatchError, 'Requires exactly one'):
            installer.apply_runtime_patches(self.site)
        shutil.rmtree(duplicate)
        shutil.rmtree(path.parent)
        with self.assertRaisesRegex(installer.PatchError, 'Requires exactly one'):
            installer.apply_runtime_patches(self.site)
        self.assert_original()

    def test_patch_hash_and_output_hash_rejected_before_writes(self):
        patch = self.bundle / 'runtime.patch'
        original = patch.read_bytes()
        patch.write_bytes(original + b'alteration\n')
        with self.assertRaisesRegex(installer.PatchError, 'patch hash mismatch'):
            installer.apply_runtime_patches(self.site)
        patch.write_bytes(original)
        self.manifest['files'][1]['patched_sha256'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(installer.PatchError, 'output hash mismatch'):
            installer.apply_runtime_patches(self.site)
        self.assert_original()

    def test_context_mismatch_is_not_fuzzy_applied(self):
        row = self.manifest['files'][1]
        changed = self.originals[row['path']].replace(b'value = 1', b'value = 0')
        (self.site / row['path']).write_bytes(changed)
        row['original_sha256'] = digest(changed)
        self.save_manifest()
        before = self.contents()
        with self.assertRaisesRegex(installer.PatchError, 'context mismatch'):
            installer.apply_runtime_patches(self.site)
        self.assertEqual(self.contents(), before)

    def test_symlinked_target_rejected(self):
        name = next(iter(self.originals))
        path = self.site / name
        other = self.site / 'external.py'
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaisesRegex(installer.PatchError, 'Symlinked'):
            installer.apply_runtime_patches(self.site)
        self.assertEqual(other.read_bytes(), self.originals[name])

    def test_manifest_cannot_add_or_redirect_targets(self):
        self.manifest['files'][0]['path'] = '../elsewhere.py'
        self.save_manifest()
        with self.assertRaisesRegex(installer.PatchError, 'exactly the three'):
            installer.apply_runtime_patches(self.site)
        self.assert_original()

    def test_staging_failure_leaves_originals_and_no_temporaries(self):
        actual = installer._stage
        calls = 0

        def stage(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('disk full')
            return actual(*args)

        with mock.patch.object(installer, '_stage', side_effect=stage):
            with self.assertRaisesRegex(OSError, 'disk full'):
                installer.apply_runtime_patches(self.site)
        self.assert_original()

    def test_second_replace_failure_rolls_back_first(self):
        actual = os.replace
        calls = 0

        def replace(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('replace denied')
            return actual(*args)

        with mock.patch.object(installer.os, 'replace', side_effect=replace):
            with self.assertRaisesRegex(OSError, 'replace denied'):
                installer.apply_runtime_patches(self.site)
        self.assertEqual(calls, 3)
        self.assert_original()

    def test_intervening_change_rejected_before_replacement(self):
        actual = installer._stage
        calls = 0
        path = self.site / next(iter(self.originals))

        def stage(*args):
            nonlocal calls
            temporary = actual(*args)
            calls += 1
            if calls == 4:
                path.write_bytes(b'concurrent installer\n')
            return temporary

        with mock.patch.object(installer, '_stage', side_effect=stage):
            with self.assertRaisesRegex(installer.PatchError, 'changed during preparation'):
                installer.apply_runtime_patches(self.site)
        self.assertEqual(path.read_bytes(), b'concurrent installer\n')
        second = list(self.originals)[1]
        self.assertEqual((self.site / second).read_bytes(), self.originals[second])
        self.assertFalse(list(self.site.rglob('*.easymagpie-*')))

    def test_failed_rollback_retains_original_backup_and_rejects_partial_state(self):
        actual = os.replace
        calls = 0

        def replace(*args):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise OSError('filesystem unavailable')
            return actual(*args)

        with mock.patch.object(installer.os, 'replace', side_effect=replace):
            with self.assertRaisesRegex(installer.PatchError, 'Rollback failed; restore'):
                installer.apply_runtime_patches(self.site)
        backups = list(self.site.rglob('*.easymagpie-*'))
        self.assertEqual(len(backups), 1)
        first = next(iter(self.originals))
        self.assertEqual(backups[0].read_bytes(), self.originals[first])
        with self.assertRaisesRegex(installer.PatchError, 'Partially patched'):
            installer.apply_runtime_patches(self.site, verify=True)

    def test_file_permissions_preserved_without_runtime_imports(self):
        for name in self.originals:
            (self.site / name).chmod(0o640)
        before = set(sys.modules)
        installer.apply_runtime_patches(self.site)
        self.assertFalse({'torch', 'vllm', 'vllm_omni'} & (set(sys.modules) - before))
        for name in self.originals:
            self.assertEqual((self.site / name).stat().st_mode & 0o777, 0o640)


@unittest.skipUnless(
    os.environ.get('EASYMAGPIE_RUNTIME_ORIGINALS'), 'Set EASYMAGPIE_RUNTIME_ORIGINALS for exact-file proof'
)
class RetainedRuntimeTests(unittest.TestCase):
    def test_exact_validated_runtime_patch_cli_and_verify(self):
        upstream = Path(os.environ['EASYMAGPIE_RUNTIME_ORIGINALS'])
        with tempfile.TemporaryDirectory() as name:
            site = Path(name)
            metadata(site)
            for row in MANIFEST['files']:
                original = (upstream / row['path']).read_bytes()
                self.assertEqual(digest(original), row['original_sha256'])
                target = site / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original)
            command = [sys.executable, str(SCRIPT), '--site-packages', str(site)]
            result = subprocess.run([*command, '--verify'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            for flags, status in [
                (['--check'], 'compatible_original'),
                ([], 'applied'),
                ([], 'verified_patched'),
                (['--verify'], 'verified_patched'),
            ]:
                result = subprocess.run([*command, *flags], capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)['status'], status)
            for row in MANIFEST['files']:
                self.assertEqual(digest((site / row['path']).read_bytes()), row['patched_sha256'])
                self.assertEqual(digest((upstream / row['path']).read_bytes()), row['original_sha256'])


if __name__ == '__main__':
    unittest.main()
