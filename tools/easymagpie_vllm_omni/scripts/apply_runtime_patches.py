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
"""Install the exact validated runtime hooks without importing vLLM or CUDA.

Run with the target environment's Python, while all workers are stopped. With
no options, apply to an entirely original installation or accept an entirely
patched installation. ``--check`` only checks compatibility; ``--verify`` only
accepts an already patched installation. ``--site-packages`` selects an offline
installation containing both packages and their distribution metadata.

All input/output hashes are checked before replacing either file. Ordinary
write errors roll back completed replacements. An interrupted process can leave
a partial installation: subsequent calls reject it; reinstall the pinned
packages before retrying. Do not run another package installer concurrently.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath

BUNDLE = Path(__file__).resolve().parents[1] / 'runtime_patches'


class PatchError(RuntimeError):
    """The installation or bundled patch does not match the validated inputs."""


def apply_runtime_patches(site_packages=None, *, check=False, verify=False):
    """Check or install both runtime files, returning their verified hashes."""
    if check and verify:
        raise PatchError('Choose only one of check and verify.')
    manifest, patches = _load_bundle()
    targets = _resolve_targets(manifest, site_packages)
    states = []
    for target in targets:
        path = target['target']
        if not path.is_file() or path.is_symlink():
            raise PatchError(f'Missing or nonregular runtime file: {path}')
        target['original'] = path.read_bytes()
        target['mode'] = stat.S_IMODE(path.stat().st_mode)
        digest = _sha(target['original'])
        if digest == target['original_sha256']:
            states.append('original')
        elif digest == target['patched_sha256']:
            states.append('patched')
        else:
            raise PatchError(f'Incompatible runtime file: {path} (SHA256 {digest})')
    if len(set(states)) != 1:
        raise PatchError('Partially patched installation; reinstall both pinned packages before retrying.')
    state = states[0]
    if verify and state != 'patched':
        raise PatchError('Runtime hooks are not installed; run this script without --verify first.')
    if state == 'original':
        for target in targets:
            target['updated'] = _apply_hunks(target['original'], patches[target['path']])
            if _sha(target['updated']) != target['patched_sha256']:
                raise PatchError(f'Patched output hash mismatch: {target["path"]}')
        if not check:
            _commit(targets)
    status = 'compatible_original' if check and state == 'original' else 'verified_patched'
    if state == 'original' and not check:
        status = 'applied'
    expected = 'original_sha256' if status == 'compatible_original' else 'patched_sha256'
    return {
        'status': status,
        'distributions': manifest['distributions'],
        'patch_sha256': manifest['patch_sha256'],
        'files': {str(target['target']): target[expected] for target in targets},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='Validate compatibility without changing files.')
    modes.add_argument('--verify', action='store_true', help='Require both files to be already patched; never write.')
    parser.add_argument('--site-packages', type=Path, help='Offline package root, including .dist-info metadata.')
    args = parser.parse_args()
    try:
        result = apply_runtime_patches(args.site_packages, check=args.check, verify=args.verify)
    except (PatchError, OSError, ValueError) as error:
        parser.exit(1, f'Runtime patch validation failed: {error}\n')
    print(json.dumps(result, indent=2))


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _load_bundle():
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('patch') != 'runtime.patch':
        raise PatchError('Unsupported runtime patch manifest.')
    if manifest.get('distributions') != {'vllm': '0.26.0', 'vllm-omni': '0.26.0'}:
        raise PatchError('Unsupported runtime distribution versions in manifest.')
    files = manifest.get('files', [])
    expected = {'vllm/v1/core/sched/scheduler.py': 'vllm', 'vllm_omni/worker/gpu_model_runner.py': 'vllm-omni'}
    if len(files) != len(expected) or {row['path']: row['distribution'] for row in files} != expected:
        raise PatchError('Manifest must describe exactly the two validated runtime files.')
    for row in files:
        for key in ('original_sha256', 'patched_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise PatchError(f'Invalid hash for {row["path"]}')
    patch = (BUNDLE / manifest['patch']).read_bytes()
    if _sha(patch) != manifest['patch_sha256']:
        raise PatchError('Bundled runtime.patch hash mismatch.')
    patches = _parse_patch(patch)
    if patches.keys() != expected.keys():
        raise PatchError('Patch and manifest file lists differ.')
    return manifest, patches


def _resolve_targets(manifest, site_packages):
    if site_packages is None:
        distributions = list(importlib.metadata.distributions())
    else:
        site_packages = Path(site_packages).resolve(strict=True)
        distributions = list(importlib.metadata.distributions(path=[str(site_packages)]))
    roots = {}
    for name, version in manifest['distributions'].items():
        matches = [d for d in distributions if (d.metadata.get('Name') or '').lower().replace('_', '-') == name]
        if len(matches) != 1 or matches[0].version != version:
            found = [d.version for d in matches]
            raise PatchError(f'Requires exactly one {name}=={version} distribution; found {found}')
        roots[name] = Path(matches[0].locate_file('')).resolve(strict=True)
    targets = []
    for row in manifest['files']:
        root = roots[row['distribution']]
        relative = PurePosixPath(row['path'])
        path = root.joinpath(*relative.parts)
        if path.resolve() != path or not path.is_relative_to(root):
            raise PatchError(f'Symlinked runtime path is unsupported: {path}')
        targets.append({**row, 'target': path})
    return targets


def _parse_patch(patch):
    """Read only exact unified text hunks; no fuzz, renames or arbitrary paths."""
    lines = patch.splitlines(keepends=True)
    parsed = {}
    index = 0
    while index < len(lines):
        match = re.fullmatch(rb'diff --git a/([^\s]+) b/([^\s]+)\n', lines[index])
        if not match or match[1] != match[2]:
            raise PatchError('Unsupported patch file header.')
        name = match[1].decode('ascii')
        if name in parsed or lines[index + 1 : index + 3] != [
            b'--- a/' + match[1] + b'\n',
            b'+++ b/' + match[1] + b'\n',
        ]:
            raise PatchError('Invalid or duplicate patch target.')
        index += 3
        hunks = []
        while index < len(lines) and not lines[index].startswith(b'diff --git '):
            header = re.fullmatch(rb'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', lines[index])
            if not header:
                raise PatchError('Unsupported patch hunk header.')
            old_start, old_count, new_start, new_count = map(int, header.groups())
            index += 1
            before, after = [], []
            while index < len(lines) and lines[index][:1] in (b' ', b'+', b'-'):
                line = lines[index]
                if line[:1] != b'+':
                    before.append(line[1:])
                if line[:1] != b'-':
                    after.append(line[1:])
                index += 1
            if len(before) != old_count or len(after) != new_count:
                raise PatchError('Patch hunk line counts differ.')
            hunks.append((old_start, new_start, before, after))
        if not hunks:
            raise PatchError('Empty patch file.')
        parsed[name] = hunks
    return parsed


def _apply_hunks(original, hunks):
    lines = original.splitlines(keepends=True)
    output, cursor = [], 0
    for old_start, new_start, before, after in hunks:
        start = old_start - 1
        if start < cursor or lines[start : start + len(before)] != before:
            raise PatchError('Patch context mismatch; no fuzzy application is allowed.')
        output.extend(lines[cursor:start])
        if len(output) != new_start - 1:
            raise PatchError('Patch output position mismatch.')
        output.extend(after)
        cursor = start + len(before)
    output.extend(lines[cursor:])
    return b''.join(output)


def _stage(path, content, mode):
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.easymagpie-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _commit(targets):
    staged, backups, committed = {}, {}, []
    try:
        for target in targets:
            path = target['target']
            staged[path] = _stage(path, target['updated'], target['mode'])
            backups[path] = _stage(path, target['original'], target['mode'])
        # Detect intervening package changes before replacing either original.
        if any(target['target'].read_bytes() != target['original'] for target in targets):
            raise PatchError('Runtime files changed during preparation; nothing was installed.')
        try:
            for target in targets:
                path = target['target']
                os.replace(staged[path], path)
                committed.append(path)
            for target in targets:
                if _sha(target['target'].read_bytes()) != target['patched_sha256']:
                    raise PatchError(f'Installed output hash mismatch: {target["target"]}')
        except BaseException as error:
            failed = []
            for path in reversed(committed):
                backup = backups.pop(path)
                try:
                    os.replace(backup, path)
                except OSError:
                    failed.append(f'{path} from {backup}')
            if failed:
                raise PatchError(
                    'Rollback failed; restore ' + '; '.join(failed) + ' before using this runtime.'
                ) from error
            raise
    finally:
        for temporary in [*staged.values(), *backups.values()]:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
