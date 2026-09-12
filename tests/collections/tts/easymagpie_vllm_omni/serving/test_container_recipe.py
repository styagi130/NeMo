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
"""Source-only guards for the standalone container's dependency correction."""

import shlex
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[5] / 'tools' / 'easymagpie_vllm_omni'
BASE = 'vllm/vllm-openai:v0.26.0@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52'


class ContainerRecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = (PACKAGE / 'Dockerfile').read_text().replace('\\\n', ' ')

    def test_builder_and_final_use_the_same_pinned_public_parent(self):
        parents = [line for line in self.recipe.splitlines() if line.startswith('FROM ')]
        self.assertEqual(parents, [f'FROM {BASE} AS cairo_builder', f'FROM {BASE}'])

    def test_native_build_prerequisites_do_not_enter_final_stage(self):
        builder, final = self.recipe.split(f'\nFROM {BASE}\n', 1)
        self.assertIn('libcairo2-dev=1.16.0-5ubuntu2.1', builder)
        self.assertIn('pkg-config=0.29.2-1ubuntu3', builder)
        self.assertIn('--no-install-recommends', builder)
        self.assertNotIn('apt-get', final)
        self.assertIn('COPY --from=cairo_builder /wheels /wheels', final)

    def test_cairo_source_and_build_resolver_are_pinned(self):
        self.assertIn('--no-deps --no-build-isolation', self.recipe)
        self.assertIn('--wheel-dir /wheels', self.recipe)
        self.assertIn(
            'https://files.pythonhosted.org/packages/19/4f/'
            '0d48a017090d4527e921d6892bc550ae869902e67859fc960f8fe63a9094/'
            'pycairo-1.26.1.tar.gz#sha256=a11b999ce55b798dbf13516ab038e0ce8b6ec299b208d7c4e767a6f7e68e8430',
            self.recipe,
        )

    def test_final_correction_is_narrow_and_package_install_is_checked(self):
        lines = [' '.join(line.split()) for line in self.recipe.splitlines()]
        self.assertIn(
            'RUN python3 -m pip install --no-cache-dir --no-deps /wheels/pycairo-*.whl nixl-cu13==1.3.1',
            lines,
        )
        self.assertIn(
            'RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/easymagpie'
            ' && python3 -m pip check',
            lines,
        )
        self.assertNotIn('--upgrade', self.recipe)
        self.assertNotIn('|| true', self.recipe)

    def test_default_entrypoint_and_entire_package_install_are_retained(self):
        self.assertIn('COPY tools/easymagpie_vllm_omni /opt/easymagpie', self.recipe)
        self.assertIn('ENTRYPOINT ["bash", "/opt/easymagpie/scripts/run_server.sh"]', self.recipe)

    def test_final_stage_has_writable_cache_defaults_without_changing_home(self):
        final = self.recipe.split(f'\nFROM {BASE}\n', 1)[1]
        values = dict(
            token.split('=', 1)
            for line in final.splitlines()
            if line.startswith('ENV ')
            for token in shlex.split(line[4:])
        )
        self.assertEqual(
            values,
            {
                'XDG_CACHE_HOME': '/tmp/easymagpie-cache',
                'VLLM_CACHE_ROOT': '/tmp/easymagpie-vllm',
                'TORCHINDUCTOR_CACHE_DIR': '/tmp/easymagpie-inductor',
                'TRITON_CACHE_DIR': '/tmp/easymagpie-triton',
                'FLASHINFER_WORKSPACE_BASE': '/tmp/easymagpie-flashinfer',
                'HF_HOME': '/tmp/easymagpie-hf',
                'SPEAKER_SAMPLES_DIR': '/tmp/easymagpie-speaker-samples',
            },
        )


if __name__ == '__main__':
    unittest.main()
