"""Provision continuation metadata without exporting an encoder or raw reserve."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from easymagpie_vllm_omni.export_rolling_context import export


class ExportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        (self.output / 'speaker_embeddings').mkdir()
        (self.output / 'codec_native').mkdir()
        (self.output / 'config.json').write_text(json.dumps(dict(embedding_dim=4)))
        (self.output / 'codec_native/config.json').write_text(
            json.dumps(dict(output_sample_rate=22050, pre_upsample_rates=[2], resblock_upsample_rates=[9, 7, 7]))
        )

    def save(self, rows):
        tensor = torch.randn(rows, 4)
        path = self.output / 'speaker_embeddings/voice.pt'
        torch.save(dict(speaker_encoding=tensor), path)
        return tensor, path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_existing_tensors_are_preserved_and_no_encoder_or_reserve_is_exported(self):
        for rows in [2, 96]:
            with self.subTest(rows=rows):
                saved, path, before = self.save(rows)
                export([dict(speaker_id='voice')], self.output, 10)
                config = json.loads((self.output / 'rolling_context/config.json').read_text())
                self.assertEqual(config['schema_version'], 5)
                self.assertEqual(config['original_has_audio']['voice'], rows != 2)
                self.assertEqual(
                    config['initial_contexts']['voice']['sha256'], hashlib.sha256(saved.numpy().tobytes()).hexdigest()
                )
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
                self.assertEqual(config['total_frames'], 250)
                self.assertEqual(config['frames_per_second'], 25)
                self.assertEqual(config['max_generated_frames'], 125)
                self.assertEqual([p.name for p in (self.output / 'rolling_context').iterdir()], ['config.json'])
                (self.output / 'rolling_context/config.json').unlink()
                (self.output / 'rolling_context').rmdir()

    def test_audio_bearing_saved_tensor_requires_no_reference_recording(self):
        self.save(96)
        export([dict(speaker_id='voice', original_has_audio=True)], self.output, 10, 75)
        config = json.loads((self.output / 'rolling_context/config.json').read_text())
        self.assertEqual(config['max_generated_frames'], 75)

    def test_invalid_budget_or_tail_is_rejected(self):
        self.save(96)
        for duration, tail in [(0, 125), (float('nan'), 125), (10, 0), (10, 250), (10, True)]:
            with self.subTest(duration=duration, tail=tail), self.assertRaises(ValueError):
                export([dict(speaker_id='voice')], self.output, duration, tail)

    def test_invalid_saved_tensor_or_audio_flag_is_rejected(self):
        for rows, flag in [(96, False), (2, 'false'), (1, True)]:
            self.save(rows)
            with self.subTest(rows=rows, flag=flag), self.assertRaises(ValueError):
                export([dict(speaker_id='voice', original_has_audio=flag)], self.output, 10)
