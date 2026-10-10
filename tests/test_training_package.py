"""Training data isolation/preflight and inference export regressions; no model load."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_globe_xtts_pilot as training
import export_training_bundle as bundle


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for split, speaker in [('train', 'a'), ('eval', 'b')]:
            (self.root / f'{split}.wav').write_bytes(b'placeholder')
            self.write_csv(split, f'{split}.wav', speaker)

    def write_csv(self, split, filename, speaker):
        (self.root / f'{split}.csv').write_text(
            f'audio_file|text|speaker_name\n{filename}|Hello.|{speaker}\n', encoding='utf-8'
        )

    def validate(self):
        return training.validate_dataset(self.root / 'train.csv', self.root / 'eval.csv')

    def test_relative_paths_are_portable(self):
        splits, report = self.validate()
        self.assertEqual(splits['train'][0]['audio_file'], str(self.root / 'train.wav'))
        self.assertEqual(report['speaker_overlap'], [])

    def test_prepared_audio_is_beneath_coqui_dataset_root(self):
        from TTS.tts.datasets import add_extra_keys
        from TTS.tts.datasets.formatters import coqui
        splits, _ = self.validate()
        original = splits['train'][0]['audio_file']
        prepared = self.root / 'run' / 'prepared'
        training.prepare_training_dataset(splits, prepared)
        for split in ('train', 'eval'):
            items = add_extra_keys(coqui(str(prepared), f'{split}.csv'), 'en', 'pilot')
            self.assertEqual(len(items), 1)
            self.assertTrue(Path(items[0]['audio_file']).is_relative_to(prepared))
            self.assertEqual(Path(items[0]['audio_file']).read_bytes(), b'placeholder')
            self.assertIn(f'audio/{split}/', items[0]['audio_unique_name'])
        self.assertEqual(splits['train'][0]['audio_file'], original)

    def test_shared_speaker_and_shared_audio_are_rejected(self):
        self.write_csv('eval', 'eval.wav', 'a')
        with self.assertRaisesRegex(ValueError, 'speakers overlap'):
            self.validate()
        self.write_csv('eval', 'train.wav', 'b')
        with self.assertRaisesRegex(ValueError, 'share an audio'):
            self.validate()

    def test_missing_files_and_missing_fields_are_rejected(self):
        self.write_csv('eval', 'absent.wav', 'b')
        with self.assertRaisesRegex(ValueError, 'missing audio'):
            self.validate()
        (self.root / 'eval.csv').write_text('audio_file|text|speaker_name\neval.wav|Hello\n')
        with self.assertRaisesRegex(ValueError, 'malformed'):
            self.validate()

    def test_no_cuda_exits_without_creating_training_output(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        output = self.root / 'output'
        with patch.dict(sys.modules, {'torch': fake_torch}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                training.main(['--train-csv', str(self.root / 'train.csv'), '--eval-csv', str(self.root / 'eval.csv'), '--output', str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())

    def test_broken_codec_stops_cpu_run_before_training_output_is_created(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), set_num_threads=lambda n: None)
        output = self.root / 'output'
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(training, 'check_audio_decoder', side_effect=RuntimeError('libnvrtc missing')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                training.main(['--train-csv', str(self.root / 'train.csv'), '--eval-csv', str(self.root / 'eval.csv'), '--output', str(output), '--allow-cpu'])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())

    def test_invalid_cli_values_are_rejected(self):
        for name, value in [('--epochs', '0'), ('--save-step', '0'), ('--workers', '-1'), ('--seed', str(2**32))]:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    training.main([name, value])
                self.assertEqual(error.exception.code, 2)


class InferenceExportTests(unittest.TestCase):
    def test_gradient_checkpointing_preserves_transformer_gradients(self):
        import torch
        from transformers import GPT2Config, GPT2Model
        torch.manual_seed(17)
        config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=16, n_positions=16,
                            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False)
        baseline, compact = GPT2Model(config), GPT2Model(config)
        compact.load_state_dict(baseline.state_dict())
        training.enable_memory_saving(compact)
        ids = torch.tensor([[1, 2, 3, 4]])
        for model in (baseline, compact):
            model.train()
            model(ids).last_hidden_state[..., 0].sum().backward()
        self.assertTrue(compact.is_gradient_checkpointing)
        for (_, left), (_, right) in zip(baseline.named_parameters(), compact.named_parameters()):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=1e-5), 'Checkpointing changed parameter gradients')

    def test_compact_snapshot_preserves_training_names_and_step_without_optimizer(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'model': {'xtts.gpt.weight': torch.tensor([1., 2.]),
                                  'xtts.dvae.weight': torch.ones(2)},
                        'optimizer': {'state': torch.ones(100)}, 'step': 200, 'epoch': 0}, root / 'checkpoint.pth')
            result = training.export_weight_snapshot(root / 'checkpoint.pth', root / 'weights.pth')
            saved = torch.load(root / 'weights.pth', weights_only=True)
            self.assertEqual(saved['step'], 200)
            self.assertEqual(saved['epoch'], 0)
            self.assertNotIn('optimizer', saved)
            self.assertEqual(set(saved['model']), {'xtts.gpt.weight'})
            self.assertEqual(saved['model']['xtts.gpt.weight'].dtype, torch.float16)
            self.assertEqual(result['sha256'], training.sha256_file(root / 'weights.pth'))

    def test_prompt_mask_uses_audio_token_grid_and_preserves_targets(self):
        import torch
        import torch.nn.functional as F
        # A three-second reference inside four seconds of speech. The upstream
        # 256-sample division masks every 1024-sample DVAE target in this case.
        indices = torch.tensor([[0, 66150]])
        length = 89
        old_targets = torch.zeros(length, dtype=torch.long)
        old_targets[:int(indices[0, 1] // 256)] = -1
        self.assertTrue(torch.isnan(F.cross_entropy(torch.zeros(length, 2), old_targets, ignore_index=-1)))
        corrected = training.correct_prompt_indices(indices, 1024, 256) // 256
        self.assertTrue(torch.equal(corrected, indices // 1024))
        targets = torch.zeros(length, dtype=torch.long)
        targets[int(corrected[0, 0]):int(corrected[0, 1])] = -1
        self.assertTrue(torch.isfinite(F.cross_entropy(torch.zeros(length, 2), targets, ignore_index=-1)))
        self.assertEqual(indices[0, 1].item(), 66150)

    def test_non_finite_loss_and_checkpoint_are_rejected(self):
        import torch
        training.require_finite({'loss': torch.tensor(1.0)}, 'loss')
        for value in (float('nan'), float('inf')):
            with self.assertRaises(FloatingPointError):
                training.require_finite({'loss': torch.tensor(value)}, 'loss')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({'model': {'xtts.gpt.weight': torch.tensor([float('nan')])}}, root / 'best_model.pth')
            with self.assertRaises(FloatingPointError):
                training.export_inference(root, root / 'config.json', root / 'vocab.json', root / 'inference')
            self.assertFalse((root / 'inference').exists())

    def test_optimizer_and_training_layers_are_removed_without_altering_voice_weights(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice_weight = torch.tensor([1.0, 2.0])
            torch.save({'model': {'xtts.gpt.weight': voice_weight, 'xtts.hifigan_decoder.weight': voice_weight + 1,
                                  'xtts.dvae.weight': torch.ones(4), 'torch_mel_spectrogram_dvae.weight': torch.ones(4)},
                        'optimizer': {'state': torch.ones(1000)}}, root / 'best_model.pth')
            (root / 'config.json').write_text('{"model":"xtts"}')
            (root / 'vocab.json').write_text('{"vocab":{}}')
            result = training.export_inference(root, root / 'config.json', root / 'vocab.json', root / 'inference')
            loaded = torch.load(root / 'inference/model.pth', weights_only=True)
            self.assertEqual(set(loaded), {'model'})
            self.assertEqual(set(loaded['model']), {'gpt.weight', 'hifigan_decoder.weight'})
            self.assertTrue(torch.equal(loaded['model']['gpt.weight'], voice_weight))
            self.assertTrue(result['optimizer_removed'])
            for name, digest in result['sha256'].items():
                self.assertEqual(training.sha256_file(root / 'inference' / name), digest)


class BundleIsolationTests(unittest.TestCase):
    def test_only_public_audio_is_exported_and_csvs_are_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            public = project / bundle.DATA_RELATIVE
            for split, speaker in [('train', 'a'), ('eval', 'b')]:
                audio = public / 'audio' / split / speaker / 'clip.wav'
                audio.parent.mkdir(parents=True)
                audio.write_bytes(b'public audio')
                (public / f'{split}.csv').write_text(f'audio_file|text|speaker_name\n{audio}|Hello.|{speaker}\n')
            (public / 'dataset_report.json').write_text('{}')
            for name in ('scripts/train_globe_xtts_pilot.py', 'scripts/export_training_bundle.py', 'tests/test_training_package.py', 'docs/FINETUNING.md', 'docs/EXPERIMENTS.md', 'docs/ACCENT_ADAPTATION.md'):
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test fixture')
            (project / 'private-user-reference.wav').write_bytes(b'private voice')
            output = project / 'bundle.zip'
            with patch.object(bundle, 'PROJECT', project), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bundle.main(['--output', str(output)]), 0)
            with zipfile.ZipFile(output) as archive:
                self.assertNotIn('private-user-reference.wav', archive.namelist())
                manifest = json.loads(archive.read('bundle_manifest.json'))
                self.assertFalse(manifest['contains_user_recordings'])
                for split in ('train', 'eval'):
                    rows = list(csv.DictReader(io.StringIO(archive.read((bundle.DATA_RELATIVE / f'{split}.csv').as_posix()).decode()), delimiter='|'))
                    self.assertFalse(Path(rows[0]['audio_file']).is_absolute())
                    self.assertIn((bundle.DATA_RELATIVE / rows[0]['audio_file']).as_posix(), archive.namelist())
            # Even a CSV injected into the public folder cannot include a private file.
            private = project / 'private-user-reference.wav'
            (public / 'eval.csv').write_text(f'audio_file|text|speaker_name\n{private}|Hello.|b\n')
            with patch.object(bundle, 'PROJECT', project), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bundle.main(['--output', str(project / 'unsafe.zip')])
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((project / 'unsafe.zip').exists())


if __name__ == '__main__':
    unittest.main()
