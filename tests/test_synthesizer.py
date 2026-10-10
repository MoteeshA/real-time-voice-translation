import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from voice_translation import synthesizer as s


class ReferenceTests(unittest.TestCase):
    def test_long_english_and_hindi_text_splitting_has_runtime_dependencies(self):
        from TTS.tts.layers.xtts.tokenizer import split_sentence
        samples = {
            'en': 'This is a natural sentence for the voice translation project. ' * 12,
            'hi': 'यह आवाज अनुवाद परियोजना के लिए एक स्पष्ट वाक्य है। ' * 12,
        }
        for language, text in samples.items():
            with self.subTest(language=language):
                parts = split_sentence(text.strip(), language, 250)
                self.assertGreater(len(parts), 1)
                self.assertTrue(all(0 < len(part) <= 250 for part in parts))
                self.assertEqual(' '.join(' '.join(parts).split()), ' '.join(text.split()))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        s._CONDITIONING_CACHE.clear()

    def tearDown(self):
        self.tmp.cleanup()
        s._CONDITIONING_CACHE.clear()

    def wav(self, name, audio):
        p = self.root / name
        sf.write(p, audio, 22050, subtype='FLOAT')
        return str(p)

    def tone(self, seconds=4):
        return (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * 22050)) / 22050)).astype('float32')

    def test_silent_short_and_clipped_rejected(self):
        for name, audio in [('silent', np.zeros(22050 * 4)), ('short', self.tone(1)),
                            ('clipped', np.ones(22050 * 4))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                s.inspect_reference(self.wav(name + '.wav', audio))

    def test_second_reference_gets_conditioning_budget(self):
        paths = [self.wav(f'{i}.wav', self.tone(20)) for i in range(2)]
        with s.prepare_references(paths) as prepared:
            for path in prepared:
                self.assertAlmostEqual(sf.info(path).duration, 15, places=2)
                self.assertLessEqual(np.max(np.abs(sf.read(path)[0])), .91)
        self.assertFalse(Path(prepared[0]).exists())

    def test_cache_uses_content_not_temporary_filename(self):
        paths = [self.wav(f'{i}.wav', self.tone()) for i in range(2)]
        model = Mock()
        model.get_conditioning_latents.return_value = ('lat', 'emb')
        tts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model))
        self.assertEqual(s.get_conditioning(tts, [paths[0]]), ('lat', 'emb'))
        self.assertEqual(s.get_conditioning(tts, [paths[1]]), ('lat', 'emb'))
        self.assertEqual(model.get_conditioning_latents.call_count, 1)
        sf.write(paths[1], self.tone() * .5, 22050)
        s.get_conditioning(tts, [paths[1]])
        self.assertEqual(model.get_conditioning_latents.call_count, 2)

    def test_native_rate_overshoot_protection_and_settings(self):
        import torch
        model = Mock(device=torch.device('cpu'))
        model.inference.return_value = {'wav': np.array([0, 1.3, -.5, .2], dtype='float32')}
        tts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model))
        output = self.root / 'result.wav'
        with patch.object(s, 'get_conditioning', return_value=('a', 'b')):
            s.synthesize_to_file(tts, text='Hello.', language='en', speaker_reference=['ref'], file_path=output)
        wav, rate = sf.read(output)
        self.assertEqual(rate, 24000)
        self.assertLessEqual(abs(wav).max(), .981)
        self.assertEqual(model.inference.call_args.kwargs['repetition_penalty'], 5)
        self.assertTrue(model.inference.call_args.kwargs['enable_text_splitting'])

    def test_silent_model_output_is_not_presented_as_success(self):
        import torch
        model = Mock(device=torch.device('cpu'))
        model.inference.return_value = {'wav': np.zeros(100)}
        tts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model))
        with patch.object(s, 'get_conditioning', return_value=('a','b')), self.assertRaises(RuntimeError):
            s.synthesize_to_file(tts, text='Hello.', language='en', speaker_reference=['ref'], file_path=self.root/'out.wav')

    def test_license_is_checked_before_model_loading(self):
        with patch.object(s, '_load_tts') as loader, self.assertRaises(RuntimeError):
            s.get_tts(False)
        loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
