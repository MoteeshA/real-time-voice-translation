import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_translation import quality


class QualityTests(unittest.TestCase):
    def test_invalid_audio_is_rejected_before_verifier_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'silent.wav'
            sf.write(path, np.zeros(32000), 16000)
            with patch.object(quality, '_speaker_verifier') as loader:
                with self.assertRaises(ValueError):
                    quality.compare_voice(path, path)
                loader.assert_not_called()

    def test_analysis_window_is_bounded_without_misreporting_clip_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'long.wav'
            t = np.arange(20 * 16000) / 16000
            sf.write(path, .1 * np.sin(2 * np.pi * 130 * t), 16000)
            audio, info = quality.load_comparison_audio(path)
            self.assertEqual(len(audio), 15 * 16000)
            self.assertEqual(info['duration_seconds'], 20)
            self.assertEqual(info['analysed_seconds'], 15)

    def test_report_does_not_present_cosine_as_a_percentage_or_certificate(self):
        clip = {'duration_seconds': 3, 'hard_clip_fraction': 0,
                'pitch': {'median_hz': None, 'spread_semitones': None}}
        text = quality.format_comparison({'speaker_cosine': .932, 'reference': clip, 'generated': clip})
        self.assertIn('0.932', text)
        self.assertIn('not a percentage or exact-match verdict', text)
        self.assertIn('Not detected', text)
        self.assertIn('first 15 seconds', text)


if __name__ == '__main__':
    unittest.main()
