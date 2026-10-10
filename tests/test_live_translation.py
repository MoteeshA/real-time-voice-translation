"""Runtime regressions; models are stubbed so no download or GPU is needed."""

import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from voice_translation import live, translation


class MicrophoneTests(unittest.TestCase):
    def test_settings_changes_before_microphone_connection_are_safe(self):
        from fastrtc import WebRTC
        from fastrtc.utils import WebRTCModel
        component = live.allow_empty_microphone(WebRTC(render=False))
        self.assertIsNone(component.preprocess(None))
        self.assertEqual(component.preprocess(WebRTCModel(root='test-session')).webrtc_id, 'test-session')
        # Updating extra inputs before an offer is expected to do nothing.
        component.set_input_gradio(None, 'reference', True)

    def test_session_id_does_not_become_an_extra_callback_argument(self):
        from fastrtc import WebRTC, ReplyOnPause
        from fastrtc.utils import WebRTCModel
        component = live.allow_empty_microphone(WebRTC(render=False))
        handler = ReplyOnPause(live.live_translate_speak)
        session = component.preprocess(WebRTCModel(root="test-session"))
        handler.set_args([session, "reference.wav", None, True, True, 0.75])
        self.assertEqual(len(handler.latest_args), 6)
        handler.latest_args[0].audio = (16000, np.zeros((1, 16000)))
        import inspect
        inspect.signature(live.live_translate_speak).bind(*handler.latest_args)
        self.assertEqual(list(live.live_translate_speak(*handler.latest_args)), [])

    def test_stereo_integer_pcm_is_scaled_before_downmix(self):
        channel = np.array([0, 16384, -32768, 32767], dtype=np.int16)
        for samples in (np.stack([channel, channel]), np.stack([channel, channel], axis=1)):
            rate, mono = live.normalize_microphone_audio((16000, samples))
            self.assertEqual(rate, 16000)
            self.assertEqual(mono.dtype, np.float32)
            np.testing.assert_allclose(mono, channel.astype(np.float32) / 32768)

    def test_unsigned_pcm_has_silence_at_midpoint(self):
        _, mono = live.normalize_microphone_audio((16000, np.array([0, 128, 255], dtype=np.uint8)))
        np.testing.assert_allclose(mono, [-1, 0, 127 / 128])

    def test_rejects_invalid_audio(self):
        for audio in ((0, np.zeros(10)), (16000, np.array([np.nan])), (16000, np.zeros((4, 4)))):
            with self.assertRaises(ValueError):
                live.normalize_microphone_audio(audio)


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "input.wav")

    def test_silence_never_loads_whisper(self):
        sf.write(self.path, np.zeros(16000, dtype=np.float32), 16000)
        with patch.object(translation, "_load_model") as load:
            with self.assertRaises(translation.NoSpeechDetectedError):
                translation.translate_hindi_to_english(self.path)
            load.assert_not_called()

    def test_missing_and_corrupt_recordings_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "Record or upload"):
            translation.translate_hindi_to_english(self.path)
        Path(self.path).write_bytes(b"invalid audio")
        with self.assertRaisesRegex(ValueError, "could not be read"):
            translation.translate_hindi_to_english(self.path)

    def test_decoded_audio_and_translation_settings(self):
        sf.write(self.path, np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.2, 16000)
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=" Hello "), SimpleNamespace(text=" world. ")]), None)
        with patch.object(translation, "_load_model", return_value=model):
            self.assertEqual(translation.translate_hindi_to_english(self.path), "Hello world.")
        args, kwargs = model.transcribe.call_args
        self.assertIsInstance(args[0], np.ndarray)
        self.assertEqual(kwargs, {"language": "hi", "task": "translate", "vad_filter": True,
                                  "condition_on_previous_text": False})

    def test_empty_vad_result_has_distinct_exception(self):
        sf.write(self.path, np.ones(16000, dtype=np.float32) * 0.1, 16000)
        model = Mock()
        model.transcribe.return_value = (iter([]), None)
        with patch.object(translation, "_load_model", return_value=model):
            with self.assertRaises(translation.NoSpeechDetectedError):
                translation.translate_hindi_to_english(self.path)


class LiveStreamTests(unittest.TestCase):
    def setUp(self):
        self.reference_calls = []
        self.released = threading.Event()

        @contextmanager
        def prepare(paths):
            self.reference_calls.append(list(paths))
            try:
                yield paths
            finally:
                self.released.set()

        self.model = Mock()
        self.model.inference_stream.return_value = iter([np.ones(4800) * 0.1, np.ones(7000) * 0.2])
        self.tts = SimpleNamespace(synthesizer=SimpleNamespace(tts_model=self.model))
        self.lock = threading.RLock()
        for target, value in (("prepare_references", prepare), ("get_tts", Mock(return_value=self.tts)),
                              ("get_conditioning", Mock(return_value=(None, None))), ("TTS_LOCK", self.lock)):
            patched = patch.object(live, target, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.audio = (16000, np.ones((2, 16000), dtype=np.int16) * 8192)

    def generator(self, reference=None):
        return live.live_translate_speak(self.audio, reference or ["primary.wav"], "second.wav", True, True)

    def test_chunks_and_temp_cleanup_and_reference_list_unchanged(self):
        paths = []

        def translate(path):
            paths.append(path)
            audio, rate = sf.read(path)
            self.assertEqual(rate, 16000)
            np.testing.assert_allclose(audio, 0.25)
            return "Hello world."

        refs = ["primary.wav"]
        with patch.object(live, "translate_hindi_to_english", side_effect=translate):
            chunks = list(self.generator(refs))
        self.assertEqual(refs, ["primary.wav"])
        self.assertEqual(self.reference_calls, [["primary.wav", "second.wav"]])
        self.assertTrue(all(not Path(path).exists() for path in paths))
        self.assertEqual([part.shape for _, part in chunks], [(1, 4800), (1, 4800), (1, 2200)])
        self.assertTrue(all(rate == 24000 for rate, _ in chunks))
        kwargs = self.model.inference_stream.call_args.kwargs
        self.assertTrue(kwargs["enable_text_splitting"])
        self.assertEqual(kwargs["repetition_penalty"], 5.0)

    def test_vad_false_positive_does_not_end_call_with_error(self):
        with patch.object(live, "translate_hindi_to_english", side_effect=translation.NoSpeechDetectedError("silence")):
            self.assertEqual(list(self.generator()), [])
        self.assertEqual(self.reference_calls, [])

    def test_worker_errors_reach_caller(self):
        self.model.inference_stream.side_effect = RuntimeError("model failed")
        with patch.object(live, "translate_hindi_to_english", return_value="Hello."):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                list(self.generator())
        self.assertTrue(self.released.wait(1))

    def test_cross_thread_cancellation_releases_model_lock(self):
        self.model.inference_stream.return_value = (np.ones(4800) * 0.1 for _ in range(100))
        with patch.object(live, "translate_hindi_to_english", return_value="Hello."):
            generator = self.generator()
            reader = threading.Thread(target=lambda: next(generator))
            reader.start()
            reader.join(timeout=3)
            self.assertFalse(reader.is_alive())
            generator.close()  # A different executor thread may close it in FastRTC.
            self.assertTrue(self.released.wait(3))
            self.assertTrue(self.lock.acquire(timeout=1))
            self.lock.release()


if __name__ == "__main__":
    unittest.main()
