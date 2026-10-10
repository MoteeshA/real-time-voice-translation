"""Hindi-to-English transcription and translation adapter using Whisper."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np


WHISPER_LOCK = RLock()


class NoSpeechDetectedError(ValueError):
    """No intelligible speech was found; live calls can skip this phrase."""


@lru_cache(maxsize=1)
def _load_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install the optional speech dependencies with `pip install -e '.[voice]'`.") from exc
    model_name = os.getenv("VOICE_WHISPER_MODEL", "small")
    device = os.getenv("VOICE_WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("VOICE_WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def translate_hindi_to_english(audio_path: str) -> str:
    """Translate a Hindi recording into English text, then return the finalized phrase."""
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Record or upload a Hindi audio file first.")
    try:
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise RuntimeError("Install the optional speech dependencies with `pip install -e '.[voice]'`.") from exc
    try:
        samples = decode_audio(audio_path, sampling_rate=16000)
    except Exception as exc:
        raise ValueError("The recording could not be read. Try uploading a WAV, MP3, or M4A file.") from exc
    if not samples.size or np.max(np.abs(samples), initial=0) < 1e-4:
        raise NoSpeechDetectedError("No speech was detected. Try a clearer recording with less background noise.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("The recording contains invalid audio samples. Please record it again.")
    # Transcription is lazy: keep both loading and segment iteration inside the
    # lock so concurrent browser actions cannot load duplicate CPU models.
    with WHISPER_LOCK:
        segments, _ = _load_model().transcribe(
            samples,
            language="hi",
            task="translate",
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    if not text:
        raise NoSpeechDetectedError("No speech was detected. Try a clearer recording with less background noise.")
    return text
