"""Pause-triggered live Hindi-to-English speech translation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Iterator

import numpy as np
import soundfile as sf

from .synthesizer import TTS_LOCK, XTTS_GENERATION, get_conditioning, get_tts, prepare_references
from .translation import NoSpeechDetectedError, translate_hindi_to_english


OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHUNK_SAMPLES = OUTPUT_SAMPLE_RATE // 5


def allow_empty_microphone(component):
    """Allow settings changes before FastRTC has a microphone session ID."""
    from fastrtc.utils import WebRTCData

    original = component.preprocess

    def preprocess(payload):
        if payload is None:
            return None
        value = original(payload)
        # FastRTC duplicates legacy string session IDs when setting audio args.
        # Its structured session type is replaced by the captured audio instead.
        return WebRTCData(webrtc_id=value) if isinstance(value, str) else value

    # Retain the original component class and its frontend assets.
    component.preprocess = preprocess
    return component


async def reset_microphone_connections(component):
    """Release local microphone sessions so a stopped call can reconnect."""
    for session_id, peer in list(component.pcs.items()):
        await peer.close()
        connections = component.clean_up(session_id)
        for connection in connections:
            connection.stop()
    return "Microphone connections cleared. Click Record to start again."


def normalize_microphone_audio(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    """Convert channel-first/channel-last PCM to finite mono float32 audio.

    Scale integer PCM before averaging channels: averaging int16 first loses
    its dtype and otherwise makes the WAV writer clip almost every sample.
    """
    sample_rate, samples = audio
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("The microphone supplied an invalid sample rate.")
    samples = np.asarray(samples)
    if samples.ndim not in (1, 2):
        raise ValueError("The microphone must supply mono or stereo audio.")
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        if limits.min == 0:
            midpoint = (limits.max + 1) / 2
            samples = (samples.astype(np.float32) - midpoint) / midpoint
        else:
            samples = samples.astype(np.float32) / max(-limits.min, limits.max)
    elif np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    else:
        raise ValueError("The microphone supplied an unsupported audio format.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("The microphone supplied invalid audio samples. Reconnect it and try again.")
    if samples.ndim == 2:
        if samples.shape[0] <= 2:
            samples = samples.mean(axis=0)
        elif samples.shape[1] <= 2:
            samples = samples.mean(axis=1)
        else:
            raise ValueError("The microphone must supply mono or stereo audio.")
    return int(sample_rate), np.clip(samples.reshape(-1), -1.0, 1.0)


def live_translate_speak(
    audio: tuple[int, np.ndarray],
    voice_reference: str | list[str] | None = None,
    extra_reference: str | None = None,
    consent: bool = False,
    license_accepted: bool = False,
    temperature: float = 0.75,
) -> Iterator[tuple[int, np.ndarray]]:
    """Translate one VAD-delimited Hindi phrase and stream its English speech.

    FastRTC's pause detector supplies one completed utterance at a time. XTTS's
    streaming inference then yields audio chunks as generation proceeds.
    """
    if not consent:
        raise ValueError("Confirm that you own the reference voice or have the speaker's permission.")
    if not license_accepted:
        raise ValueError("Review and accept the non-commercial Coqui XTTS license before using this model.")
    if not voice_reference:
        raise ValueError("Upload a clean, consented voice reference before starting the microphone.")
    reference_paths = list(voice_reference) if isinstance(voice_reference, list) else [voice_reference]
    if extra_reference:
        reference_paths.append(extra_reference)

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling temperature must be a positive number.")
    # Structured FastRTC sessions carry the VAD-delimited audio payload.
    if hasattr(audio, "webrtc_id"):
        audio = audio.audio
    sample_rate, samples = normalize_microphone_audio(audio)
    if samples.size < sample_rate // 2 or np.max(np.abs(samples), initial=0) < 1e-4:
        return

    with tempfile.NamedTemporaryFile(prefix="live-hindi-", suffix=".wav", delete=False) as wav_file:
        input_path = Path(wav_file.name)
    try:
        sf.write(input_path, samples, sample_rate)
        try:
            english_text = translate_hindi_to_english(str(input_path))
        except NoSpeechDetectedError:
            # VAD can fire on a breath or background sound; keep the call alive.
            return
    finally:
        input_path.unlink(missing_ok=True)

    # FastRTC may resume/close this generator on different executor threads.
    # A dedicated producer owns the model RLock for the complete utterance;
    # yielding while holding it here could strand it on another thread.
    queue: Queue[np.ndarray | Exception | None] = Queue(maxsize=8)
    cancelled = Event()

    def send(item: np.ndarray | Exception | None) -> bool:
        while not cancelled.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce() -> None:
        try:
            with prepare_references(reference_paths) as prepared_reference, TTS_LOCK:
                if cancelled.is_set():
                    return
                tts = get_tts(license_accepted=True)
                model = tts.synthesizer.tts_model
                conditioning, speaker_embedding = get_conditioning(tts, prepared_reference)
                pending = np.empty((1, 0), dtype=np.float32)
                for chunk in model.inference_stream(
                    english_text,
                    "en",
                    conditioning,
                    speaker_embedding,
                    temperature=temperature,
                    **XTTS_GENERATION,
                ):
                    if cancelled.is_set():
                        return
                    if hasattr(chunk, "detach"):
                        chunk = chunk.detach().cpu().numpy()
                    chunk = np.asarray(chunk, dtype=np.float32).reshape(1, -1)
                    if not np.all(np.isfinite(chunk)):
                        raise RuntimeError("Speech generation returned invalid audio. Try a cleaner voice reference.")
                    chunk = np.clip(chunk, -1.0, 1.0)
                    pending = np.concatenate((pending, chunk), axis=1)
                    while pending.shape[1] >= OUTPUT_CHUNK_SAMPLES:
                        if not send(pending[:, :OUTPUT_CHUNK_SAMPLES].copy()):
                            return
                        pending = pending[:, OUTPUT_CHUNK_SAMPLES:]
                if pending.shape[1]:
                    send(pending)
        except Exception as exc:
            send(exc)
        finally:
            send(None)

    worker = Thread(target=produce, name="live-xtts", daemon=True)
    worker.start()
    try:
        while True:
            item = queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield OUTPUT_SAMPLE_RATE, item
    finally:
        cancelled.set()
