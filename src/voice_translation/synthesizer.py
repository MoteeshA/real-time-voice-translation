"""XTTS reference validation, cached voice conditioning, and serialized synthesis."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf

TTS_LOCK = threading.RLock()
XTTS_CONDITIONING = {"gpt_cond_len": 30, "gpt_cond_chunk_len": 4,
                     "max_ref_length": 30, "sound_norm_refs": False}
# Match the released XTTS config (the low-level API defaults differ).
XTTS_GENERATION = {"length_penalty": 1.0, "repetition_penalty": 5.0,
                   "top_k": 50, "top_p": 0.85, "speed": 1.0,
                   "enable_text_splitting": True}
_CONDITIONING_CACHE = OrderedDict()


def _reference_audio(path):
    if not path or not Path(path).is_file():
        raise ValueError("Select an existing voice recording first.")
    # Bound decoding: reference material beyond 60 seconds is not useful here.
    audio, sr = librosa.load(path, sr=22050, mono=True, duration=61)
    if audio.size > 60 * sr:
        raise ValueError("Use a reference clip of 60 seconds or less; 15–30 seconds is a useful starting point.")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("This reference is empty or contains invalid audio samples.")
    peak = float(np.max(np.abs(audio)))
    if peak < 0.001:
        raise ValueError("The reference is silent or too quiet. Record closer to the microphone.")
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    if clipped > 0.01:
        raise ValueError("The reference is distorted by clipping. Lower the microphone gain and record again.")
    trimmed, _ = librosa.effects.trim(audio, top_db=35, frame_length=1024, hop_length=256)
    intervals = librosa.effects.split(trimmed, top_db=30, frame_length=1024, hop_length=256)
    active_seconds = float(sum(end - start for start, end in intervals) / sr)
    # This is an energy check, not a speech recognizer or a speaker verifier.
    if active_seconds < 3:
        raise ValueError("Reference needs at least 3 seconds of audible material; record 15–30 seconds of clean speech.")
    rms = float(np.sqrt(np.mean(trimmed ** 2)))
    return trimmed, sr, {"duration_seconds": len(trimmed) / sr,
                         "active_seconds": active_seconds,
                         "peak_dbfs": 20 * np.log10(max(peak, 1e-10)),
                         "rms_dbfs": 20 * np.log10(max(rms, 1e-10)),
                         "clipped_fraction": clipped}


def inspect_reference(path):
    """Return measured reference levels; these cannot certify accent or similarity."""
    return _reference_audio(path)[2]


@contextmanager
def prepare_references(reference_paths):
    paths = list(reference_paths) if isinstance(reference_paths, (list, tuple)) else [reference_paths]
    if not 1 <= len(paths) <= 2:
        raise ValueError("Choose one or two recordings of the same speaker.")
    with tempfile.TemporaryDirectory(prefix="voice-reference-") as workdir:
        prepared = []
        for index, path in enumerate(paths):
            audio, sr, _ = _reference_audio(path)
            # Give both references a share of the 30-second conditioning window.
            audio = audio[:int(30 / len(paths) * sr)]
            peak = float(np.max(np.abs(audio)))
            # Avoid huge gain boosts that exaggerate noise in quiet references.
            audio = audio * min(4.0, 0.9 / peak)
            destination = Path(workdir) / f"reference-{index}.wav"
            sf.write(destination, audio, sr, subtype="PCM_16")
            prepared.append(str(destination))
        yield prepared


def get_conditioning(tts, prepared_reference_paths):
    """Cache ephemeral conditioning in RAM, keyed by audio content and model."""
    paths = list(prepared_reference_paths)
    digest = hashlib.sha256()
    for path in paths:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    key = (id(tts), digest.hexdigest())
    with TTS_LOCK:
        if key not in _CONDITIONING_CACHE:
            _CONDITIONING_CACHE[key] = tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=paths, **XTTS_CONDITIONING)
            if len(_CONDITIONING_CACHE) > 4:
                _CONDITIONING_CACHE.popitem(last=False)
        _CONDITIONING_CACHE.move_to_end(key)
        return _CONDITIONING_CACHE[key]


def synthesize_to_file(tts, *, text, speaker_reference, language, file_path,
                       temperature=0.75, seed=42):
    """Use native 24 kHz audio and a repeatable seed for listening comparisons."""
    text = text.strip()
    if not text or len(text) > 2000:
        raise ValueError("Enter between 1 and 2,000 characters; split longer passages into parts.")
    if language not in {"en", "hi"}:
        raise ValueError("Choose English or Hindi.")
    if not np.isfinite(temperature) or not 0.55 <= temperature <= 0.95:
        raise ValueError("Temperature must be between 0.55 and 0.95.")
    if seed is None or not np.isfinite(seed) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be a whole number between 0 and 4294967295.")
    import torch
    with TTS_LOCK:
        model = tts.synthesizer.tts_model
        conditioning, embedding = get_conditioning(tts, speaker_reference)
        devices = [model.device.index or 0] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            result = model.inference(text, language, conditioning, embedding,
                                     temperature=float(temperature), **XTTS_GENERATION)
        wav = np.asarray(result["wav"], dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
            raise RuntimeError("The model returned invalid or silent audio. Try a clearer reference or a new seed.")
        # Only attenuate overshoots; do not boost every utterance to full scale.
        peak = float(np.max(np.abs(wav)))
        if peak > 0.98:
            wav *= 0.98 / peak
        sf.write(file_path, wav, 24000, subtype="PCM_16")
    return str(file_path)


@lru_cache(maxsize=1)
def _load_tts():
    import torch
    from TTS.api import TTS
    try:
        threads = int(os.getenv("VOICE_TORCH_THREADS", "2"))
    except ValueError as exc:
        raise RuntimeError("VOICE_TORCH_THREADS must be a positive integer.") from exc
    if threads < 1:
        raise RuntimeError("VOICE_TORCH_THREADS must be a positive integer.")
    torch.set_num_threads(threads)
    device = os.getenv("VOICE_TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("VOICE_TTS_DEVICE must be cpu or cuda; this app has validated CPU inference only on this Mac.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no compatible GPU is available.")
    model_path, config_path = os.getenv("VOICE_TTS_MODEL_PATH"), os.getenv("VOICE_TTS_CONFIG_PATH")
    if bool(model_path) != bool(config_path):
        raise RuntimeError("Set both VOICE_TTS_MODEL_PATH and VOICE_TTS_CONFIG_PATH.")
    if model_path:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        vocab_path = os.getenv("VOICE_TTS_VOCAB_PATH", str(Path(model_path).parent / "vocab.json"))
        for path in (model_path, config_path, vocab_path):
            if not Path(path).is_file():
                raise RuntimeError(f"Missing XTTS inference file: {path}")
        config = XttsConfig()
        config.load_json(config_path)
        if config.model != "xtts":
            raise RuntimeError("Use the original XTTS inference config.json, not the GPT training run config.")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=vocab_path, eval=True)
        model.to(device)
        return SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model, output_sample_rate=24000))
    return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
               progress_bar=False).to(device)


def get_tts(license_accepted=False):
    if not license_accepted:
        raise RuntimeError("Review and accept the XTTS-v2 non-commercial model license in the app first.")
    os.environ["COQUI_TOS_AGREED"] = "1"
    # lru_cache alone can load twice when live and typed requests arrive together.
    with TTS_LOCK:
        return _load_tts()
