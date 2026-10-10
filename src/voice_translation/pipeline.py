"""Model-agnostic interfaces for the speech translation pipeline."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranslationResult:
    source_text: str
    translated_text: str
    source_language: str
    target_language: str


@dataclass(frozen=True)
class SpeechChunk:
    audio: bytes
    sample_rate: int
    is_final: bool = False


class Translator(Protocol):
    def translate(self, audio: bytes, source_language: str, target_language: str) -> TranslationResult:
        """Transcribe and translate one finalized speech segment."""


class VoiceSynthesizer(Protocol):
    def synthesize(self, text: str, language: str, speaker_reference: bytes) -> list[SpeechChunk]:
        """Generate target-language speech conditioned on a consented reference."""
