"""Local demo: consented reference-voice TTS and Hindi-to-English voice translation."""

from __future__ import annotations

import tempfile
import os
import time
from pathlib import Path

import gradio as gr
from fastrtc import ReplyOnPause, WebRTC

from voice_translation.live import allow_empty_microphone, live_translate_speak
from voice_translation.synthesizer import get_tts, inspect_reference, prepare_references, synthesize_to_file
from voice_translation.translation import translate_hindi_to_english
from voice_translation.quality import compare_voice, format_comparison

# Prefer the locally verified research checkpoint, while retaining explicit
# environment overrides and a reproducible pretrained comparison mode.
adapted_dir = Path(__file__).resolve().parent / "models/xtts-india-south-asia-adapted/inference"
if (os.getenv("VOICE_USE_BASELINE") != "1"
        and not os.getenv("VOICE_TTS_MODEL_PATH")
        and not os.getenv("VOICE_TTS_CONFIG_PATH")
        and all((adapted_dir / name).is_file() for name in ("model.pth", "config.json", "vocab.json"))):
    os.environ["VOICE_TTS_MODEL_PATH"] = str(adapted_dir / "model.pth")
    os.environ["VOICE_TTS_CONFIG_PATH"] = str(adapted_dir / "config.json")
    os.environ.setdefault("VOICE_TTS_VOCAB_PATH", str(adapted_dir / "vocab.json"))


def require_consent(
    voice_reference: str | None,
    extra_reference: str | None,
    consent: bool,
    model_license: bool,
):
    if not consent:
        raise gr.Error("Confirm that this is your voice or that you have the speaker's permission.")
    if not voice_reference:
        raise gr.Error("Upload a clean voice reference recording first.")
    if not model_license:
        raise gr.Error("Review and accept the non-commercial XTTS-v2 license below before generating speech.")
    return [voice_reference] + ([extra_reference] if extra_reference else [])


def compare_recordings(reference, generated, consent, model_license):
    require_consent(reference, None, consent, model_license)
    try:
        return format_comparison(compare_voice(reference, generated))
    except Exception as exc:
        raise gr.Error(f"Voice comparison failed: {exc}") from exc


def generate_typed(
    text: str,
    language: str,
    voice_reference: str,
    extra_reference: str | None,
    consent: bool,
    model_license: bool,
    temperature: float,
    seed: int = 42,
):
    reference = require_consent(voice_reference, extra_reference, consent, model_license)
    if not text or not text.strip():
        raise gr.Error("Enter the text you want spoken.")
    with tempfile.NamedTemporaryFile(prefix="voice-translation-", suffix=".wav", delete=False) as temp_file:
        output_path = temp_file.name
    try:
        with prepare_references(reference) as prepared_reference:
            synthesize_to_file(
                get_tts(license_accepted=True),
                text=text.strip(),
                speaker_reference=prepared_reference,
                language=language,
                file_path=output_path,
                temperature=temperature,
                seed=seed,
            )
        return output_path
    except Exception as exc:
        Path(output_path).unlink(missing_ok=True)
        raise gr.Error(f"Speech generation failed: {exc}") from exc


def translate_and_speak(
    audio_path: str,
    voice_reference: str,
    extra_reference: str | None,
    consent: bool,
    model_license: bool,
    temperature: float,
    seed: int = 42,
):
    reference = require_consent(voice_reference, extra_reference, consent, model_license)
    if not audio_path:
        raise gr.Error("Record or upload Hindi speech first.")
    try:
        translated_text = translate_hindi_to_english(audio_path)
    except Exception as exc:
        raise gr.Error(f"Translation failed: {exc}") from exc
    with tempfile.NamedTemporaryFile(prefix="translated-voice-", suffix=".wav", delete=False) as temp_file:
        output_path = temp_file.name
    try:
        with prepare_references(reference) as prepared_reference:
            synthesize_to_file(
                get_tts(license_accepted=True),
                text=translated_text,
                speaker_reference=prepared_reference,
                language="en",
                file_path=output_path,
                temperature=temperature,
                seed=seed,
            )
        return translated_text, output_path
    except Exception as exc:
        Path(output_path).unlink(missing_ok=True)
        raise gr.Error(f"Translated speech generation failed: {exc}") from exc


def check_references(reference, extra):
    if not reference:
        raise gr.Error("Record or upload a reference first.")
    lines = []
    try:
        for index, path in enumerate([reference] + ([extra] if extra else []), 1):
            report = inspect_reference(path)
            lines.append(
                f"**Clip {index}:** {report['duration_seconds']:.1f}s after edge trimming; "
                f"{report['active_seconds']:.1f}s above the energy threshold; "
                f"average level {report['rms_dbfs']:.1f} dBFS."
            )
            if report["active_seconds"] < 10:
                lines.append("A longer clean recording may give the model more pronunciation examples.")
            if report["rms_dbfs"] < -35:
                lines.append("This clip is quiet. Move closer to the microphone; avoid digitally boosting background noise.")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    lines.append("These level checks cannot verify the speaker, accent, background noise, or voice similarity. Listen to each clip before generating.")
    return "\n\n".join(lines)


def generate_with_report(*args):
    started = time.perf_counter()
    path = generate_typed(*args)
    return path, f"Generated in {time.perf_counter() - started:.1f}s · seed {int(args[-1])} · native 24 kHz audio. Compare pronunciation and delivery with your reference."


with gr.Blocks(title="Real-Time Voice Translation", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        """# Real-Time Voice Translation
        Clone a **consented** voice from a short reference recording. Generate speech from typed text,
        or translate a Hindi recording into English speech in the reference voice. Processing is local
        after model files are downloaded. The live tab translates completed phrases as you speak, after a pause."""
    )
    reference = gr.Audio(label="Voice reference (your voice or a voice you have permission to use)", type="filepath")
    with gr.Accordion("Optional second reference from the same speaker", open=False):
        extra_reference = gr.File(
            label="Second voice reference (same speaker, clean speech)",
            file_count="single", file_types=["audio"], type="filepath",
        )
    gr.Markdown(
        "For Indian-accented English, use 15–30 seconds of clear English spoken in the Indian accent you want. "
        "For Hindi output, use a Hindi reference. Keep one speaker, speak naturally, and avoid music, effects, "
        "or background noise. Use the same speaker and accent in both clips. XTTS can copy voice timbre across "
        "languages, but it may not preserve accent or prosody."
    )
    with gr.Accordion("Check the recording and get reference tips", open=False):
        gr.Markdown(
            "Record in a quiet room, at a steady microphone distance, in the delivery you want to hear. "
            "For English, read a few English sentences naturally. Avoid whispering or exaggerated acting unless that is the style you want. "
            "For the first comparison, type the same sentence you recorded. Use the same text, reference and seed when comparing settings. "
            "A second clip should have the same speaker and delivery; both clips share a 30-second conditioning window."
        )
        check_button = gr.Button("Check reference recording")
        reference_report = gr.Markdown()
        check_button.click(check_references, [reference, extra_reference], reference_report)
    consent = gr.Checkbox(label="I own this voice or have the speaker's explicit permission to generate speech with it.")
    model_license = gr.Checkbox(
        label="I have read and agree to the non-commercial Coqui Public Model License (CPML): https://coqui.ai/cpml"
    )
    with gr.Accordion("Generation settings", open=False):
        temperature = gr.Slider(
            minimum=0.55, maximum=0.95, value=0.75, step=0.05,
            label="Sampling temperature (default 0.75)",
        )
        seed = gr.Number(value=42, precision=0, minimum=0, maximum=4294967295,
                         label="Generation seed (keep fixed for comparisons; change for another take)")
    gr.Markdown(
        "**Configured model:** " + (
            "adapted/custom XTTS checkpoint. Accent and voice quality need listening comparison."
            if os.getenv("VOICE_TTS_MODEL_PATH") else
            "pretrained XTTS-v2 baseline."
        )
    )

    with gr.Tab("Text → voice"):
        text = gr.Textbox(label="Text", lines=4, placeholder="Type what you want the voice to say…")
        language = gr.Dropdown(choices=[("English", "en"), ("Hindi", "hi")], value="en", label="Speech language")
        typed_button = gr.Button("Generate speech", variant="primary")
        typed_audio = gr.Audio(label="Generated speech", type="filepath", interactive=False)
        generation_report = gr.Markdown()
        with gr.Accordion("Compare reference and generated voice", open=False):
            gr.Markdown(
                "Use the same sentence in your reference and generated speech for a fair delivery comparison. "
                "The local speaker score cannot certify an exact voice, accent or naturalness match."
            )
            compare_button = gr.Button("Compare these recordings")
            comparison_report = gr.Markdown()
            compare_button.click(
                compare_recordings, [reference, typed_audio, consent, model_license], comparison_report,
                concurrency_id="speech-generation", concurrency_limit=1,
            )
        typed_button.click(
            generate_with_report,
            [text, language, reference, extra_reference, consent, model_license, temperature, seed],
            [typed_audio, generation_report],
            concurrency_id="speech-generation", concurrency_limit=1,
        )

    with gr.Tab("Hindi → English"):
        source_audio = gr.Audio(label="Hindi speech to translate", sources=["microphone", "upload"], type="filepath")
        translate_button = gr.Button("Translate and speak", variant="primary")
        translated_text = gr.Textbox(label="English translation")
        translated_audio = gr.Audio(label="English speech in the reference voice", type="filepath")
        translate_button.click(
            translate_and_speak,
            [source_audio, reference, extra_reference, consent, model_license, temperature, seed],
            [translated_text, translated_audio],
            concurrency_id="speech-generation", concurrency_limit=1,
        )

    with gr.Tab("Live translation"):
        gr.Markdown("Speak Hindi; after a short pause, the phrase is translated and English audio starts playing in the selected voice. This is streaming microphone input with pause-based phrase translation, not word-by-word simultaneous interpretation.")
        live_audio = allow_empty_microphone(WebRTC(
            label="Live microphone", mode="send-receive", modality="audio",
            full_screen=False, height=240,
        ))
        async def reset_live_microphone():
            from voice_translation.live import reset_microphone_connections
            return await reset_microphone_connections(live_audio)

        reset_button = gr.Button("Reset microphone connection")
        live_status = gr.Markdown()
        reset_button.click(reset_live_microphone, inputs=[], outputs=live_status, queue=False)
        live_audio.stream(
            fn=ReplyOnPause(live_translate_speak, can_interrupt=True),
            inputs=[live_audio, reference, extra_reference, consent, model_license, temperature],
            outputs=[live_audio],
            time_limit=120,
        )

    gr.Markdown(
        "Model note: XTTS-v2 is distributed under the Coqui Public Model License, which is non-commercial. "
        "Check the model and dataset licenses before any use beyond academic research."
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860,
                        inbrowser=os.getenv("VOICE_OPEN_BROWSER") == "1")
