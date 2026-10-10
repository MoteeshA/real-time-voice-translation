# Real-Time Voice Translation

A local research prototype with three modes:

1. **Type text → cloned speech.** Enter Hindi or English text and generate audio conditioned on an uploaded voice reference.
2. **Hindi speech → English cloned speech.** Record or upload a Hindi phrase, translate it to English, then speak the translation in the same reference voice.
3. **Live phrase translation.** Speak Hindi and receive streamed English speech after a pause.

**Current status:** the local prototype runs and a one-epoch public-data XTTS accent adaptation pilot completed on a Kaggle GPU. The pilot used 629 training clips from 15 speakers and 192 validation clips from 4 other speakers, with regional labels including India, Pakistan and Sri Lanka. Inference and regression checks passed; exact voice, accent, tone and naturalness matching remain unproven. Model weights and personal recordings are excluded from this repository. See [the pilot results](docs/GPU_PILOT_RESULTS.md).

The live tab streams microphone audio and replies after each detected pause; it is phrase-based rather than word-by-word simultaneous interpretation. Model files download on first use. The app runs on `127.0.0.1` and requires confirmation that the reference voice is yours or that you have permission to use it. It also displays a separate XTTS-v2 non-commercial CPML agreement checkbox; review the linked terms and check it only if you agree. This explicit app confirmation avoids Coqui's interactive license prompt failing with `EOFError` in the browser.

## Run the app

Use Python 3.10–3.13. On the current Apple Silicon development machine, Python 3.12 is available in the Codex runtime. From a Python 3.12 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice]'
python app.py
```

On first use after accepting the in-app CPML checkbox, XTTS-v2 downloads its model checkpoint and faster-whisper downloads the `small` ASR model. The defaults use CPU for compatibility. For Whisper, configure a smaller/faster model with `VOICE_WHISPER_MODEL=base` or `tiny` if needed.

For a fine-tuned XTTS checkpoint, set `VOICE_TTS_MODEL_PATH` and `VOICE_TTS_CONFIG_PATH` to its inference model and original XTTS config. Keep the matching `vocab.json` beside the model, or set `VOICE_TTS_VOCAB_PATH`. The loader uses XTTS's explicit checkpoint API. See [docs/FINETUNING.md](docs/FINETUNING.md).

The local app automatically selects the verified checkpoint in `models/xtts-india-south-asia-adapted/inference` when those three files are present. Explicit model environment settings take precedence. Set `VOICE_USE_BASELINE=1` before starting a fresh server to compare the pretrained model. The source ZIP does not contain the model weights.

The GPU pilot completed on 9 October 2026. Training, validation, artifact checksums, local inference and the typed/translation APIs passed. Accent, speaker identity and tone improvement have not been established by listening evaluation. See [the pilot results](docs/GPU_PILOT_RESULTS.md).

The interface checks reference duration, amplitude and clipping. It shares cached voice conditioning between typed and live speech, uses the same released-model sampling settings in both paths, and lets you keep a fixed seed for repeatable typed comparisons. These are reliability improvements, not evidence of perceptual similarity or accent improvement. Run regression checks with `python -m unittest discover -s tests -v`.

## Research question

Can adapting a multilingual speech model on licensed, multi-speaker Indian-English audio improve accent and prosody while short reference clips continue to condition each speaker's identity?

## Research setup

- Curate a licensed, transcribed, multi-speaker Indian-English dataset with audio, speaker IDs, and evidence for accent labels.
- Split by speaker identity; hold out speakers entirely from training and tuning.
- Compare untouched XTTS with a small multi-speaker accent-adaptation run, using short held-out speaker references at inference.
- Treat recording the target user for speaker-specific fine-tuning as a separate optional experiment; it is not needed for accent training or reference-based zero-shot cloning.
- Measure translation quality, intelligibility, speaker match, accent/prosody match, naturalness, and latency. Do not claim exact cloning from embedding scores alone.

The app accepts two same-speaker reference clips and uses XTTS sampling temperature 0.75 by default. XTTS does not provide a dedicated Indian-English accent control; a short reference alone cannot promise exact pitch, tone, accent, and prosody on arbitrary text. The primary research plan is public multi-speaker accent adaptation; target-speaker recording for training is optional and separate. Completing that experiment requires a compatible training environment and held-out listening evaluation.

## Model and papers

- XTTS-v2 is the implementation: its paper describes multilingual zero-shot TTS and upstream docs describe GPT-encoder fine-tuning. Its Coqui Public Model License is non-commercial. CPU inference works on this Mac; full CPU fine-tuning exceeded its practical memory budget. [Paper](https://arxiv.org/abs/2406.04904) · [upstream fine-tuning docs](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md#training)
- OpenVoice V2 is a comparison candidate because its paper studies cross-lingual tone-color cloning and controllable accent, rhythm, pauses, and intonation; upstream documentation lists English among its native V2 languages. It is not a drop-in replacement or a guarantee of exact matching. Check the license of its code, checkpoint, base TTS, vocoder, and every other component before use. [Paper](https://arxiv.org/abs/2312.01479) · [official repository and usage](https://github.com/myshell-ai/OpenVoice)
- Seamless is related work for expressive and streaming speech translation. [Paper](https://arxiv.org/abs/2312.05187)

XTTS-v2 is licensed under the Coqui Public Model License, which restricts commercial use. Read the model and dataset terms before sharing checkpoints or generated audio.

## Project structure

- `app.py`: local Gradio interface.
- `src/voice_translation/translation.py`: Hindi → English Whisper translation adapter.
- `src/voice_translation/pipeline.py`: model-independent pipeline contracts.
- `scripts/prepare_xtts_dataset.py`: consent-checked, speaker-disjoint fine-tuning manifest preparation.
- `scripts/prepare_speaker_adaptation.py`: validates one speaker's consented WAV recordings and creates session-disjoint adaptation/evaluation lists.
- `docs/ACCENT_ADAPTATION.md`: primary public-data accent-training workflow.
- `docs/VOICE_RECORDING_PROTOCOL.md`: optional target-speaker fine-tuning guide; not required for accent adaptation.
- `scripts/download_vctk_subset.py`: downloads a small CC BY 4.0 VCTK subset for a reproducible research pilot.
- `docs/PROJECT_PLAN.md`: research plan and milestones.
- `docs/EXPERIMENTS.md`: controlled experiment matrix and measures.
- `docs/FINETUNING.md`: training data format and upstream fine-tuning procedure.

## Local voice comparison

After generating typed speech, open **Compare reference and generated voice** and click **Compare these recordings**. The independent WavLM speaker model runs locally; its public weights download once on first use. No recording is sent to a remote inference service. Analysis is bounded to the first 15 seconds per clip.

Use the same sentence when comparing pitch and timing. Cosine scores are not percentages, universal pass/fail thresholds, or proof of exact identity. Accent, emotional delivery and naturalness still require listening. The local controlled experiments favoured the adapted XTTS backend over pretrained XTTS; the experimental OpenVoice conversion stage was rejected because it did not improve the independent score and introduced pitch trade-offs. Personal comparison clips and reports are kept outside the source/training packages.
