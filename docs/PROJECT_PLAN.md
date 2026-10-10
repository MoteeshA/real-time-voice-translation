# Project plan

## Goal

Build a real-time speech translation prototype that generates translated English in a consenting speaker's voice, then test whether multi-speaker Indian-English accent adaptation improves accent delivery and naturalness while short runtime references preserve speaker identity.

## Core distinction

- **Zero-shot voice cloning:** the evaluation speaker is unseen; a short audio sample provides conditioning, with no speaker-specific weight update.
- **Base-model fine-tuning:** train a component on a licensed multi-speaker corpus, then evaluate on held-out speakers. This can remain zero-shot with respect to those speakers.
- **Speaker adaptation:** update weights or fit speaker-specific parameters from the live speaker. This is a separate setting and must not be described as zero-shot.
- **Progressive reference conditioning:** as the conversation provides more clean audio, select/update the speaker reference or cached embedding without updating model weights.

The primary research track uses openly licensed, multi-speaker Indian-English data and holds speakers out for evaluation. A short consented reference clip at inference conditions the target speaker identity; the target user need not be included in training. A separate target-speaker fine-tuning track is optional and should not be confused with accent training. Per-user online weight training while speaking is not part of the initial study.

## Proposed architecture

1. Capture microphone audio in short overlapping chunks and apply voice activity detection.
2. Stabilize a phrase boundary before translation to avoid translating incomplete clauses.
3. Transcribe and translate the phrase using a speech translation or ASR + machine translation component.
4. Synthesize target-language audio conditioned on a quality-filtered speaker reference.
5. Stream generated chunks to playback; record time-to-first-audio and total phrase delay.
6. Update the cached speaker reference only from clean, sufficiently long user speech. Never feed generated audio back into the reference pool.

A cascaded baseline is easier to inspect and measure than a single end-to-end model. Begin with recorded utterances; add live microphone streaming after the offline quality pipeline works.

## Accent adaptation plan

1. Select a licensed, transcribed multi-speaker Indian-English dataset with usable accent/source metadata. Do not use VCTK as an Indian-English dataset.
2. Split by speaker identity so evaluation voices never appear in training or tuning.
3. Compare the unchanged XTTS checkpoint with a small multi-speaker accent-adaptation run. Keep each test speaker's short runtime reference fixed across both conditions.
4. Fine-tune only documented components with a smoke test; keep the untouched base checkpoint and training logs. Monitor speaker similarity, intelligibility, and overfitting.
5. Evaluate on fixed held-out English text using blinded listener ratings for accent and naturalness, plus intelligibility and speaker-similarity measures. Then connect to Hindi→English translation with the translated text fixed for TTS comparisons.
6. Treat an optional target-speaker fine-tune as a separate ablation, not as a prerequisite for voice cloning or accent training.
7. Report dataset amount, language/accent, speaker split, compute, filtering, hyperparameters, checkpoint, and licensing.

The installed XTTS-v2 model supports English and Hindi generation; both are smoke-tested in this project. XTTS generates speech from text and does not translate incoming speech. Faster-whisper provides Hindi-to-English speech translation. Evaluate translation accuracy separately from voice synthesis.

## Milestones

### M0 — Reproducible baseline
- Choose language direction, dataset, compute budget, and checkpoint licenses.
- Run transcription/translation and voice generation on a small consented or licensed sample.
- Save timings and outputs with fixed test prompts.

### M1 — Offline voice-conditioned translation
- Implement modular ASR/translation/TTS adapters.
- Generate translated speech using a fixed reference clip.
- Build a simple evaluation manifest and human listening protocol.

### M2 — Fine-tuning
- Prepare speaker-disjoint data splits and a reproducible training configuration.
- Fine-tune the selected trainable component and compare with the base checkpoint.

### M3 — Progressive reference conditioning
- Score reference segments for duration, clipping, noise, and speech confidence.
- Compare fixed initial audio against quality-aware updates from later user speech.
- Prevent output audio and other speakers from entering the reference cache.

### M4 — Streaming prototype and study
- Add chunking, phrase-finalization, playback queue, and latency logging.
- Evaluate quality/latency trade-offs and document limitations.

## Evaluation

Use the same utterances and held-out speakers for all system variants. Report:
- Translation quality: COMET or BLEU where language tooling supports it, plus human adequacy ratings.
- Intelligibility: ASR-derived WER/CER on generated speech, with a human check for ASR bias.
- Speaker similarity: a speaker-verification embedding similarity score, plus blinded human similarity ratings.
- Accent and prosody: blinded ratings for target-accent match; compare F0/pitch range, contour, pause duration, and speaking rate as secondary objective measures.
- Naturalness: blinded MOS-style ratings.
- Latency: time to first audio and end-of-phrase delay, median and 90th percentile.
- Operational measures: dropped/duplicated chunks and real-time factor.

## Safety and data handling

Use only the user's own voice or voices with explicit, documented permission. Obtain consent for recording and model adaptation; allow deletion of recordings and cached speaker representations. Label generated audio in demos. Keep a consented test set separate from training data.

## Current implementation status

The local Gradio app supports typed Hindi/English text to reference-voice speech, recorded Hindi-to-English translation, and a live microphone path that detects pauses, translates each completed phrase, and streams cloned English audio chunks. XTTS-v2 is the TTS candidate; faster-whisper supplies Hindi-to-English speech translation. Scripts prepare a speaker-disjoint multi-speaker pilot and a separate one-speaker adaptation dataset with session-disjoint evaluation. The recording protocol is ready, but no target-speaker recordings or fine-tuning run have been completed. This host has 8 GB unified memory, no CUDA GPU, and the Coqui trainer does not select Apple MPS for training.

## Immediate next actions

1. The GLOBE V2 pilot is prepared with 15 training and 4 validation speakers. Its label combines India, Pakistan and Sri Lanka; describe this limitation accurately.
2. Run the portable pilot on a CUDA GPU. The M2 CPU attempt loaded and started the trainer but was stopped for heavy memory pressure without a saved adapted checkpoint.
3. Compare unchanged and adapted checkpoints on fixed held-out English test text, with the same runtime reference.
4. Add/compare prosody guidance if the accent adaptation shows value; then evaluate pitch, accent, naturalness, speaker match, and intelligibility.
5. Integrate only the validated checkpoint and compare end-to-end latency and quality in offline and live modes.
