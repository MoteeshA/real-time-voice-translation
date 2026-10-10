# GPU pilot results — 9 October 2026

Run `20261009T062346Z-34787293` completed the original Coqui XTTS GPT fine-tuning configuration on a Tesla T4. No selective layer freezing was added. The run started from the verified partial CPU checkpoint at batch 100, with a new optimizer, then performed one full epoch. Batch size was 1, accumulation was 8, and learning rate was 0.000005.

The pinned public GLOBE V2 selection contains 629 training clips from 15 speakers and 192 validation clips from four different speakers. The accent labels include India, Pakistan and Sri Lanka; this is a South Asian English pilot, not an India-only dataset. No personal voice recordings were uploaded.

Validation finished with finite averages: text cross-entropy 0.0232153, audio-token cross-entropy 3.9022563 and total loss 3.9254715. These are model losses, not perceptual voice-similarity scores. Separate final-test speakers and listening evaluation are still required.

The inference export removed optimizer state and training-only parameters and retained the original model precision. Its archive SHA-256 is `d22f6b470b794ae1dfb2235e928ddaf4b8b38b55abfb91ad0b99019259782f0c`. Model SHA-256 is `55ce63e5260da5401a46227794fc012d75be51510c1eb365372461ecb2c3361c`. The archive and all three inference files were verified after recovery to the Mac.

A local public-reference smoke test produced finite 24 kHz speech. Whisper-small transcribed both pretrained and adapted versions of one English sentence exactly: zero word errors for that sentence. This does not demonstrate an accent or tone improvement. The reference speaker belongs to the validation selection, so this comparison is not an unseen final test.

The adapted Gradio API passed reference checking, consent rejection, typed English generation and Hindi-to-English translation with cloned speech. The first typed request took 46.84 seconds including model loading; the subsequent translation request took 11.45 seconds. The initial run passed 30 regression tests; the current application passes 36. Subsequent browser microphone debugging fixed a duplicated FastRTC session argument and added connection reset. Browser logs confirm microphone phrases reach the callback, but a complete browser-microphone-to-playback session is still not certified.

The app now defaults to the recovered adapted checkpoint. The verified archive remains in Downloads, and the inference files are in the project model directory. Full training logs and checksums are preserved in the task outputs. Kaggle's version-save requests encountered a server response error, so their queued status must not be treated as a verified cloud backup. The GPU session was stopped only after local recovery and log preservation.

Exact speaker identity, pitch, accent and expressive delivery remain unproven. Academic use is subject to XTTS's non-commercial Coqui Public Model License and the source dataset's terms.

## Reproduction and artifact provenance

- Dataset revision: `ff1d667d02492b92f6c019c2c35b2db04e61a077` of GLOBE V2.
- Training launcher: [`scripts/train_globe_xtts_pilot.py`](../scripts/train_globe_xtts_pilot.py); data preparation and portable export are in `scripts/`.
- Runtime: Python, coqui-tts 0.27.5, trainer 0.3.3, PyTorch 2.11.0 with CUDA 12.8. The Kaggle runtime offered two T4 cards; this run selected one using `CUDA_VISIBLE_DEVICES=0`.
- Training: one epoch, batch size 1, gradient accumulation 8, learning rate 5e-6; batch-100 weights-only warm start with optimizer reset. The training step counter reached 629. This is a pilot, not a converged full-scale training result.
- Inference config SHA-256: `ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8`.
- Vocabulary SHA-256: `928260878a59da8a72a2a5b7687fea29d5106137669d90945430fe17e415304a`.

The repository includes source, training procedures and reported results. It excludes large model weights, public audio bundles, raw training logs, private recordings and local run artifacts. Dataset selection and training depend on separately prepared audio and manifests; a source checkout alone is not a complete reproduction package. Local run metadata, training config and full console log are preserved with the development artifacts. No claim of exact cloning, Indian-only accent adaptation or human-level naturalness is supported by this pilot.
