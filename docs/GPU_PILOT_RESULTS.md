# GPU pilot results — 9 October 2026

Run `20261009T062346Z-34787293` completed the original Coqui XTTS GPT fine-tuning configuration on a Tesla T4. No selective layer freezing was added. The run started from the verified partial CPU checkpoint at batch 100, with a new optimizer, then performed one full epoch. Batch size was 1, accumulation was 8, and learning rate was 0.000005.

The pinned public GLOBE V2 selection contains 629 training clips from 15 speakers and 192 validation clips from four different speakers. The accent labels include India, Pakistan and Sri Lanka; this is a South Asian English pilot, not an India-only dataset. No personal voice recordings were uploaded.

Validation finished with finite averages: text cross-entropy 0.0232153, audio-token cross-entropy 3.9022563 and total loss 3.9254715. These are model losses, not perceptual voice-similarity scores. Separate final-test speakers and listening evaluation are still required.

The inference export removed optimizer state and training-only parameters and retained the original model precision. Its archive SHA-256 is `d22f6b470b794ae1dfb2235e928ddaf4b8b38b55abfb91ad0b99019259782f0c`. Model SHA-256 is `55ce63e5260da5401a46227794fc012d75be51510c1eb365372461ecb2c3361c`. The archive and all three inference files were verified after recovery to the Mac.

A local public-reference smoke test produced finite 24 kHz speech. Whisper-small transcribed both pretrained and adapted versions of one English sentence exactly: zero word errors for that sentence. This does not demonstrate an accent or tone improvement. The reference speaker belongs to the validation selection, so this comparison is not an unseen final test.

The adapted Gradio API passed reference checking, consent rejection, typed English generation and Hindi-to-English translation with cloned speech. The first typed request took 46.84 seconds including model loading; the subsequent translation request took 11.45 seconds. All 30 local regression tests passed. Browser microphone interaction has not been retested with this checkpoint.

The app now defaults to the recovered adapted checkpoint. The verified archive remains in Downloads, and the inference files are in the project model directory. Full training logs and checksums are preserved in the task outputs. Kaggle's version-save requests encountered a server response error, so their queued status must not be treated as a verified cloud backup. The GPU session was stopped only after local recovery and log preservation.

Exact speaker identity, pitch, accent and expressive delivery remain unproven. Academic use is subject to XTTS's non-commercial Coqui Public Model License and the source dataset's terms.
