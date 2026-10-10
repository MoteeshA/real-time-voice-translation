# Accent adaptation experiment

## Conditions

| ID | Generator | Speaker reference | Purpose |
|---|---|---|---|
| A | Pretrained XTTS-v2 | Fixed reference from an unseen speaker | Zero-shot baseline |
| B | XTTS-v2 adapted on public South Asian English training speakers | Same reference | Measure public accent-adaptation effect |

Keep input text, language, reference audio, decoding settings, and seed fixed across A and B. Evaluate multiple synthesis seeds after the initial paired check. Translation quality is evaluated separately; use identical translated text to isolate the TTS change. Any later prosody processing or target-speaker fine-tuning needs its own ablation, rather than silently changing condition B.

## Splits and measures

- Train/validation speaker sets must be disjoint. The current pilot uses 15 train and four validation speakers; validation loss selects the checkpoint.
- Reserve additional unseen speakers and sentences as a final test set before selecting settings. The current validation set cannot support an untouched final-test claim.
- If separately testing personal speaker adaptation, split that speaker by recording session and report it separately from zero-shot generalization.
- Exclude noisy, clipped, overlapping, mistranscribed, or generated audio using predeclared rules. The public pilot still needs a systematic listening review.
- Measure intelligibility (WER/CER), speaker-embedding similarity, blinded accent/naturalness ratings, and latency. Pitch comparison is meaningful only for matched text and comparable delivery; do not interpret average pitch as identity accuracy.

| Condition | WER/CER | Speaker similarity | Accent/naturalness listener rating | First-audio latency p50/p90 | Phrase delay p50/p90 |
|---|---:|---:|---:|---:|---:|
| A | Not measured | Not measured | Not measured | Not measured | Not measured |
| B | No checkpoint | No checkpoint | No checkpoint | No checkpoint | No checkpoint |

Accent adaptation may improve local pronunciation and rhythm, but can worsen unseen-speaker cloning or multilingual pronunciation. Publish both improvements and regressions. It cannot guarantee identical voice tone or commercial-service quality.

## Hardware smoke test — 2026-10-07

- Data: GLOBE V2, first 32 training shards scanned. The pilot includes 629 training clips from 15 speakers and 192 validation clips from four speakers. The label is `India and South Asia (India, Pakistan, Sri Lanka)`, so it cannot support an India-only accent claim. Dataset-card license: CC0-1.0.
- Setup: XTTS-v2 GPT fine-tuning, English, one epoch, batch size 1, gradient accumulation 8, zero additional loader workers. Initial logged loss was 0.5135 (text CE 0.0217, mel CE 4.0864).
- Outcome: the 8 GB Apple M2 CPU run reached the first batch but was stopped when memory footprint reached 7.5 GB, with 9 GB peak and heavy swapping. No trained checkpoint or evaluation result was produced. This is hardware feasibility evidence, not a model result.

## Portable training preparation — 2026-10-07

The launcher now refuses non-CUDA training before loading model weights unless explicitly overridden for diagnostics. It validates the data split, records configuration/environment/hashes, logs every 10 batches, saves every 50 batches and at epoch completion, and exports model/config/vocabulary after success. The public-only ZIP uses portable relative audio paths and includes a checksum manifest. Local validation covers data paths, 821 audio decodes, no train/validation speaker overlap, unsafe-path protection, and CLI preflight behavior; CUDA training and adapted inference remain unverified.

Next experiment: run the one-epoch GPU feasibility job, inspect accepted sample counts and losses, generate paired A/B samples, then select a larger training schedule only if intelligibility and identity remain acceptable. Record the final test split before tuning the schedule.

## Training package verification — 2026-10-08

Seven lightweight regression tests pass: portable CSV paths, missing inputs, speaker/audio overlap, argument validation, CUDA preflight, public-only archive isolation, and inference export using a small tensor checkpoint. The export strips optimizer state and training-only layers while preserving inference weights. Upstream Coqui trainer 0.3.3 saves `best_model.pth` after epoch validation; its `save_checkpoint()` takes no arguments. The launcher calls the pinned epoch loop so interrupts update run metadata instead of hard-exiting Python. These checks do not substitute for a CUDA training run or an adapted-checkpoint listening comparison.

## Colab T4 attempt and recovery — 8 October 2026

The public pilot ran on a Tesla T4 (14.6 GiB), Python 3.13, PyTorch/TorchAudio 2.11.0+cu130. The visible log reached step 440/628 and reported non-finite average mel loss. A later runtime reset removed the temporary project directory and checkpoints. This run cannot support a claim of successful fine-tuning.

The pinned upstream GPT code divides waveform mask indices by 256 while applying them to targets encoded at a 1024-sample stride. The launcher compensates before that division. A regression reproduces an all-ignored target and NaN cross-entropy, then verifies that the corrected mask leaves finite loss. The launcher now rejects non-finite training/validation losses and exported parameters. Optional `--backup-dir` writes the latest complete optimizer checkpoint through a temporary file and saves metadata/config/source; inference files are copied after success. A new GPU run and held-out inference evaluation remain required.

## Later recovery and resource limits

A repaired T4 run produced finite losses and reached checkpoint 300, but full optimizer backups hit Drive quota. A new model warm-start from local checkpoint 300 (optimizer reset) completed its training loop and entered evaluation; the last saved log was step 620/628, with average mel CE 3.5754936463890536. Colab terminated the runtime before a successful export was verified, then refused GPU allocation due to usage limits. CPU recovery found metadata and only a 223,608,832-byte partial backup. This is progress evidence, not a completed model or validation result. Compact weight snapshots and direct downloads were subsequently added and locally tested; cloud CPU training remains unexecuted.

## CPU codec repair and feasibility measurement

Colab CPU used PyTorch/TorchAudio 2.11.0+cpu. The default Linux TorchCodec wheel depended on libnvrtc.so.13 and caused the XTTS dataset loader to recursively retry every file. Reinstalling TorchCodec 0.17.0+cpu from the official PyTorch CPU wheel index fixed direct FLAC decoding on train and validation inputs. An early decoder check now stops before model loading when this dependency is broken.

The CPU retry reached step 10/628: average step time 11.839391803741455 seconds, average loader time 0.15141258239746094 seconds, average mel CE 5.602881526947021, and reported RAM 10.16/12.67 GB. These are early feasibility observations; no CPU training completion, validation score, or perceptual improvement is claimed.

## CPU SIGKILL and memory optimizations

The first decoded CPU run reached visible batch 90 and then returned -9 (SIGKILL), with no checkpoint files. The earlier RAM measurement was 10.16/12.67 GB. The visible cgroup counters did not identify OOM, so the cause remains probable memory pressure rather than confirmed. The retry uses non-reentrant gradient checkpointing, sequential AdamW updates, live model-only FP16 snapshots and optimizer-free full-precision best checkpoints. A small GPT2 regression matches parameter gradients with and without checkpointing. Initial retry loss was finite; sustained operation and snapshot success remain unverified.
