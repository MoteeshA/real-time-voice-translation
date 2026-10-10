# XTTS accent fine-tuning

The primary experiment adapts XTTS-v2 on public multi-speaker English speech from GLOBE V2. The speaker cloned at inference does not need to appear in training: a short reference conditions speaker identity. Accent adaptation is a hypothesis to measure, not a promise of exact pitch, tone, or identity. See [ACCENT_ADAPTATION.md](ACCENT_ADAPTATION.md).

## Current status

The pilot has 629 training clips from 15 speakers and 192 validation clips from four different speakers. Its label is **India and South Asia (India, Pakistan, Sri Lanka)**, not India only. The source dataset is [GLOBE V2](https://huggingface.co/datasets/MushanW/GLOBE_V2), revision `ff1d667d02492b92f6c019c2c35b2db04e61a077`, with a CC0-1.0 dataset card. The initial CPU attempt exceeded the practical memory budget of the 8 GB Apple M2 and was stopped. **No adapted checkpoint or demonstrated accent improvement exists yet.**

## Portable CUDA pilot

`python scripts/export_training_bundle.py` builds `../outputs/globe-xtts-gpu-training.zip` using only the public pilot audio, relative CSV filenames, training launcher, pinned direct requirements, and documentation. The archive manifest includes each file's SHA-256. No user reference recordings, cached models, or virtual environment are included.

Extract on a Linux CUDA machine with Python 3.11 or 3.12. A 16–24 GB GPU is a starting target, not a measured hardware requirement; reduce batch size on out-of-memory failures. Allow at least 40 GB disk for dependencies, model files, and optimizer checkpoints. Install matching CUDA builds of `torch==2.11.0` and `torchaudio==2.11.0` using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install `requirements-training.txt` from the bundle. A Colab T4 pilot was attempted but interrupted by a runtime reset after non-finite average loss; successful completion remains unverified. These direct package versions match the local import-tested stack; transitive dependencies are recorded in the environment rather than fully locked.

From the extracted directory:

```bash
python -m pip install -r requirements-training.txt
python scripts/train_globe_xtts_pilot.py --check-data-only
python scripts/train_globe_xtts_pilot.py --epochs 1 --batch-size 1 --grad-accum 8
```

For a repository checkout, install `python -m pip install -e '.[voice]'` and ensure `coqui-tts==0.27.5` is selected. Review the [Coqui Public Model License](https://coqui.ai/cpml) before running; the upstream trainer downloads its XTTS model files. The launcher supports exactly Coqui 0.27.5 because it overrides the demo's config and Trainer classes, avoiding edits to installed source. XTTS model files are downloaded from upstream `main`, so retain the recorded SHA-256 assets to reproduce an experiment; this is not a permanently pinned model download.

The launcher checks file existence, nonempty transcripts, duplicate audio, and speaker/audio overlap before importing the model. Relative audio filenames resolve against their CSV's directory. CUDA is required by default, before model loading or download. `--allow-cpu` is an explicit diagnostic override and should not be used on the 8 GB Mac. `--check-data-only` requires only Python and does not load audio or validate its sound quality.

Default logs occur every 10 batches; checkpoints every 50 batches, with the latest two retained. The upstream trainer selects `best_model.pth` using validation loss at each epoch end, and the wrapper also saves a final optimizer checkpoint. Each attempt receives a unique timestamped output directory with `run_metadata.json`, `training_config.json`, prepared CSVs, software versions, seed, hardware, base-file hashes, and output paths. Interrupts/failures can leave partial runs; inspect metadata and logs rather than treating a directory as training success. The data loader may filter overly long audio/text according to upstream limits; compare its accepted sample counts with the manifest. One epoch is a feasibility check, not the final paper result.

## Load the resulting checkpoint

Only a successful run exports an `inference/` directory containing:

- `model.pth`: inference weights from the best saved checkpoint, with optimizer state and training-only layers removed.
- `config.json`: the original XTTS inference config, **not** the GPT trainer's `config.json`.
- `vocab.json`: the matching XTTS vocabulary.

Copy that entire directory back to the project machine. The exporter follows upstream XTTS compatibility conversion: it removes the trainer's `xtts.` prefixes and training-only DVAE/mel layers, and saves only the model state. This avoids loading optimizer tensors during inference on the 8 GB Mac. The application uses the direct `Xtts.load_checkpoint(checkpoint_path=..., vocab_path=...)` path. No manual renaming or key stripping is necessary. Full optimizer checkpoints remain in the GPU training directory.

```bash
export VOICE_TTS_MODEL_PATH="/absolute/path/to/inference/model.pth"
export VOICE_TTS_CONFIG_PATH="/absolute/path/to/inference/config.json"
export VOICE_TTS_VOCAB_PATH="/absolute/path/to/inference/vocab.json"
python app.py
```

Restart the application when changing checkpoints. The vocabulary variable is optional when `vocab.json` is next to the model. Keep the pretrained baseline available. Before using the adapted checkpoint as default, compare both models on identical held-out text and speaker references for intelligibility, similarity, accent, naturalness, and latency.

The four validation speakers guide checkpoint selection. They are therefore **not an untouched final test set**. Add separate unseen speakers and text for the paper's final evaluation; never choose settings using the final test set.

## Optional target-speaker experiment

Personal voice training is separate from this public accent experiment and is not required for reference-based cloning. If explicitly desired, [VOICE_RECORDING_PROTOCOL.md](VOICE_RECORDING_PROTOCOL.md) and `scripts/prepare_speaker_adaptation.py` prepare a session-disjoint private dataset. Do not use that single-speaker dataset with this public pilot exporter or its speaker-disjoint training validator. The separate upstream UI launcher `scripts/finetune_xtts.py` remains available for that experiment.

Public GLOBE data being CC0 does not change XTTS's non-commercial model license. No uploaded data, paid cloud resource, or remote training job is created by these scripts.

For persistent storage, pass `--backup-dir /content/drive/MyDrive/VoiceTranslationTraining/checkpoints` after authorizing and mounting Drive. Keep sufficient free storage for the current and temporary replacement checkpoint (roughly twice one full checkpoint), plus the final inference bundle. The launcher stops on non-finite losses and rejects non-finite model parameters during export.

## Quota-safe snapshots and CPU fallback

Full optimizer backups exceeded the institutional Drive quota. `--backup-dir` now writes compact FP16 model weights (`latest_model_fp16.pth`) without optimizer or training-only DVAE/mel layers. These are model warm-start snapshots, not exact optimizer resumes. Verify remote persistence before relying on them. `--download-snapshots` announces compact files for the notebook to download directly to your computer, including before validation. The final successful inference export retains the original checkpoint precision.

`--initial-checkpoint` restores model weights for a new epoch and resets optimizer state; metadata records this distinction. The CUDA quota is currently exhausted. `Train_XTTS_Accent_CPU_Fallback.ipynb` is a prepared diagnostic alternative on a cloud CPU with sufficient RAM. CPU speed and memory have not been measured; do not run it on the 8 GB Mac.
