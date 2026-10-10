"""Export only the public GLOBE pilot and its CUDA training launcher as a ZIP."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

from train_globe_xtts_pilot import FIELDS, validate_dataset

PROJECT = Path(__file__).resolve().parents[1]
DATA_RELATIVE = Path("data/globe-v2-india-south-asia/pilot")
README = """# GLOBE South Asian English XTTS pilot

This archive contains public GLOBE V2 audio only: 629 train clips / 15 speakers
and 192 validation clips / 4 different speakers. It contains no user reference
recordings, pretrained models, virtual environments, or trained checkpoint.

Extract on a Linux CUDA GPU machine with Python 3.11 or 3.12. A 16–24 GB GPU is
a starting target, not a tested memory guarantee. Do not run this on the 8 GB Mac.
Allow at least 40 GB disk for dependencies, models and optimizer checkpoints.

1. Create and activate a virtual environment.
2. Install torch==2.11.0 and torchaudio==2.11.0 using the CUDA wheel command
   appropriate for the GPU driver from https://pytorch.org/get-started/locally/.
3. Run `python -m pip install -r requirements-training.txt`.
4. Run `python scripts/train_globe_xtts_pilot.py --check-data-only`.
5. Review the XTTS model license: https://coqui.ai/cpml . The training process
   downloads the upstream XTTS-v2 model. Resulting model remains non-commercial.
6. Run `python scripts/train_globe_xtts_pilot.py --epochs 1 --batch-size 1 --grad-accum 8`.

A GPU must be visible to PyTorch. The launcher refuses CPU training by default.
It logs every 10 batches and checkpoints every 50 batches and after each epoch.
After success, `models/xtts-india-south-asia-pilot/<run>/inference/` contains
model.pth, config.json, vocab.json; copy all three back to the inference machine.
`run_metadata.json` records settings, versions, hardware and artifact hashes.
See docs/FINETUNING.md for exact app environment variables and evaluation steps.

This is a feasibility pilot. Validation speakers are used to select a checkpoint;
a separate unseen test set is still needed for the research paper. There is no
claim of accent improvement or exact voice reproduction before evaluation.
"""
REQUIREMENTS = """# Install matched CUDA torch/torchaudio 2.11.0 wheels first (see README).
# Versions match the local import-tested training stack; CUDA execution remains unverified.
coqui-tts[codec]==0.27.5
coqui-tts-trainer==0.3.3
transformers==4.57.6
soundfile==0.14.0
torch==2.11.0
torchaudio==2.11.0
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT.parent / "outputs" / "globe-xtts-gpu-training.zip")
    parser.add_argument("--force", action="store_true", help="Replace an existing export at the chosen path")
    args = parser.parse_args(argv)
    destination = args.output.expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        parser.error("output must have a .zip extension")
    if destination.exists() and not args.force:
        parser.error("output already exists; use --force to replace this export")
    root = (PROJECT / DATA_RELATIVE).resolve()
    try:
        splits, report = validate_dataset(root / "train.csv", root / "eval.csv")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    entries: dict[str, bytes | Path] = {
        "README.md": README.encode(), "requirements-training.txt": REQUIREMENTS.encode(),
        "scripts/train_globe_xtts_pilot.py": PROJECT / "scripts/train_globe_xtts_pilot.py",
        "scripts/export_training_bundle.py": PROJECT / "scripts/export_training_bundle.py",
        "tests/test_training_package.py": PROJECT / "tests/test_training_package.py",
        "docs/FINETUNING.md": PROJECT / "docs/FINETUNING.md",
        "docs/EXPERIMENTS.md": PROJECT / "docs/EXPERIMENTS.md",
        "docs/ACCENT_ADAPTATION.md": PROJECT / "docs/ACCENT_ADAPTATION.md",
    }
    # Explicit allowlist; never walk other project data or upload/reference caches.
    for split, rows in splits.items():
        normalized = []
        for row in rows:
            audio = Path(row["audio_file"])
            try:
                relative = audio.relative_to(root)
            except ValueError:
                parser.error(f"Refusing non-public-pilot audio: {audio}")
            if len(relative.parts) != 4 or relative.parts[:2] != ("audio", split):
                parser.error(f"Unexpected public audio layout: {relative}")
            if relative.parts[2] != row["speaker_name"] or audio.suffix.lower() not in (".flac", ".wav"):
                parser.error(f"Unexpected public audio metadata: {relative}")
            normalized.append({**{key: row[key] for key in FIELDS}, "audio_file": relative.as_posix()})
            entries[(DATA_RELATIVE / relative).as_posix()] = audio
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, delimiter="|")
        writer.writeheader()
        writer.writerows(normalized)
        entries[(DATA_RELATIVE / f"{split}.csv").as_posix()] = buffer.getvalue().encode()
    source_report = json.loads((root / "dataset_report.json").read_text())
    entries[(DATA_RELATIVE / "dataset_report.json").as_posix()] = json.dumps(source_report, indent=2).encode()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MushanW/GLOBE_V2", "dataset_revision": "ff1d667d02492b92f6c019c2c35b2db04e61a077",
        "dataset_url": "https://huggingface.co/datasets/MushanW/GLOBE_V2",
        "dataset_license": "CC0-1.0 (dataset card)",
        "accent_label": "India and South Asia (India, Pakistan, Sri Lanka)",
        "split": {key: report[key] for key in ("train", "eval", "speaker_overlap")},
        "contains_user_recordings": False, "contains_models": False, "files": {},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.zip")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, source in sorted(entries.items()):
                payload = source.read_bytes() if isinstance(source, Path) else source
                manifest["files"][name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                archive.writestr(name, payload)
            archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC verification failed")
            for name in archive.namelist():
                if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                    raise ValueError(f"Unsafe archive entry: {name}")
                if name in manifest["files"] and hashlib.sha256(archive.read(name)).hexdigest() != manifest["files"][name]["sha256"]:
                    raise ValueError(f"Archive hash mismatch: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
                      "audio_clips": sum(len(rows) for rows in splits.values()), "verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
