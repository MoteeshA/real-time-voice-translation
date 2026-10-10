"""Download a small, speaker-labelled VCTK subset for the XTTS pilot study.

VCTK source mirror: https://huggingface.co/datasets/saeedzou/vctk-16khz
The dataset card lists CC BY 4.0. Keep its attribution with research outputs.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


TRAIN_SPEAKERS = {"p225", "p226", "p227"}
EVAL_SPEAKERS = {"p228"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/vctk-subset"))
    parser.add_argument("--utterances-per-speaker", type=int, default=80)
    args = parser.parse_args()
    if args.utterances_per_speaker < 2:
        parser.error("Use at least two utterances per speaker")

    args.output.mkdir(parents=True, exist_ok=True)
    audio_root = args.output / "audio"
    audio_root.mkdir(exist_ok=True)
    needed = TRAIN_SPEAKERS | EVAL_SPEAKERS
    counts = {speaker: 0 for speaker in needed}
    manifest_path = args.output / "manifest.csv"

    try:
        rows = load_dataset("saeedzou/vctk-16khz", split="train", streaming=True)
        rows = rows.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        parser.error(f"Could not open public VCTK dataset; check network/datasets installation: {exc}")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "text", "speaker_id", "consent"])
        writer.writeheader()
        for item in rows:
            speaker = str(item.get("speaker_id", ""))
            if speaker not in needed or counts[speaker] >= args.utterances_per_speaker:
                if all(counts[s] >= args.utterances_per_speaker for s in needed):
                    break
                continue
            transcript = str(item.get("text", "")).strip()
            audio = item.get("audio")
            if not transcript or not isinstance(audio, dict) or not audio.get("bytes"):
                continue
            filename = str(item.get("file_id") or item.get("id") or f"{speaker}_{counts[speaker]:04d}")
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
            destination = (audio_root / f"{filename}.wav").resolve()
            samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            sf.write(destination, samples, sample_rate)
            writer.writerow({
                "audio_path": str(destination),
                "text": transcript.replace("|", " ").replace("\n", " "),
                "speaker_id": speaker,
                "consent": "yes",
            })
            counts[speaker] += 1
            if all(counts[s] >= args.utterances_per_speaker for s in needed):
                break

    missing = {speaker: count for speaker, count in counts.items() if count < args.utterances_per_speaker}
    if missing:
        parser.error(f"Dataset stream ended before all selected speakers were collected: {missing}")
    print(f"Saved {sum(counts.values())} utterances: {counts}")
    print(f"Training speakers: {sorted(TRAIN_SPEAKERS)}; held-out evaluation speaker: {sorted(EVAL_SPEAKERS)}")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
