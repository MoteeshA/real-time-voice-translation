"""Make speaker-disjoint XTTS train/evaluation CSVs from a consented manifest.

Input columns: audio_path,text,speaker_id,consent
Output columns: audio_file|text|speaker_name (the XTTS fine-tuning demo format)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV containing audio_path,text,speaker_id,consent")
    parser.add_argument("--output", type=Path, default=Path("data/xtts"))
    parser.add_argument("--eval-fraction", type=float, default=0.2, help="Fraction of speakers held out for evaluation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-speakers", help="Comma-separated explicit held-out speaker IDs")
    args = parser.parse_args()

    if not 0.0 < args.eval_fraction < 1.0:
        parser.error("--eval-fraction must be between 0 and 1")
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    required = {"audio_path", "text", "speaker_id", "consent"}
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            parser.error("Manifest must contain: audio_path,text,speaker_id,consent")
        for line_number, row in enumerate(reader, start=2):
            path = Path(row["audio_path"]).expanduser().resolve()
            text = (row["text"] or "").strip()
            speaker = (row["speaker_id"] or "").strip()
            consent = (row["consent"] or "").strip().lower()
            if consent not in {"yes", "true", "1", "consented"}:
                parser.error(f"Line {line_number}: consent must be explicitly yes/true/1/consented")
            if not path.is_file() or not text or not speaker:
                parser.error(f"Line {line_number}: audio file, transcript, and speaker_id are required")
            grouped[speaker].append({"audio_file": str(path), "text": text, "speaker_name": speaker})

    speakers = sorted(grouped)
    if len(speakers) < 2:
        parser.error("At least two consenting speakers are required for a speaker-disjoint research evaluation")
    if args.eval_speakers:
        eval_speakers = {speaker.strip() for speaker in args.eval_speakers.split(",") if speaker.strip()}
        if not eval_speakers or not eval_speakers.issubset(speakers) or eval_speakers == set(speakers):
            parser.error("--eval-speakers must name known speakers and leave at least one speaker for training")
    else:
        random.Random(args.seed).shuffle(speakers)
        eval_count = max(1, round(len(speakers) * args.eval_fraction))
        eval_speakers = set(speakers[:eval_count])
    train_rows = [row for speaker in speakers if speaker not in eval_speakers for row in grouped[speaker]]
    eval_rows = [row for speaker in speakers if speaker in eval_speakers for row in grouped[speaker]]

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.csv", train_rows), ("eval.csv", eval_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(train_rows)} training utterances from {len(speakers) - len(eval_speakers)} speakers")
    print(f"Wrote {len(eval_rows)} evaluation utterances from {len(eval_speakers)} held-out speakers")
    print(f"Output directory: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
