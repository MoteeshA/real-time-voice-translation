"""Validate one consented speaker's WAV manifest and create session-disjoint XTTS CSVs.

Input CSV columns: audio_path,text,speaker_id,session_id,consent
Output CSV columns: audio_file|text|speaker_name (XTTS fine-tuning demo format)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path


def inspect_wav(path: Path) -> tuple[float, int, float]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            rate = audio.getframerate()
            frames = audio.getnframes()
            if audio.getcomptype() != "NONE" or channels != 1 or width != 2:
                raise ValueError("expected uncompressed mono 16-bit PCM WAV")
            if rate not in {16000, 22050, 24000, 44100, 48000}:
                raise ValueError(f"unsupported sample rate {rate}; use 16–48 kHz PCM WAV")
            peak_frames = clipped = 0
            while True:
                raw = audio.readframes(65536)
                if not raw:
                    break
                # PCM is little-endian signed 16-bit. Check for samples at the rail.
                count = len(raw) // 2
                samples = memoryview(raw).cast("h")
                clipped += sum(1 for sample in samples if abs(sample) >= 32760)
                peak_frames += count
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"cannot read PCM WAV: {exc}") from exc
    return frames / rate, rate, clipped / max(peak_frames, 1)


def choose_eval_sessions(
    durations: dict[str, float], fraction: float, seed: int
) -> set[str]:
    sessions = sorted(durations)
    target = sum(durations.values()) * fraction
    # For a modest number of recording sessions, select the closest duration split.
    if len(sessions) <= 18:
        candidates = []
        for size in range(1, len(sessions)):
            for chosen in itertools.combinations(sessions, size):
                heldout_seconds = sum(durations[name] for name in chosen)
                candidates.append((abs(heldout_seconds - target), heldout_seconds, chosen))
        best_error = min(candidate[0] for candidate in candidates)
        tied = [candidate[2] for candidate in candidates if candidate[0] <= best_error + 1e-9]
        return set(random.Random(seed).choice(tied))
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    selected: set[str] = set()
    total = 0.0
    for session in shuffled:
        if len(selected) == len(sessions) - 1:
            break
        if total < target:
            selected.add(session)
            total += durations[session]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with audio_path,text,speaker_id,session_id,consent")
    parser.add_argument("--output", type=Path, default=Path("data/private_speaker/xtts"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 0.0 < args.eval_fraction < 1.0:
        parser.error("--eval-fraction must be between 0 and 1")
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")

    grouped: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    session_durations: dict[str, float] = defaultdict(float)
    speakers: set[str] = set()
    seen_paths: set[Path] = set()
    required = {"audio_path", "text", "speaker_id", "session_id", "consent"}
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            parser.error("Manifest must contain: audio_path,text,speaker_id,session_id,consent")
        for line_number, row in enumerate(reader, start=2):
            path = Path(row["audio_path"]).expanduser().resolve()
            text = (row["text"] or "").strip()
            speaker = (row["speaker_id"] or "").strip()
            session = (row["session_id"] or "").strip()
            consent = (row["consent"] or "").strip().lower()
            if consent not in {"yes", "true", "1", "consented"}:
                parser.error(f"Line {line_number}: explicit consent is required")
            if not path.is_file() or not text or not speaker or not session:
                parser.error(f"Line {line_number}: audio, transcript, speaker_id, and session_id are required")
            if path in seen_paths:
                parser.error(f"Line {line_number}: duplicate audio_path")
            seen_paths.add(path)
            speakers.add(speaker)
            try:
                duration, rate, clipped_ratio = inspect_wav(path)
            except ValueError as exc:
                parser.error(f"Line {line_number}: {path.name}: {exc}")
            if not 1.0 <= duration <= 15.0:
                parser.error(f"Line {line_number}: {path.name}: clip is {duration:.1f}s; expected 1–15s")
            grouped[session].append({"audio_file": str(path), "text": text, "speaker_name": speaker,
                                     "duration": duration, "sample_rate": rate, "clipped_ratio": clipped_ratio})
            session_durations[session] += duration

    if len(speakers) != 1:
        parser.error("This workflow is for exactly one target speaker; create a separate manifest per speaker")
    if len(grouped) < 3:
        parser.error("Record at least three separate sessions to hold out a meaningful evaluation session")

    eval_sessions = choose_eval_sessions(session_durations, args.eval_fraction, args.seed)
    train_rows = [row for session, rows in grouped.items() if session not in eval_sessions for row in rows]
    eval_rows = [row for session, rows in grouped.items() if session in eval_sessions for row in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.csv", train_rows), ("eval.csv", eval_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)

    report = {
        "speaker_count": 1,
        "utterance_count": len(train_rows) + len(eval_rows),
        "train_utterances": len(train_rows),
        "train_minutes": round(sum(float(row["duration"]) for row in train_rows) / 60, 2),
        "train_sessions": sorted(set(grouped) - eval_sessions),
        "eval_utterances": len(eval_rows),
        "eval_minutes": round(sum(float(row["duration"]) for row in eval_rows) / 60, 2),
        "eval_sessions": sorted(eval_sessions),
        "sample_rates_hz": sorted({int(row["sample_rate"]) for row in train_rows + eval_rows}),
        "clipped_utterances": sum(float(row["clipped_ratio"]) > 0.001 for row in train_rows + eval_rows),
        "note": "Review and listen to clips and transcripts before training. Evaluation sessions must not be used for training or tuning.",
    }
    (args.output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"XTTS train/eval manifests written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
