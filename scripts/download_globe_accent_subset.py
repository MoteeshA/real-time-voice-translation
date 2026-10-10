"""Download a balanced, speaker-disjoint GLOBE V2 Indian/South Asian pilot.

Requires metadata JSONL from scan_globe_accent.py. Audio is fetched only for
selected rows via the Hugging Face dataset viewer API; the full corpus is not
downloaded. The dataset is public CC0, but its accent tag groups India with
Pakistan and Sri Lanka, which must be reported as a limitation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


DATASET = "MushanW/GLOBE_V2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="train.jsonl produced by scan_globe_accent.py")
    parser.add_argument("--output", type=Path, default=Path("data/globe-v2-india-south-asia/pilot"))
    parser.add_argument("--min-per-speaker", type=int, default=20)
    parser.add_argument("--max-per-speaker", type=int, default=50)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.eval_fraction < 1 or args.max_per_speaker < args.min_per_speaker:
        parser.error("Use 0 < --eval-fraction < 1 and max-per-speaker >= min-per-speaker")
    if not args.metadata.is_file():
        parser.error(f"Metadata not found: {args.metadata}")

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    with args.metadata.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("transcript") and row.get("speaker_id"):
                by_speaker[str(row["speaker_id"])].append(row)
    eligible = {speaker: rows for speaker, rows in by_speaker.items() if len(rows) >= args.min_per_speaker}
    speakers = sorted(eligible)
    if len(speakers) < 5:
        parser.error(f"Only {len(speakers)} speakers have >= {args.min_per_speaker} clips; need at least five for a split")

    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    eval_count = max(1, round(len(speakers) * args.eval_fraction))
    eval_speakers = set(speakers[:eval_count])
    selected: list[dict] = []
    for speaker in speakers:
        rows = sorted(eligible[speaker], key=lambda row: int(row["row_index"]))
        if len(rows) > args.max_per_speaker:
            rows = sorted(rng.sample(rows, args.max_per_speaker), key=lambda row: int(row["row_index"]))
        for row in rows:
            row = dict(row)
            row["target_split"] = "eval" if speaker in eval_speakers else "train"
            selected.append(row)
    selected.sort(key=lambda row: int(row["row_index"]))

    split_speakers = {
        split: sorted({str(row["speaker_id"]) for row in selected if row["target_split"] == split})
        for split in ("train", "eval")
    }
    split_seconds = {
        split: sum(float(row["duration"]) for row in selected if row["target_split"] == split)
        for split in ("train", "eval")
    }
    pages = sorted({(int(row["row_index"]) // 100) * 100 for row in selected})
    report = {
        "dataset": DATASET,
        "license": "CC0-1.0 (dataset card)",
        "accent_label": "India and South Asia (India, Pakistan, Sri Lanka)",
        "source_shards_scanned": "first 32 GLOBE V2 train shards only",
        "selection": {"minimum_clips_per_speaker": args.min_per_speaker, "maximum_clips_per_speaker": args.max_per_speaker, "seed": args.seed},
        "selected_viewer_pages": len(pages),
        "speakers": {split: len(ids) for split, ids in split_speakers.items()},
        "utterances": {split: sum(row["target_split"] == split for row in selected) for split in ("train", "eval")},
        "hours": {split: round(seconds / 3600, 3) for split, seconds in split_seconds.items()},
        "speaker_overlap": sorted(set(split_speakers["train"]) & set(split_speakers["eval"])),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    audio_root = args.output / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    download_jobs: list[dict[str, str | int]] = []
    csv_rows: dict[str, list[dict[str, str]]] = {"train": [], "eval": []}
    total_bytes = 0
    summary_path = args.metadata.parent / "summary.json"
    revision = json.loads(summary_path.read_text(encoding="utf-8")).get("dataset_revision") if summary_path.is_file() else None
    if not revision:
        parser.error(f"Expected dataset_revision in {summary_path}; re-run the metadata scanner")
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for record in selected:
        if not record.get("shard") or "local_row" not in record:
            parser.error("Metadata rows must include shard and local_row; re-run scripts/scan_globe_accent.py")
        by_shard[str(record["shard"])].append(record)

    with tempfile.TemporaryDirectory(prefix="globe-selected-shards-") as temp_cache:
        for shard_index, (shard, shard_records) in enumerate(sorted(by_shard.items()), start=1):
            print(f"Fetching selected source shard {shard_index}/{len(by_shard)}: {shard}", flush=True)
            local_path = hf_hub_download(
                repo_id=DATASET,
                filename=shard,
                repo_type="dataset",
                revision=revision,
                cache_dir=temp_cache,
            )
            parquet = pq.ParquetFile(local_path)
            targets_by_group: dict[int, list[tuple[int, dict]]] = defaultdict(list)
            local_offset = 0
            for group_index in range(parquet.num_row_groups):
                group_count = parquet.metadata.row_group(group_index).num_rows
                group_end = local_offset + group_count
                for record in shard_records:
                    local_row = int(record["local_row"])
                    if local_offset <= local_row < group_end:
                        targets_by_group[group_index].append((local_row - local_offset, record))
                local_offset = group_end

            for group_index, targets in targets_by_group.items():
                table = parquet.read_row_group(group_index, columns=["audio", "speaker_id", "transcript", "accent"])
                data = table.to_pydict()
                for row_position, record in targets:
                    if data["speaker_id"][row_position] != record["speaker_id"] or data["accent"][row_position] != record["accent"]:
                        raise RuntimeError(f"Metadata mismatch in {shard} row {record['local_row']}; refusing to train on misaligned audio")
                    audio_record = data["audio"][row_position] or {}
                    audio_bytes = audio_record.get("bytes")
                    if not audio_bytes:
                        raise RuntimeError(f"Missing audio bytes in {shard} row {record['local_row']}")
                    split = str(record["target_split"])
                    speaker = str(record["speaker_id"])
                    row_index = int(record["row_index"])
                    suffix = Path(str(audio_record.get("path") or "sample.flac")).suffix or ".flac"
                    destination = audio_root / split / speaker / f"row-{row_index:07d}{suffix}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(audio_bytes)
                    total_bytes += len(audio_bytes)
                    csv_rows[split].append({
                        "audio_file": str(destination.resolve()),
                        "text": str(data["transcript"][row_position]).strip(),
                        "speaker_name": speaker,
                    })
            print(f"Extracted {sum(len(rows) for rows in csv_rows.values())}/{len(selected)} selected utterances", flush=True)

    for split, rows in csv_rows.items():
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
            writer.writeheader()
            writer.writerows(rows)
    report["downloaded_bytes"] = total_bytes
    (args.output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Prepared XTTS manifests at {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
