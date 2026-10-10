"""Scan GLOBE V2 parquet metadata without downloading embedded audio files.

Writes matching accent rows with global split row indices so a later step can
download only the selected audio examples from the Hugging Face dataset viewer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq


REPO = "MushanW/GLOBE_V2"
ACCENT = "India and South Asia"


def file_parts(name: str) -> tuple[str, int]:
    match = re.search(r"data/(train|val|test)-(\d+)-of-\d+\.parquet$", name)
    if not match:
        raise ValueError(f"Unexpected GLOBE parquet name: {name}")
    return match.group(1), int(match.group(2))


def read_shard(name: str) -> tuple[str, int, list[dict[str, object]], int]:
    split, shard_number = file_parts(name)
    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO}/{name}", "rb") as handle:
        parquet = pq.ParquetFile(handle)
        row_count = parquet.metadata.num_rows
        records: list[dict[str, object]] = []
        local_offset = 0
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                group, columns=["speaker_id", "transcript", "accent", "duration"]
            )
            rows = table.to_pydict()
            for index, accent in enumerate(rows["accent"]):
                if accent and ACCENT.lower() in accent.lower():
                    records.append({
                        "split": split,
                        "shard": name,
                        "shard_number": shard_number,
                        "local_row": local_offset + index,
                        "speaker_id": rows["speaker_id"][index],
                        "transcript": rows["transcript"][index],
                        "accent": accent,
                        "duration": float(rows["duration"][index]),
                    })
            local_offset += table.num_rows
    return split, shard_number, records, row_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/globe-v2-india-south-asia"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-shards", type=int, default=0, help="Limit each selected split to the first N shards (0 scans all).")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        parser.error("--workers must be between 1 and 12")

    info = HfApi().dataset_info(REPO, files_metadata=False)
    names = [sibling.rfilename for sibling in info.siblings if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet")]
    names.sort(key=lambda name: (file_parts(name)[0], file_parts(name)[1]))
    selected_splits = set(args.splits)
    names = [name for name in names if file_parts(name)[0] in selected_splits]
    if args.max_shards:
        chosen = []
        per_split: Counter[str] = Counter()
        for name in names:
            split, _ = file_parts(name)
            if per_split[split] < args.max_shards:
                chosen.append(name)
                per_split[split] += 1
        names = chosen
    if not names:
        parser.error(f"No parquet shards found in {REPO}")
    print(f"Scanning {len(names)} GLOBE V2 metadata shards for accent label containing {ACCENT!r}.", flush=True)

    results: dict[tuple[str, int], tuple[list[dict[str, object]], int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_shard, name): name for name in names}
        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                split, shard_number, records, row_count = future.result()
            except Exception as exc:
                print(f"FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                raise
            results[(split, shard_number)] = (records, row_count)
            completed += 1
            print(f"metadata shards {completed}/{len(names)} scanned", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    split_totals: Counter[str] = Counter()
    split_matches: Counter[str] = Counter()
    split_seconds: Counter[str] = Counter()
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for split, shard_number in sorted(results):
        records, row_count = results[(split, shard_number)]
        for record in records:
            record["row_index"] = split_totals[split] + int(record["local_row"])
            split_rows[split].append(record)
            split_matches[split] += 1
            split_seconds[split] += int(round(float(record["duration"])))
            speaker_splits[str(record["speaker_id"])].add(split)
        split_totals[split] += row_count

    for split, records in split_rows.items():
        path = args.output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_split_speakers = {
        split: len({str(row["speaker_id"]) for row in records})
        for split, records in split_rows.items()
    }
    summary = {
        "dataset": REPO,
        "dataset_revision": info.sha,
        "accent_filter": ACCENT,
        "split_total_rows": dict(split_totals),
        "accent_rows": dict(split_matches),
        "accent_hours": {split: round(seconds / 3600, 2) for split, seconds in split_seconds.items()},
        "accent_speakers": by_split_speakers,
        "speakers_crossing_splits": sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1),
        "metadata_files": {split: f"{split}.jsonl" for split in split_rows},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote metadata only; no dataset audio has been downloaded. Folder: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
