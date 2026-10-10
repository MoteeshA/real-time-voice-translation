"""Find GLOBE V2 India/South Asia clips via the dataset viewer metadata API.

This avoids downloading the large embedded-audio parquet shards. Only matching
transcripts, speaker labels, durations, and dataset row indexes are saved.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://datasets-server.huggingface.co"
DATASET = "MushanW/GLOBE_V2"
ACCENT_SUBSTRING = "india and south asia"
PAGE_LENGTH = 100


def get_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "voice-translation-research/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 16))
        except (TimeoutError, urllib.error.URLError):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("unreachable retry loop")


def query_rows(split: str, offset: int) -> tuple[int, list[dict]]:
    params = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": PAGE_LENGTH,
    })
    result = get_json(f"{BASE_URL}/rows?{params}")
    matches = []
    for entry in result.get("rows", []):
        row = entry["row"]
        if ACCENT_SUBSTRING in str(row.get("accent", "")).lower():
            matches.append({
                "split": split,
                "row_index": int(entry.get("row_idx", offset)),
                "speaker_id": str(row.get("speaker_id", "")),
                "transcript": str(row.get("transcript", "")).strip(),
                "accent": str(row.get("accent", "")),
                "duration": float(row.get("duration", 0.0)),
            })
    return offset, matches


def scan_split(split: str, workers: int) -> tuple[int, list[dict]]:
    initial = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split, "offset": 0, "length": PAGE_LENGTH})
    first = get_json(f"{BASE_URL}/rows?{initial}")
    total = int(first["num_rows_total"])
    offsets = list(range(0, total, PAGE_LENGTH))
    found: list[dict] = []
    first_rows = query_rows(split, 0)[1]
    found.extend(first_rows)
    remaining = offsets[1:]
    completed = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(query_rows, split, offset): offset for offset in remaining}
        for future in as_completed(futures):
            _, matches = future.result()
            found.extend(matches)
            completed += 1
            if completed % 250 == 0 or completed == len(offsets):
                print(f"{split}: scanned {completed}/{len(offsets)} metadata pages; matching clips={len(found)}", flush=True)
    found.sort(key=lambda row: row["row_index"])
    return total, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/globe-v2-india-south-asia"))
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")

    # The dataset viewer labels the development split `val`.
    splits = ["val" if name == "validation" else name for name in args.splits]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": DATASET, "accent_filter": ACCENT_SUBSTRING, "splits": {}}
    for split in splits:
        print(f"Scanning {split} metadata from {DATASET} (audio is not downloaded).", flush=True)
        total, rows = scan_split(split, args.workers)
        path = args.output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        speakers = {row["speaker_id"] for row in rows if row["speaker_id"]}
        hours = sum(row["duration"] for row in rows) / 3600
        summary["splits"][split] = {
            "total_rows": total,
            "matching_clips": len(rows),
            "matching_speakers": len(speakers),
            "hours": round(hours, 3),
            "metadata_file": path.name,
        }
        print(json.dumps(summary["splits"][split], indent=2), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metadata only to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
