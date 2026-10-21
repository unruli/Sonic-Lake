#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soniclake.io import read_jsonl, write_jsonl
from soniclake.systems import run_closed_loop


def select_items(records: Path, limit: int, repetitions: int) -> list[dict]:
    rows = [row for row in read_jsonl(records) if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"no successful records in {records}")
    if limit < 1 or repetitions < 1:
        raise ValueError("limit and repetitions must be positive")
    if limit < len(rows):
        indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [len(rows) // 2]
        rows = [rows[index] for index in indices]
    return (rows * repetitions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark concurrent real artifact access and decode.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--codec", choices=("source", "pcm", "flac", "opus"), required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    items = select_items(args.records, args.limit, args.repetitions)
    measurements, summary = run_closed_loop(
        items,
        codec=args.codec,
        concurrency=args.concurrency,
        cache_mode=args.cache_mode,
        ffmpeg=args.ffmpeg,
    )
    summary.update(representation=args.representation, records=str(args.records))
    write_jsonl(args.output, (row.to_dict() for row in measurements))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
