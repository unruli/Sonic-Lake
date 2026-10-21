#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate actor-labelled CREMA-D evaluation inputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="*")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample IDs")
    actors = [str(row["labels"]["actor_id"]) for row in rows]
    emotions = [str(row["labels"]["emotion"]) for row in rows]
    if len(set(actors)) < 5:
        raise ValueError("fewer than five actors; five-fold group evaluation is impossible")

    expected = set(ids)
    coverage = {}
    for path in args.records or []:
        record_ids = {row["sample_id"] for row in read_rows(path)}
        missing = expected - record_ids
        extra = record_ids - expected
        if missing or extra:
            raise ValueError(f"{path}: missing={len(missing)}, extra={len(extra)}")
        coverage[str(path)] = len(record_ids)

    print(json.dumps({
        "samples": len(rows),
        "actors": len(set(actors)),
        "samples_per_actor": dict(sorted(Counter(actors).items())),
        "emotions": dict(sorted(Counter(emotions).items())),
        "validated_record_files": len(coverage),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
