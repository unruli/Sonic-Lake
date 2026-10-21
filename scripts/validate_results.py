#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from soniclake.io import read_jsonl
from soniclake.schema import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized SonicLake result rows.")
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()

    keys: set[tuple[str, ...]] = set()
    counts: Counter[str] = Counter()
    for path in args.results:
        for index, raw in enumerate(read_jsonl(path), start=1):
            try:
                result = RunResult(**raw)
                result.validate()
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{path}:{index}: {exc}") from exc
            key = (
                result.run_id,
                result.sample_id,
                result.representation,
                result.operating_point,
                result.query_family,
            )
            if key in keys:
                raise SystemExit(f"{path}:{index}: duplicate result key {key}")
            keys.add(key)
            counts[result.status] += 1
    print("validated", len(keys), "rows", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()

