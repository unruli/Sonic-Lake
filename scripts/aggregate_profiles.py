#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soniclake.io import read_jsonl
from soniclake.profiles import aggregate_results, profiles_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sample-level results into representation profiles.")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = (row for path in args.results for row in read_jsonl(path))
    payload = profiles_to_dict(aggregate_results(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload)} profiles to {args.output}")


if __name__ == "__main__":
    main()

