#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soniclake.optimizer import normalize_frequencies, optimize
from soniclake.profiles import profiles_from_dict


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a SonicLake physical design by exact subset search.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--archive-hours", type=float, default=1.0)
    parser.add_argument("--horizon-queries", type=float, default=1.0)
    parser.add_argument("--storage-budget-bytes", type=float)
    parser.add_argument("--max-materializations", type=int)
    parser.add_argument("--storage-weight", type=float, default=1.0)
    parser.add_argument("--construction-weight", type=float, default=1.0)
    parser.add_argument("--query-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workload = load_object(args.workload)
    design = optimize(
        profiles_from_dict(load_object(args.profiles)),
        normalize_frequencies(workload["frequencies"]),
        workload["thresholds"],
        archive_hours=args.archive_hours,
        horizon_queries=args.horizon_queries,
        storage_budget_bytes=args.storage_budget_bytes,
        max_materializations=args.max_materializations,
        storage_weight=args.storage_weight,
        construction_weight=args.construction_weight,
        query_weight=args.query_weight,
    )
    rendered = json.dumps(design.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

