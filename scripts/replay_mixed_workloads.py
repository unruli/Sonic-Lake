#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from soniclake.router import CatalogRouter


def candidate_name(row: dict) -> str:
    return f'{row["representation"]}@{row["operating_point"]}'


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def load_measurements(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                grouped[(candidate_name(row), row["query_family"])].append(row)
    return dict(grouped)


def load_lookups(directory: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for summary_path in directory.glob("*.summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = candidate_name(summary)
        rows_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
        if rows_path.exists():
            values = [json.loads(line)["lookup_ns"] / 1e9 for line in rows_path.read_text(encoding="utf-8").splitlines()]
            if values:
                result[name] = values
    return result


def draw_trace(frequencies: dict[str, float], count: int, seed: int) -> list[str]:
    families = sorted(frequencies)
    weights = [frequencies[name] for name in families]
    return random.Random(seed).choices(families, weights=weights, k=count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay mixed query traces through SonicLake catalogs.")
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queries <= 0:
        raise ValueError("queries must be positive")

    measurements = load_measurements(args.measurements)
    lookups = load_lookups(args.retrieval_dir) if args.retrieval_dir else {}
    workload_spec = json.loads(args.workloads.read_text(encoding="utf-8"))["workloads"]
    layouts = [json.loads(line) for line in args.layouts.read_text(encoding="utf-8").splitlines()]
    traces = {name: draw_trace(freq, args.queries, args.seed) for name, freq in workload_spec.items()}

    output_rows = []
    for catalog in layouts:
        if catalog.get("status") != "ok":
            continue
        workload = catalog["workload"]
        router = CatalogRouter(catalog["assignment"], measurements, lookups)
        routed = [router.route(family) for family in traces[workload]]
        overheads = [item.router_ns for item in routed]
        executions = [item.measured_execution_s for item in routed]
        output_rows.append({
            "workload": workload,
            "policy": catalog["policy"],
            "quality_retention": catalog["quality_retention"],
            "budget_pcm_multiplier": catalog["budget_pcm_multiplier"],
            "queries": len(routed),
            "representations": catalog["representations"],
            "assignment": catalog["assignment"],
            "router_median_ns": statistics.median(overheads),
            "router_p95_ns": percentile(overheads, 0.95),
            "measured_execution_total_s": sum(executions),
            "measured_execution_mean_s": statistics.mean(executions),
        })

    summary = {
        "method": "catalog control-path execution with measured per-record latency replay",
        "seed": args.seed,
        "queries_per_catalog": args.queries,
        "catalogs_replayed": len(output_rows),
        "router_median_ns_across_catalogs": statistics.median(row["router_median_ns"] for row in output_rows),
        "router_p95_ns_across_catalogs": percentile([row["router_p95_ns"] for row in output_rows], 0.95),
        "catalogs": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "catalogs"}, indent=2))


if __name__ == "__main__":
    main()
