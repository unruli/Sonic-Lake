#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(rows: list[dict], count: int = 6) -> dict:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["workload"], row["quality_retention"], row["budget_pcm_multiplier"])
        grouped.setdefault(key, {})[row["policy"]] = row
    candidates = []
    prefixes = ("pcm_s16le@", "flac@", "opus@")
    for key, policies in grouped.items():
        if not {"single", "adaptive"}.issubset(policies):
            continue
        adaptive = policies["adaptive"]
        if not all(name.startswith(prefixes) for name in adaptive["representations"]):
            continue
        single = policies["single"]
        gain = (single["expected_query_s"] - adaptive["expected_query_s"]) / single["expected_query_s"]
        candidates.append((gain, key, single, adaptive))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible regimes for requested count={count}")
    indices = sorted({round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)})
    chosen = []
    for rank, index in enumerate(indices):
        gain, key, single, adaptive = candidates[index]
        chosen.append({
            "id": f"r{rank + 1}",
            "workload": key[0],
            "quality_retention": key[1],
            "budget_pcm_multiplier": key[2],
            "predicted_latency_reduction": gain,
            "single": {
                "representations": single["representations"],
                "assignment": single["assignment"],
            },
            "adaptive": {
                "representations": adaptive["representations"],
                "assignment": adaptive["assignment"],
            },
        })
    return {
        "selection_method": "even quantiles of predicted adaptive gain among PCM/FLAC/Opus-only feasible pairs",
        "source_rows": len(rows),
        "eligible_pairs": len(candidates),
        "regimes": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze representative live-serving regimes.")
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.layouts.read_text(encoding="utf-8").splitlines() if line]
    payload = select(rows, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
