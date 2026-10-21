#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from soniclake.io import read_jsonl


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_sweep(rows: list[dict]) -> dict:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    representation_counts: dict[str, Counter] = defaultdict(Counter)
    assignment_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["workload"], row["quality_retention"], row["budget_pcm_multiplier"])
        grouped[key][row["policy"]] = row
        representation_counts[row["policy"]].update(row["representations"])
        for family, representation in row["assignment"].items():
            assignment_counts[f"{row['policy']}:{family}"][representation] += 1

    paired = [policies for policies in grouped.values() if {"single", "adaptive"} <= set(policies)]
    reductions = []
    for policies in paired:
        single = float(policies["single"]["expected_query_s"])
        adaptive = float(policies["adaptive"]["expected_query_s"])
        reductions.append((single - adaptive) / single)
    wins = [value for value in reductions if value > 1e-12]
    return {
        "conditions": 168,
        "single_feasible": sum(row["status"] == "ok" and row["policy"] == "single" for row in rows),
        "adaptive_feasible": sum(row["status"] == "ok" and row["policy"] == "adaptive" for row in rows),
        "both_feasible": len(paired),
        "adaptive_wins": len(wins),
        "adaptive_win_rate_when_both_feasible": len(wins) / len(paired) if paired else None,
        "winning_reduction_median": float(np.median(wins)) if wins else None,
        "winning_reduction_p95": float(np.quantile(wins, 0.95)) if wins else None,
        "winning_reduction_max": max(wins) if wins else None,
        "materialization_counts": {
            policy: dict(counts.most_common()) for policy, counts in representation_counts.items()
        },
        "assignment_counts": {
            key: dict(counts.most_common()) for key, counts in assignment_counts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FLAC and retrieval-aware SonicLake extensions.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    production = args.root / "results" / "production"
    profiles = load_json(production / "profiles_flac" / "representation_profiles.json")
    retrieval_profiles = load_json(
        production / "profiles_flac" / "representation_profiles_retrieval.json"
    )
    retrieval = {
        path.name.removesuffix(".summary.json"): load_json(path)
        for path in sorted((production / "retrieval").glob("*.summary.json"))
    }
    payload = {
        "flac_profile": profiles["flac@lossless_c8"],
        "flac_retrieval_profile": retrieval_profiles["flac@lossless_c8"],
        "audio_scan_sweep": summarize_sweep(
            list(read_jsonl(production / "optimizer_flac" / "layout_sweep.jsonl"))
        ),
        "retrieval_aware_sweep": summarize_sweep(
            list(read_jsonl(production / "optimizer_retrieval" / "layout_sweep.jsonl"))
        ),
        "retrieval": {
            name: {
                key: summary[key]
                for key in (
                    "macro_precision", "macro_recall", "macro_f1", "index_bytes_per_audio_hour",
                    "transcription_s_per_audio_hour", "median_lookup_ns", "queries", "documents",
                )
            }
            for name, summary in retrieval.items()
        },
        "policy_baselines": load_json(
            production / "flac_analysis" / "policy_baselines_with_flac_summary.json"
        ),
        "robust_selection": load_json(
            production / "flac_analysis" / "robust_multiseed_summary.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "flac_profile": payload["flac_profile"],
        "audio_scan_sweep": payload["audio_scan_sweep"],
        "retrieval_aware_sweep": payload["retrieval_aware_sweep"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
