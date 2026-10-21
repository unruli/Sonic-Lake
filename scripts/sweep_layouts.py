#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soniclake.io import write_jsonl
from soniclake.optimizer import normalize_frequencies, optimize
from soniclake.profiles import profiles_from_dict


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def source_quality(analysis_root: Path) -> dict[str, float]:
    lexical = load_json(analysis_root / "quality_lexical.json")
    events = load_json(analysis_root / "quality_events.json")
    emotion = load_json(analysis_root / "quality_emotion.json")
    if lexical["metric"] != "wer":
        raise ValueError("lexical analysis must report WER")
    return {
        "lexical": 1.0 - float(lexical["baseline_estimate"]),
        "acoustic_event": float(events["baseline_estimate"]),
        "emotion": float(emotion["baseline_estimate"]),
    }


def run_sweep(
    profiles_payload: dict,
    analysis_root: Path,
    sweep: dict,
    source_override: dict[str, float] | None = None,
) -> list[dict]:
    profiles = profiles_from_dict(profiles_payload)
    pcm_names = [name for name in profiles if name.startswith("pcm_s16le@")]
    if len(pcm_names) != 1:
        raise ValueError(f"expected one PCM profile, found {pcm_names}")
    pcm_bytes_per_hour = profiles[pcm_names[0]].storage_bytes_per_hour
    source = source_override if source_override is not None else source_quality(analysis_root)
    rows = []
    for workload_name, raw_frequencies in sweep["workloads"].items():
        frequencies = normalize_frequencies(raw_frequencies)
        for retention in sweep["quality_retentions"]:
            thresholds = {family: source[family] * float(retention) for family in frequencies}
            for multiplier in sweep["budget_pcm_multipliers"]:
                budget = pcm_bytes_per_hour * float(multiplier)
                for policy, max_materializations in (("single", 1), ("adaptive", 3)):
                    base = {
                        "workload": workload_name,
                        "quality_retention": float(retention),
                        "budget_pcm_multiplier": float(multiplier),
                        "storage_budget_bytes_per_hour": budget,
                        "policy": policy,
                        "max_materializations": max_materializations,
                        "thresholds": thresholds,
                    }
                    try:
                        design = optimize(
                            profiles,
                            frequencies,
                            thresholds,
                            storage_budget_bytes=budget,
                            max_materializations=max_materializations,
                            storage_weight=0.0,
                            construction_weight=0.0,
                            query_weight=1.0,
                        )
                        rows.append({**base, "status": "ok", **design.to_dict()})
                    except ValueError as exc:
                        rows.append({**base, "status": "infeasible", "error": str(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep workload, quality-retention, and storage-budget regimes.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-quality-json",
        type=Path,
        help="Optional direct family-to-quality mapping for an alternate query workload.",
    )
    args = parser.parse_args()

    source_override = load_json(args.source_quality_json) if args.source_quality_json else None
    rows = run_sweep(
        load_json(args.profiles), args.analysis_root, load_json(args.config), source_override
    )
    write_jsonl(args.output, rows)
    feasible = sum(row["status"] == "ok" for row in rows)
    print(f"wrote {len(rows)} sweep rows; feasible={feasible}, infeasible={len(rows) - feasible}")


if __name__ == "__main__":
    main()
