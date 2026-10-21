#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_heldout_design import (
    FAMILIES,
    ProductionMeasurements,
    conservative_profiles,
    load_json,
    try_optimize,
)
from soniclake.baselines import (
    construction_break_even,
    deterministic_random_feasible_single,
    fixed_representation_design,
)
from soniclake.heldout import calibration_ids, holdout_ids, realize_design
from soniclake.io import write_jsonl
from soniclake.optimizer import normalize_frequencies


def named_representation(profiles: dict, prefix: str) -> str:
    matches = [name for name in profiles if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one representation with prefix {prefix!r}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SonicLake with explicit physical-design baselines.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--break-even-output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(2027, 2028, 2029, 2030, 2031))
    parser.add_argument("--calibration-fractions", type=float, nargs="+", default=(0.10, 0.20, 0.50))
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-calibration-samples", type=int, default=10)
    args = parser.parse_args()

    sweep = load_json(args.config)
    measurements = ProductionMeasurements(args.root)
    rows: list[dict] = []
    break_even_rows: list[dict] = []

    for seed in args.seeds:
        holdout = {
            family: holdout_ids(measurements.family_ids(family), seed=seed)
            for family in FAMILIES
        }
        test_profiles = measurements.profiles(holdout)
        test_source = measurements.source_quality(holdout)
        test_pcm = named_representation(test_profiles, "pcm_s16le@")

        for fraction in args.calibration_fractions:
            calibration = {
                family: calibration_ids(measurements.family_ids(family), fraction, seed=seed)
                for family in FAMILIES
            }
            calibration_profiles = measurements.profiles(calibration)
            calibration_source = measurements.source_quality(calibration)
            calibration_pcm = named_representation(calibration_profiles, "pcm_s16le@")
            calibration_flac = named_representation(calibration_profiles, "flac@lossless_c8")
            calibration_opus48 = named_representation(calibration_profiles, "opus@48kbps")
            smallest = min(
                calibration_profiles,
                key=lambda name: (calibration_profiles[name].storage_bytes_per_hour, name),
            )

            for retention in sweep["quality_retentions"]:
                retention = float(retention)
                train_thresholds = {
                    family: quality * retention for family, quality in calibration_source.items()
                }
                test_thresholds = {
                    family: quality * retention for family, quality in test_source.items()
                }
                certificates = measurements.quality_certificates(
                    calibration, retention, args.confidence
                )
                robust_profiles = conservative_profiles(
                    calibration_profiles, certificates, args.min_calibration_samples
                )
                robust_thresholds = {family: 0.0 for family in FAMILIES}

                for multiplier in sweep["budget_pcm_multipliers"]:
                    multiplier = float(multiplier)
                    train_budget = (
                        calibration_profiles[calibration_pcm].storage_bytes_per_hour * multiplier
                    )
                    test_budget = test_profiles[test_pcm].storage_bytes_per_hour * multiplier
                    for workload_name, raw_frequencies in sweep["workloads"].items():
                        frequencies = normalize_frequencies(raw_frequencies)
                        uniform = {family: 1.0 / len(FAMILIES) for family in FAMILIES}
                        oracle = try_optimize(
                            test_profiles, frequencies, test_thresholds, test_budget, 3
                        )
                        selections = {
                            "pcm": fixed_representation_design(
                                calibration_pcm,
                                calibration_profiles,
                                frequencies,
                                train_thresholds,
                                train_budget,
                            ),
                            "flac": fixed_representation_design(
                                calibration_flac,
                                calibration_profiles,
                                frequencies,
                                train_thresholds,
                                train_budget,
                            ),
                            "opus48": fixed_representation_design(
                                calibration_opus48,
                                calibration_profiles,
                                frequencies,
                                train_thresholds,
                                train_budget,
                            ),
                            "smallest": fixed_representation_design(
                                smallest,
                                calibration_profiles,
                                frequencies,
                                train_thresholds,
                                train_budget,
                            ),
                            "random_feasible_single": deterministic_random_feasible_single(
                                calibration_profiles,
                                frequencies,
                                train_thresholds,
                                train_budget,
                                key=(
                                    f"{seed}|{fraction}|{retention}|{multiplier}|{workload_name}"
                                ),
                            ),
                            "workload_agnostic_single": try_optimize(
                                calibration_profiles, uniform, train_thresholds, train_budget, 1
                            ),
                            "point_single": try_optimize(
                                calibration_profiles, frequencies, train_thresholds, train_budget, 1
                            ),
                            "point_adaptive": try_optimize(
                                calibration_profiles, frequencies, train_thresholds, train_budget, 3
                            ),
                            "robust_adaptive": try_optimize(
                                robust_profiles, frequencies, robust_thresholds, train_budget, 3
                            ),
                            "holdout_oracle_adaptive": oracle,
                        }
                        base = {
                            "seed": seed,
                            "calibration_fraction": fraction,
                            "calibration_samples": {k: len(v) for k, v in calibration.items()},
                            "holdout_samples": {k: len(v) for k, v in holdout.items()},
                            "quality_retention": retention,
                            "budget_pcm_multiplier": multiplier,
                            "training_workload": workload_name,
                        }
                        oracle_realized = None
                        if oracle is not None:
                            oracle_realized = realize_design(
                                oracle, test_profiles, frequencies, test_thresholds
                            )

                        realized_by_policy = {}
                        for policy, design in selections.items():
                            row = {**base, "policy": policy}
                            if design is None:
                                rows.append({**row, "status": "infeasible"})
                                continue
                            realized = realize_design(
                                design, test_profiles, frequencies, test_thresholds
                            )
                            realized_by_policy[policy] = realized
                            row.update({
                                "status": "ok",
                                "selected": design.to_dict(),
                                "holdout": realized.to_dict(),
                                "holdout_constraint_satisfied": not realized.violated_families,
                                "holdout_budget_exceeded": (
                                    realized.storage_bytes_per_hour > test_budget
                                ),
                            })
                            if oracle_realized is not None:
                                row["latency_regret_fraction"] = (
                                    realized.expected_query_s / oracle_realized.expected_query_s - 1.0
                                )
                            if policy == "robust_adaptive":
                                row["selected_certificates"] = {
                                    family: certificates[representation][family]
                                    for family, representation in design.assignment.items()
                                }
                            rows.append(row)

                        baseline = realized_by_policy.get("point_single")
                        if baseline is not None:
                            for candidate_policy in ("point_adaptive", "robust_adaptive"):
                                candidate = realized_by_policy.get(candidate_policy)
                                if candidate is None:
                                    continue
                                break_even_rows.append({
                                    **base,
                                    "baseline_policy": "point_single",
                                    "candidate_policy": candidate_policy,
                                    "baseline_constraint_satisfied": not baseline.violated_families,
                                    "candidate_constraint_satisfied": not candidate.violated_families,
                                    **construction_break_even(baseline, candidate),
                                })

    write_jsonl(args.output, rows)
    write_jsonl(args.break_even_output, break_even_rows)
    print(
        f"wrote {len(rows)} policy rows and {len(break_even_rows)} break-even rows"
    )


if __name__ == "__main__":
    main()
