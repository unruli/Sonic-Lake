#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from soniclake.heldout import (
    calibration_ids,
    holdout_ids,
    layout_stability,
    paired_quality_margin,
    quality_from_rows,
    realize_design,
)
from soniclake.io import read_jsonl, write_jsonl
from soniclake.optimizer import normalize_frequencies, optimize
from soniclake.schema import FamilyProfile, RepresentationProfile


REPRESENTATIONS = {
    "pcm": "pcm_s16le",
    "flac": "flac_lossless_c8",
    "opus24": "opus_24kbps",
    "opus48": "opus_48kbps",
    "opus96": "opus_96kbps",
    "dac4": "dac_4",
    "dac8": "dac_8",
    "dac16": "dac_16",
    "encodec3": "encodec_3",
    "encodec6": "encodec_6",
    "encodec12": "encodec_12",
    "mimi8": "mimi_8",
    "mimi16": "mimi_16",
    "mimi32": "mimi_32",
}

FAMILIES = {
    "lexical": ("librispeech", "asr"),
    "acoustic_event": ("esc50", "events"),
    "emotion": ("crema_d", "emotion"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def query_seconds(family: str, row: dict) -> float:
    elapsed = float(row["inference_s"])
    if family == "emotion":
        elapsed += float(row["feature_extraction_s"])
    return elapsed


class ProductionMeasurements:
    def __init__(self, root: Path) -> None:
        production = root / "results" / "production"
        self.evaluation_root = production / "evaluation"
        self.encoding_root = production / "encoding"
        self.decode_root = production / "decode"
        self.source_rows: dict[str, list[dict]] = {}
        self.evaluation_rows: dict[tuple[str, str], list[dict]] = {}
        self.encoding_rows: dict[tuple[str, str], dict[str, dict]] = {}
        self.decode_rtf: dict[tuple[str, str], float] = {}
        self.names: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for family, (dataset, prefix) in FAMILIES.items():
            source_path = self.evaluation_root / f"{prefix}_source.jsonl"
            self.source_rows[family] = list(read_jsonl(source_path))
            source_ids = {row["sample_id"] for row in self.source_rows[family] if row["status"] == "ok"}
            if not source_ids:
                raise ValueError(f"no successful source rows for {family}")
            for short_name, encoding_suffix in REPRESENTATIONS.items():
                evaluation = list(read_jsonl(self.evaluation_root / f"{prefix}_{short_name}.jsonl"))
                encoding = {
                    row["sample_id"]: row
                    for row in read_jsonl(self.encoding_root / f"{dataset}_{encoding_suffix}.jsonl")
                    if row["status"] == "ok"
                }
                evaluation_ids = {row["sample_id"] for row in evaluation if row["status"] == "ok"}
                if evaluation_ids != source_ids or set(encoding) != source_ids:
                    raise ValueError(f"sample mismatch for {family}/{short_name}")
                decode_path = self.decode_root / f"{dataset}_{short_name}.summary.json"
                if dataset == "librispeech" and not decode_path.exists():
                    decode_path = self.decode_root / f"{short_name}.summary.json"
                self.evaluation_rows[(family, short_name)] = evaluation
                self.encoding_rows[(family, short_name)] = encoding
                self.decode_rtf[(family, short_name)] = float(load_json(decode_path)["aggregate_decode_rtf"])
                first = next(iter(encoding.values()))
                self.names[short_name] = f"{first['representation']}@{first['operating_point']}"

    def family_ids(self, family: str) -> set[str]:
        return {row["sample_id"] for row in self.source_rows[family] if row["status"] == "ok"}

    def source_quality(self, ids: dict[str, set[str]]) -> dict[str, float]:
        return {
            family: quality_from_rows(
                family,
                [row for row in rows if row["sample_id"] in ids[family]],
            )
            for family, rows in self.source_rows.items()
        }

    def profiles(self, ids: dict[str, set[str]]) -> dict[str, RepresentationProfile]:
        profiles: dict[str, RepresentationProfile] = {}
        for short_name in REPRESENTATIONS:
            families: dict[str, FamilyProfile] = {}
            artifacts: dict[tuple[str, str], dict] = {}
            for family in FAMILIES:
                selected_eval = [
                    row
                    for row in self.evaluation_rows[(family, short_name)]
                    if row["sample_id"] in ids[family]
                ]
                if not selected_eval:
                    raise ValueError(f"empty split for {family}/{short_name}")
                encoded = self.encoding_rows[(family, short_name)]
                rtf = self.decode_rtf[(family, short_name)]
                costs = []
                for row in selected_eval:
                    artifact = encoded[row["sample_id"]]
                    artifacts[(family, row["sample_id"])] = artifact
                    costs.append(query_seconds(family, row) + float(artifact["duration_s"]) * rtf)
                families[family] = FamilyProfile(
                    quality=quality_from_rows(family, selected_eval),
                    query_cost_s=fmean(costs),
                )
            artifact_rows = list(artifacts.values())
            hours = sum(float(row["duration_s"]) for row in artifact_rows) / 3600
            profile = RepresentationProfile(
                name=self.names[short_name],
                storage_bytes_per_hour=sum(int(row["artifact_bytes"]) for row in artifact_rows) / hours,
                construction_s_per_hour=sum(float(row["encode_s"]) for row in artifact_rows) / hours,
                families=families,
            )
            profile.validate()
            profiles[profile.name] = profile
        return profiles

    def quality_certificates(
        self,
        ids: dict[str, set[str]],
        retention: float,
        confidence: float,
    ) -> dict[str, dict[str, dict]]:
        certificates: dict[str, dict[str, dict]] = {}
        for short_name in REPRESENTATIONS:
            name = self.names[short_name]
            certificates[name] = {}
            for family in FAMILIES:
                source = [
                    row for row in self.source_rows[family]
                    if row["sample_id"] in ids[family]
                ]
                candidate = [
                    row for row in self.evaluation_rows[(family, short_name)]
                    if row["sample_id"] in ids[family]
                ]
                certificates[name][family] = paired_quality_margin(
                    family,
                    source,
                    candidate,
                    retention,
                    confidence=confidence,
                ).to_dict()
        return certificates


def conservative_profiles(
    profiles: dict[str, RepresentationProfile],
    certificates: dict[str, dict[str, dict]],
    min_samples: int,
) -> dict[str, RepresentationProfile]:
    result = {}
    for name, profile in profiles.items():
        families = {}
        for family, item in profile.families.items():
            certificate = certificates[name][family]
            lower_bound = certificate["lower_bound"]
            certified_margin = (
                float(lower_bound)
                if certificate["samples"] >= min_samples and lower_bound is not None
                else -1.0
            )
            families[family] = FamilyProfile(
                quality=certified_margin,
                query_cost_s=item.query_cost_s,
            )
        result[name] = RepresentationProfile(
            name=profile.name,
            storage_bytes_per_hour=profile.storage_bytes_per_hour,
            construction_s_per_hour=profile.construction_s_per_hour,
            families=families,
        )
    return result


def try_optimize(
    profiles: dict[str, RepresentationProfile],
    frequencies: dict[str, float],
    thresholds: dict[str, float],
    budget: float,
    max_materializations: int,
):
    try:
        return optimize(
            profiles,
            frequencies,
            thresholds,
            storage_budget_bytes=budget,
            max_materializations=max_materializations,
            storage_weight=0.0,
            construction_weight=0.0,
            query_weight=1.0,
        )
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate profile-trained layouts on a disjoint holdout split.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--selection-methods",
        nargs="+",
        choices=("point", "jackknife_lcb"),
        default=("point",),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-calibration-samples", type=int, default=30)
    parser.add_argument("--skip-shift", action="store_true")
    parser.add_argument(
        "--calibration-fractions",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
    )
    args = parser.parse_args()
    if args.min_calibration_samples < 2:
        parser.error("--min-calibration-samples must be at least 2")

    sweep = load_json(args.config)
    measurements = ProductionMeasurements(args.root)
    holdout = {
        family: holdout_ids(measurements.family_ids(family), seed=args.seed)
        for family in FAMILIES
    }
    test_profiles = measurements.profiles(holdout)
    test_source = measurements.source_quality(holdout)
    pcm_name = next(name for name in test_profiles if name.startswith("pcm_s16le@"))

    rows: list[dict] = []
    shift_rows: list[dict] = []
    for fraction in args.calibration_fractions:
        calibration = {
            family: calibration_ids(measurements.family_ids(family), fraction, seed=args.seed)
            for family in FAMILIES
        }
        calibration_samples = {family: len(ids) for family, ids in calibration.items()}
        if any(count == 0 for count in calibration_samples.values()):
            for retention in sweep["quality_retentions"]:
                for multiplier in sweep["budget_pcm_multipliers"]:
                    for policy, limit in (("single", 1), ("adaptive", 3)):
                        for train_name in sweep["workloads"]:
                            for method in args.selection_methods:
                                row = {
                                    "seed": args.seed,
                                    "selection_method": method,
                                    "calibration_fraction": fraction,
                                    "calibration_samples": calibration_samples,
                                    "holdout_samples": {k: len(v) for k, v in holdout.items()},
                                    "quality_retention": float(retention),
                                    "budget_pcm_multiplier": float(multiplier),
                                    "policy": policy,
                                    "max_materializations": limit,
                                    "training_workload": train_name,
                                    "status": "insufficient_calibration_samples",
                                }
                                if method == "jackknife_lcb":
                                    row.update({
                                        "confidence": args.confidence,
                                        "min_calibration_samples": args.min_calibration_samples,
                                    })
                                rows.append(row)
            continue
        calibration_profiles = measurements.profiles(calibration)
        calibration_source = measurements.source_quality(calibration)
        calibration_pcm = next(name for name in calibration_profiles if name.startswith("pcm_s16le@"))
        for retention in sweep["quality_retentions"]:
            train_thresholds = {
                family: quality * float(retention) for family, quality in calibration_source.items()
            }
            test_thresholds = {family: quality * float(retention) for family, quality in test_source.items()}
            selection_profiles = {"point": calibration_profiles}
            selection_thresholds = {"point": train_thresholds}
            certificates = None
            if "jackknife_lcb" in args.selection_methods:
                certificates = measurements.quality_certificates(
                    calibration,
                    float(retention),
                    args.confidence,
                )
                selection_profiles["jackknife_lcb"] = conservative_profiles(
                    calibration_profiles,
                    certificates,
                    args.min_calibration_samples,
                )
                selection_thresholds["jackknife_lcb"] = {family: 0.0 for family in FAMILIES}
            for multiplier in sweep["budget_pcm_multipliers"]:
                train_budget = calibration_profiles[calibration_pcm].storage_bytes_per_hour * float(multiplier)
                test_budget = test_profiles[pcm_name].storage_bytes_per_hour * float(multiplier)
                for policy, limit in (("single", 1), ("adaptive", 3)):
                    for train_name, raw_train_frequencies in sweep["workloads"].items():
                        train_frequencies = normalize_frequencies(raw_train_frequencies)
                        for method in args.selection_methods:
                            selected = try_optimize(
                                selection_profiles[method],
                                train_frequencies,
                                selection_thresholds[method],
                                train_budget,
                                limit,
                            )
                            base = {
                                "seed": args.seed,
                                "selection_method": method,
                                "calibration_fraction": fraction,
                                "calibration_samples": calibration_samples,
                                "holdout_samples": {k: len(v) for k, v in holdout.items()},
                                "quality_retention": float(retention),
                                "budget_pcm_multiplier": float(multiplier),
                                "policy": policy,
                                "max_materializations": limit,
                                "training_workload": train_name,
                            }
                            if method == "jackknife_lcb":
                                base.update({
                                    "confidence": args.confidence,
                                    "min_calibration_samples": args.min_calibration_samples,
                                })
                            if selected is None:
                                rows.append({**base, "status": "calibration_infeasible"})
                                continue
                            realized = realize_design(selected, test_profiles, train_frequencies, test_thresholds)
                            oracle = try_optimize(
                                test_profiles, train_frequencies, test_thresholds, test_budget, limit
                            )
                            row = {
                                **base,
                                "status": "ok",
                                "selected": selected.to_dict(),
                                "holdout": realized.to_dict(),
                                "holdout_budget_exceeded": realized.storage_bytes_per_hour > test_budget,
                            }
                            if method == "jackknife_lcb" and certificates is not None:
                                row["selected_certificates"] = {
                                    family: certificates[representation][family]
                                    for family, representation in selected.assignment.items()
                                }
                            if oracle is not None:
                                oracle_realized = realize_design(
                                    oracle, test_profiles, train_frequencies, test_thresholds
                                )
                                row["holdout_oracle"] = oracle.to_dict()
                                row["layout_stability"] = layout_stability(selected, oracle)
                                row["latency_regret_fraction"] = (
                                    realized.expected_query_s / oracle_realized.expected_query_s - 1.0
                                )
                            rows.append(row)

                            if args.skip_shift:
                                continue
                            for target_name, raw_target_frequencies in sweep["workloads"].items():
                                target_frequencies = normalize_frequencies(raw_target_frequencies)
                                shifted = realize_design(
                                    selected, test_profiles, target_frequencies, test_thresholds
                                )
                                target_oracle = try_optimize(
                                    test_profiles, target_frequencies, test_thresholds, test_budget, limit
                                )
                                shift = {
                                    **base,
                                    "status": "ok",
                                    "target_workload": target_name,
                                    "selected": selected.to_dict(),
                                    "holdout": shifted.to_dict(),
                                }
                                if target_oracle is not None:
                                    oracle_shifted = realize_design(
                                        target_oracle, test_profiles, target_frequencies, test_thresholds
                                    )
                                    shift["holdout_oracle"] = target_oracle.to_dict()
                                    shift["latency_regret_fraction"] = (
                                        shifted.expected_query_s / oracle_shifted.expected_query_s - 1.0
                                    )
                                shift_rows.append(shift)

    write_jsonl(args.output, rows)
    write_jsonl(args.shift_output, shift_rows)
    print(f"wrote {len(rows)} held-out rows and {len(shift_rows)} workload-shift rows")


if __name__ == "__main__":
    main()
