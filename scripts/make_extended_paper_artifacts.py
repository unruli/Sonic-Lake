#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extended SonicLake paper artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = args.root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    analysis = root / "results/production/flac_analysis"

    # Calibration sensitivity: matched point/robust selections avoid comparing
    # methods on different feasible subsets.
    summary = read_json(analysis / "robust_multiseed_summary.json")["heldout"]["by_calibration_fraction"]
    fractions = sorted(float(key) for key in summary)
    point_success, robust_success, coverage, point_regret, robust_regret = [], [], [], [], []
    for fraction in fractions:
        matched = summary[str(fraction)]["matched"]
        point_success.append(matched["point"]["holdout_constraint_satisfaction_rate"])
        robust_success.append(matched["jackknife_lcb"]["holdout_constraint_satisfaction_rate"])
        coverage.append(matched["robust_coverage_of_point_feasible"])
        point_regret.append(matched["point"]["latency_regret_p95"])
        robust_regret.append(matched["jackknife_lcb"]["latency_regret_p95"])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    x = np.array(fractions) * 100
    axes[0].plot(x, np.array(point_success, dtype=float) * 100, "o-", label="Point")
    axes[0].plot(x, np.array(robust_success, dtype=float) * 100, "s-", label="Uncertainty-aware")
    axes[0].plot(x, np.array(coverage, dtype=float) * 100, "^--", label="Robust coverage")
    axes[0].set(xlabel="Calibration fraction (%)", ylabel="Matched rate (%)", ylim=(0, 103))
    axes[0].set_xscale("log")
    axes[0].set_xticks(x, [f"{v:g}" for v in x])
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, np.array(point_regret, dtype=float) * 100, "o-", label="Point")
    axes[1].plot(x, np.array(robust_regret, dtype=float) * 100, "s-", label="Uncertainty-aware")
    axes[1].set(xlabel="Calibration fraction (%)", ylabel="p95 latency regret (%)")
    axes[1].set_xscale("log")
    axes[1].set_xticks(x, [f"{v:g}" for v in x])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "calibration_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "calibration_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Workload-shift behavior for robust adaptive layouts at the retained
    # 10--50% calibration range, aggregated across five seeds and all regimes.
    shift_rows: list[dict] = []
    for path in sorted((analysis / "robust_multiseed").glob("seed_*_shift.jsonl")):
        shift_rows.extend(read_jsonl(path))
    workloads = ["balanced", "event_heavy", "emotion_heavy", "lexical_heavy"]
    pcm_storage = read_json(root / "results/production/profiles_flac/representation_profiles.json")["pcm_s16le@source_sr"]["storage_bytes_per_hour"]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in shift_rows:
        if (row.get("selection_method") == "jackknife_lcb" and row.get("policy") == "adaptive"
                and row.get("status") == "ok" and row["calibration_fraction"] in {0.1, 0.2, 0.5}):
            grouped[(row["training_workload"], row["target_workload"])].append(row)
    success = np.full((4, 4), np.nan)
    regret = np.full((4, 4), np.nan)
    counts = np.zeros((4, 4), dtype=int)
    for i, training in enumerate(workloads):
        for j, target in enumerate(workloads):
            rows = grouped[(training, target)]
            valid = []
            regrets = []
            for row in rows:
                budget = row["budget_pcm_multiplier"] * pcm_storage
                satisfies = not row["holdout"]["violated_families"] and row["holdout"]["storage_bytes_per_hour"] <= budget + 1e-6
                valid.append(satisfies)
                if satisfies and row["latency_regret_fraction"] is not None:
                    regrets.append(row["latency_regret_fraction"])
            if rows:
                counts[i, j] = len(rows)
                success[i, j] = 100 * sum(valid) / len(valid)
                regret[i, j] = 100 * quantile(regrets, 0.5)

    labels = [name.replace("_heavy", "").title() for name in workloads]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)
    images = [
        axes[0].imshow(success, vmin=80, vmax=100, cmap="viridis"),
        axes[1].imshow(regret, vmin=0, vmax=max(1, np.nanmax(regret)), cmap="magma_r"),
    ]
    titles = ["Held-out constraint satisfaction (%)", "Median valid latency regret (%)"]
    matrices = [success, regret]
    for axis, image, title, matrix in zip(axes, images, titles, matrices):
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(4), labels, rotation=25, ha="right")
        axis.set_yticks(range(4), labels)
        axis.set_xlabel("Target workload")
        axis.set_ylabel("Training workload")
        for i in range(4):
            for j in range(4):
                rgba = image.cmap(image.norm(matrix[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                          color="black" if luminance > 0.55 else "white")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output / "workload_shift.pdf", bbox_inches="tight")
    fig.savefig(output / "workload_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Empirical break-even CDF for valid adaptive wins.
    break_rows = read_jsonl(analysis / "break_even_with_flac.jsonl")
    fig, axis = plt.subplots(figsize=(6.7, 3.7), constrained_layout=True)
    break_summary = {}
    for policy, label, marker in [
        ("point_adaptive", "Point adaptive", "o"),
        ("robust_adaptive", "Uncertainty-aware adaptive", "s"),
    ]:
        values = sorted(
            row["construction_break_even_queries_per_audio_hour"] for row in break_rows
            if row["candidate_policy"] == policy and row["baseline_constraint_satisfied"]
            and row["candidate_constraint_satisfied"] and row["candidate_is_faster"]
            and row["construction_break_even_queries_per_audio_hour"] is not None
        )
        y = np.arange(1, len(values) + 1) / len(values)
        axis.plot(values, y, label=f"{label} (n={len(values)})", marker=marker, markevery=max(1, len(values) // 12), markersize=3)
        break_summary[policy] = {
            "count": len(values), "median": quantile(values, 0.5), "p95": quantile(values, 0.95),
            "fraction_by_1000": sum(value <= 1000 for value in values) / len(values),
        }
    axis.axvline(1_000, color="#666666", linestyle="--", linewidth=1, label="1,000 queries/hour")
    axis.set_xscale("log")
    axis.set(xlabel="Queries per archived audio-hour to break even", ylabel="Fraction of valid adaptive wins")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "break_even_cdf.pdf", bbox_inches="tight")
    fig.savefig(output / "break_even_cdf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    artifact_summary = {
        "calibration_fractions": fractions,
        "workload_shift_counts": counts.tolist(),
        "workload_shift_success_percent": success.tolist(),
        "workload_shift_median_regret_percent": regret.tolist(),
        "break_even": break_summary,
    }
    (output / "extended_artifact_summary.json").write_text(
        json.dumps(artifact_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
