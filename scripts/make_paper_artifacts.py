#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FAMILY_LABELS = {
    "lexical": "Lexical quality (1 − WER)",
    "acoustic_event": "Event accuracy",
    "emotion": "Emotion macro-F1",
}
DATASET_LABELS = {"librispeech": "LibriSpeech", "esc50": "ESC-50", "crema_d": "CREMA-D"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def display_name(name: str) -> str:
    representation, point = name.split("@", 1)
    labels = {
        "pcm_s16le": "PCM",
        "flac": "FLAC",
        "opus": "Opus",
        "dac": "DAC",
        "encodec": "EnCodec",
        "mimi": "Mimi",
    }
    return f"{labels.get(representation, representation)} {point.replace('source_sr', '')}".strip()


def family_group(name: str) -> str:
    return name.split("@", 1)[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready SonicLake result tables and figures.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--decode-root", type=Path)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    profiles_path = args.profiles or (
        args.root / "results/production/profiles/representation_profiles.json"
    )
    sweep_path = args.sweep or (args.root / "results/production/optimizer/layout_sweep.jsonl")
    analysis_root = args.analysis_root or (args.root / "results/production/analysis")
    decode_root = args.decode_root or (args.root / "results/production/decode")
    profiles = load_json(profiles_path)
    sweep = read_jsonl(sweep_path)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    profile_rows = []
    for name, profile in sorted(profiles.items(), key=lambda item: item[1]["storage_bytes_per_hour"]):
        row = {
            "representation": display_name(name),
            "profile_key": name,
            "storage_mb_per_audio_hour": profile["storage_bytes_per_hour"] / 1_000_000,
            "construction_s_per_audio_hour": profile["construction_s_per_hour"],
        }
        for family in FAMILY_LABELS:
            row[f"{family}_quality"] = profile["families"][family]["quality"]
            row[f"{family}_query_s"] = profile["families"][family]["query_cost_s"]
        profile_rows.append(row)
    write_csv(output / "representation_profiles.csv", profile_rows)

    quality_analyses = {
        "lexical": load_json(analysis_root / "quality_lexical.json"),
        "acoustic_event": load_json(analysis_root / "quality_events.json"),
        "emotion": load_json(analysis_root / "quality_emotion.json"),
    }
    short_names = {
        "pcm_s16le@source_sr": "pcm", "flac@lossless_c8": "flac",
        "opus@24kbps": "opus24", "opus@48kbps": "opus48",
        "opus@96kbps": "opus96", "dac@4cb": "dac4", "dac@8cb": "dac8", "dac@16cb": "dac16",
        "encodec@3kbps": "encodec3", "encodec@6kbps": "encodec6", "encodec@12kbps": "encodec12",
        "mimi@8cb": "mimi8", "mimi@16cb": "mimi16", "mimi@32cb": "mimi32",
    }
    quality_ci_rows = []
    quality_ci: dict[tuple[str, str], tuple[float, float]] = {}
    for family, analysis in quality_analyses.items():
        for profile_key, short in short_names.items():
            comparison = analysis["comparisons"][short]
            low, high = comparison["ci95"]
            estimate = comparison["estimate"]
            if family == "lexical":
                estimate, low, high = 1.0 - estimate, 1.0 - high, 1.0 - low
            quality_ci[(family, profile_key)] = (low, high)
            quality_ci_rows.append({
                "query_family": family,
                "representation": display_name(profile_key),
                "profile_key": profile_key,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "delta_vs_source": comparison["delta_vs_baseline"],
                "delta_ci95_low": comparison["delta_ci95"][0],
                "delta_ci95_high": comparison["delta_ci95"][1],
            })
    write_csv(output / "quality_confidence_intervals.csv", quality_ci_rows)

    decode_rows = []
    for dataset in DATASET_LABELS:
        for key in profiles:
            short = short_names[key]
            path = decode_root / f"{dataset}_{short}.summary.json"
            if dataset == "librispeech" and not path.exists():
                path = decode_root / f"{short}.summary.json"
            summary = load_json(path)
            decode_rows.append({
                "dataset": DATASET_LABELS[dataset],
                "representation": display_name(key),
                "profile_key": key,
                "aggregate_decode_rtf": summary["aggregate_decode_rtf"],
                "median_decode_ms": 1000 * summary["median_decode_s"],
                "p95_decode_ms": 1000 * summary["p95_decode_s"],
            })
    write_csv(output / "decode_latency.csv", decode_rows)

    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in sweep:
        key = (row["workload"], row["quality_retention"], row["budget_pcm_multiplier"])
        groups[key][row["policy"]] = row
    comparisons = []
    for key, policies in groups.items():
        single, adaptive = policies["single"], policies["adaptive"]
        if single["status"] == adaptive["status"] == "ok":
            gain = (single["expected_query_s"] - adaptive["expected_query_s"]) / single["expected_query_s"]
            comparisons.append((gain, key, single, adaptive))
    wins = [item for item in comparisons if item[0] > 1e-9]
    optimizer_summary = {
        "conditions": len(groups),
        "both_feasible": len(comparisons),
        "adaptive_wins": len(wins),
        "max_latency_reduction": max(item[0] for item in comparisons),
        "median_winning_latency_reduction": float(np.median([item[0] for item in wins])),
        "top_conditions": [
            {
                "latency_reduction": gain,
                "workload": key[0],
                "quality_retention": key[1],
                "budget_pcm_multiplier": key[2],
                "single": single["representations"],
                "adaptive": adaptive["representations"],
                "assignment": adaptive["assignment"],
            }
            for gain, key, single, adaptive in sorted(comparisons, reverse=True)[:10]
        ],
    }
    (output / "optimizer_summary.json").write_text(
        json.dumps(optimizer_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    colors = {
        "pcm_s16le": "#333333", "flac": "#56B4E9", "opus": "#0072B2",
        "dac": "#D55E00", "encodec": "#009E73", "mimi": "#CC79A7",
    }
    markers = {
        "pcm_s16le": "s", "flac": "v", "opus": "o", "dac": "^",
        "encodec": "D", "mimi": "P",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    for axis, family in zip(axes, FAMILY_LABELS):
        seen = set()
        for row in profile_rows:
            group = family_group(row["profile_key"])
            estimate = row[f"{family}_quality"]
            low, high = quality_ci[(family, row["profile_key"])]
            axis.errorbar(
                row["storage_mb_per_audio_hour"], estimate,
                yerr=[[estimate - low], [high - estimate]],
                color=colors[group], marker=markers[group], markersize=7,
                capsize=2, linewidth=1, linestyle="none",
            )
            seen.add(group)
        axis.set_xscale("log")
        axis.set_xlabel("Storage (MB / audio-hour)")
        axis.set_ylabel(FAMILY_LABELS[family])
        axis.grid(True, which="both", alpha=0.25)
    legend_order = ("pcm_s16le", "flac", "opus", "dac", "encodec", "mimi")
    legend_names = {
        "pcm_s16le": "PCM", "flac": "FLAC", "opus": "Opus", "dac": "DAC",
        "encodec": "EnCodec", "mimi": "Mimi",
    }
    handles = [
        Line2D([0], [0], color=colors[group], marker=markers[group], linestyle="none", label=legend_names[group])
        for group in legend_order
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.savefig(output / "quality_storage_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(output / "quality_storage_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 4.3), constrained_layout=True)
    keys = [row["profile_key"] for row in profile_rows]
    x = np.arange(len(keys))
    width = 0.25
    for offset, (dataset, label) in enumerate(DATASET_LABELS.items()):
        values = []
        for key in keys:
            match = next(row for row in decode_rows if row["dataset"] == label and row["profile_key"] == key)
            values.append(match["aggregate_decode_rtf"])
        axis.bar(x + (offset - 1) * width, values, width=width, label=label)
    axis.set_yscale("log")
    axis.set_ylabel("Decode real-time factor (log scale)")
    axis.set_xticks(x, [display_name(key) for key in keys], rotation=45, ha="right", fontsize=8)
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output / "decode_rtf.pdf", bbox_inches="tight")
    fig.savefig(output / "decode_rtf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    workloads = list(load_json(args.root / "config/workload_sweep.json")["workloads"])
    retentions = sorted({row["quality_retention"] for row in sweep})
    budgets = sorted({row["budget_pcm_multiplier"] for row in sweep})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    heatmap_cmap = plt.get_cmap("viridis").copy()
    heatmap_cmap.set_bad("#e6e6e6")
    for axis, workload in zip(axes.flat, workloads):
        matrix = np.full((len(retentions), len(budgets)), np.nan)
        for i, retention in enumerate(retentions):
            for j, budget in enumerate(budgets):
                policies = groups[(workload, retention, budget)]
                if policies["single"]["status"] == policies["adaptive"]["status"] == "ok":
                    single = policies["single"]["expected_query_s"]
                    adaptive = policies["adaptive"]["expected_query_s"]
                    matrix[i, j] = 100 * (single - adaptive) / single
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=35, cmap=heatmap_cmap)
        axis.set_title(workload.replace("_", " ").title())
        axis.set_xticks(range(len(budgets)), [str(value) for value in budgets], rotation=45)
        axis.set_yticks(range(len(retentions)), [str(value) for value in retentions])
        axis.set_xlabel("Storage budget (× PCM)")
        axis.set_ylabel("Quality retention")
    fig.colorbar(image, ax=axes, label="Adaptive latency reduction (%)", shrink=0.82)
    fig.text(0.5, -0.01, "Gray cells are infeasible under the quality and storage constraints.", ha="center", fontsize=9)
    fig.savefig(output / "optimizer_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "optimizer_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote paper artifacts to {output}")


if __name__ == "__main__":
    main()
