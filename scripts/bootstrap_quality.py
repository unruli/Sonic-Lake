#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from soniclake.io import read_jsonl


def representation_name(path: Path) -> str:
    stem = path.name.removesuffix(".jsonl")
    if "_" not in stem:
        raise ValueError(f"cannot infer representation from {path}")
    return stem.split("_", 1)[1]


def align(baseline: list[dict], target: list[dict]) -> tuple[list[dict], list[dict]]:
    base = {row["sample_id"]: row for row in baseline if row["status"] == "ok"}
    other = {row["sample_id"]: row for row in target if row["status"] == "ok"}
    if set(base) != set(other):
        raise ValueError(f"sample mismatch: baseline={len(base)}, target={len(other)}")
    ids = sorted(base)
    return [base[key] for key in ids], [other[key] for key in ids]


def macro_f1(gold: np.ndarray, predicted: np.ndarray) -> float:
    classes = np.unique(gold)
    scores = []
    for label in classes:
        true_positive = np.sum((gold == label) & (predicted == label))
        false_positive = np.sum((gold != label) & (predicted == label))
        false_negative = np.sum((gold == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def point_metric(family: str, rows: list[dict]) -> float:
    if family == "lexical":
        return sum(row["word_errors"] for row in rows) / sum(row["reference_words"] for row in rows)
    if family == "acoustic_event":
        return float(np.mean([row["correct"] for row in rows]))
    if family == "emotion":
        return macro_f1(
            np.asarray([row["gold"] for row in rows]),
            np.asarray([row["prediction"] for row in rows]),
        )
    raise ValueError(f"unsupported family: {family}")


def _metric_batch(family: str, rows: list[dict], indices: np.ndarray) -> np.ndarray:
    if family == "lexical":
        errors = np.asarray([row["word_errors"] for row in rows], dtype=float)
        words = np.asarray([row["reference_words"] for row in rows], dtype=float)
        return errors[indices].sum(axis=1) / words[indices].sum(axis=1)
    if family == "acoustic_event":
        correct = np.asarray([row["correct"] for row in rows], dtype=float)
        return correct[indices].mean(axis=1)
    if family == "emotion":
        gold = np.asarray([row["gold"] for row in rows])
        predicted = np.asarray([row["prediction"] for row in rows])
        sampled_gold = gold[indices]
        sampled_predicted = predicted[indices]
        scores = np.zeros(indices.shape[0], dtype=float)
        classes = np.unique(gold)
        for label in classes:
            true_positive = np.sum((sampled_gold == label) & (sampled_predicted == label), axis=1)
            false_positive = np.sum((sampled_gold != label) & (sampled_predicted == label), axis=1)
            false_negative = np.sum((sampled_gold == label) & (sampled_predicted != label), axis=1)
            denominator = 2 * true_positive + false_positive + false_negative
            scores += np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=denominator != 0,
            )
        return scores / len(classes)
    raise ValueError(f"unsupported family: {family}")


def paired_bootstrap(
    family: str,
    baseline: list[dict],
    target: list[dict],
    *,
    replicates: int,
    seed: int,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    base, other = align(baseline, target)
    rng = np.random.default_rng(seed)
    target_samples = []
    delta_samples = []
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        indices = rng.integers(0, len(base), size=(count, len(base)))
        base_metrics = _metric_batch(family, base, indices)
        target_metrics = _metric_batch(family, other, indices)
        target_samples.append(target_metrics)
        delta_samples.append(target_metrics - base_metrics)
    return np.concatenate(target_samples), np.concatenate(delta_samples)


def interval(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap quality intervals against source audio.")
    parser.add_argument("--family", required=True, choices=("lexical", "acoustic_event", "emotion"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    baseline = list(read_jsonl(args.baseline))
    baseline_ok = [row for row in baseline if row["status"] == "ok"]
    payload = {
        "family": args.family,
        "metric": {"lexical": "wer", "acoustic_event": "accuracy", "emotion": "macro_f1"}[args.family],
        "baseline": representation_name(args.baseline),
        "baseline_estimate": point_metric(args.family, baseline_ok),
        "replicates": args.replicates,
        "seed": args.seed,
        "comparisons": {},
    }
    for path in args.targets:
        target = list(read_jsonl(path))
        _, target_ok = align(baseline, target)
        samples, deltas = paired_bootstrap(
            args.family, baseline, target, replicates=args.replicates, seed=args.seed
        )
        name = representation_name(path)
        payload["comparisons"][name] = {
            "estimate": point_metric(args.family, target_ok),
            "ci95": interval(samples),
            "delta_vs_baseline": point_metric(args.family, target_ok) - payload["baseline_estimate"],
            "delta_ci95": interval(deltas),
            "samples": len(target_ok),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['comparisons'])} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
