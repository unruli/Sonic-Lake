from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import fmean
from statistics import NormalDist
from typing import Iterable

from .optimizer import Design
from .schema import RepresentationProfile


def split_score(sample_id: str, seed: int = 0) -> float:
    """Map an immutable sample identifier to a reproducible value in [0, 1)."""
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def calibration_ids(
    sample_ids: Iterable[str],
    fraction: float,
    *,
    seed: int = 0,
    calibration_pool: float = 0.5,
) -> set[str]:
    if not 0 < fraction <= calibration_pool < 1:
        raise ValueError("require 0 < fraction <= calibration_pool < 1")
    return {sample_id for sample_id in sample_ids if split_score(sample_id, seed) < fraction}


def holdout_ids(
    sample_ids: Iterable[str],
    *,
    seed: int = 0,
    calibration_pool: float = 0.5,
) -> set[str]:
    if not 0 < calibration_pool < 1:
        raise ValueError("calibration_pool must be in (0, 1)")
    return {sample_id for sample_id in sample_ids if split_score(sample_id, seed) >= calibration_pool}


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    if not gold or len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must be nonempty and aligned")
    scores = []
    for label in sorted(set(gold)):
        tp = sum(actual == label and guess == label for actual, guess in zip(gold, predicted))
        fp = sum(actual != label and guess == label for actual, guess in zip(gold, predicted))
        fn = sum(actual == label and guess != label for actual, guess in zip(gold, predicted))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return fmean(scores)


def quality_from_rows(family: str, rows: list[dict]) -> float:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        raise ValueError(f"no successful rows for {family}")
    if family == "lexical":
        words = sum(int(row["reference_words"]) for row in ok)
        if words <= 0:
            raise ValueError("lexical rows contain no reference words")
        wer = sum(int(row["word_errors"]) for row in ok) / words
        return 1.0 - wer
    if family == "acoustic_event":
        return fmean(float(row["correct"]) for row in ok)
    if family == "emotion":
        return macro_f1(
            [str(row["gold"]) for row in ok],
            [str(row["prediction"]) for row in ok],
        )
    raise ValueError(f"unsupported family: {family}")


@dataclass(frozen=True)
class QualityMarginEstimate:
    point_margin: float
    standard_error: float | None
    lower_bound: float | None
    samples: int
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _aligned_successful_rows(source_rows: list[dict], candidate_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    source = {row["sample_id"]: row for row in source_rows if row.get("status") == "ok"}
    candidate = {row["sample_id"]: row for row in candidate_rows if row.get("status") == "ok"}
    if set(source) != set(candidate):
        raise ValueError("source and candidate rows must contain identical successful sample IDs")
    if not source:
        raise ValueError("quality-margin estimation requires successful paired rows")
    sample_ids = sorted(source)
    return [source[key] for key in sample_ids], [candidate[key] for key in sample_ids]


def _leave_one_out_quality(family: str, rows: list[dict]) -> list[float]:
    """Compute delete-one quality estimates from sufficient statistics."""
    count = len(rows)
    if count < 2:
        raise ValueError("jackknife estimation requires at least two rows")
    if family == "lexical":
        total_errors = sum(int(row["word_errors"]) for row in rows)
        total_words = sum(int(row["reference_words"]) for row in rows)
        estimates = []
        for row in rows:
            words = total_words - int(row["reference_words"])
            if words <= 0:
                raise ValueError("leave-one-out lexical sample contains no reference words")
            errors = total_errors - int(row["word_errors"])
            estimates.append(1.0 - errors / words)
        return estimates
    if family == "acoustic_event":
        total = sum(float(row["correct"]) for row in rows)
        return [(total - float(row["correct"])) / (count - 1) for row in rows]
    if family == "emotion":
        labels = sorted({str(row["gold"]) for row in rows})
        confusion = Counter((str(row["gold"]), str(row["prediction"])) for row in rows)
        estimates = []
        for removed in rows:
            removed_pair = (str(removed["gold"]), str(removed["prediction"]))
            confusion[removed_pair] -= 1
            scores = []
            for label in labels:
                true_positive = confusion[(label, label)]
                false_positive = sum(
                    value for (gold, predicted), value in confusion.items()
                    if gold != label and predicted == label
                )
                false_negative = sum(
                    value for (gold, predicted), value in confusion.items()
                    if gold == label and predicted != label
                )
                denominator = 2 * true_positive + false_positive + false_negative
                scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
            estimates.append(fmean(scores))
            confusion[removed_pair] += 1
        return estimates
    raise ValueError(f"unsupported family: {family}")


def paired_quality_margin(
    family: str,
    source_rows: list[dict],
    candidate_rows: list[dict],
    retention: float,
    *,
    confidence: float = 0.95,
) -> QualityMarginEstimate:
    """Estimate a one-sided paired jackknife bound for a retention constraint."""
    if not 0 <= retention <= 1:
        raise ValueError("retention must be in [0, 1]")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be in (0.5, 1)")
    source, candidate = _aligned_successful_rows(source_rows, candidate_rows)
    point_margin = quality_from_rows(family, candidate) - retention * quality_from_rows(family, source)
    count = len(source)
    if count < 2:
        return QualityMarginEstimate(point_margin, None, None, count, confidence)

    source_loo = _leave_one_out_quality(family, source)
    candidate_loo = _leave_one_out_quality(family, candidate)
    margin_loo = [other - retention * base for base, other in zip(source_loo, candidate_loo)]
    mean_margin = fmean(margin_loo)
    standard_error = sqrt((count - 1) / count * sum((value - mean_margin) ** 2 for value in margin_loo))
    lower_bound = point_margin - NormalDist().inv_cdf(confidence) * standard_error
    return QualityMarginEstimate(point_margin, standard_error, lower_bound, count, confidence)


@dataclass(frozen=True)
class RealizedDesign:
    representations: tuple[str, ...]
    assignment: dict[str, str]
    storage_bytes_per_hour: float
    construction_s_per_hour: float
    expected_query_s: float
    family_quality: dict[str, float]
    quality_margin: dict[str, float]
    violated_families: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def realize_design(
    design: Design,
    profiles: dict[str, RepresentationProfile],
    frequencies: dict[str, float],
    thresholds: dict[str, float],
) -> RealizedDesign:
    if set(frequencies) != set(thresholds) or set(frequencies) != set(design.assignment):
        raise ValueError("frequencies, thresholds, and assignments must cover the same families")
    missing = set(design.representations) - set(profiles)
    if missing:
        raise ValueError(f"profiles missing selected representations: {sorted(missing)}")

    family_quality: dict[str, float] = {}
    quality_margin: dict[str, float] = {}
    expected_query_s = 0.0
    for family, frequency in frequencies.items():
        name = design.assignment[family]
        item = profiles[name].families[family]
        family_quality[family] = item.quality
        quality_margin[family] = item.quality - thresholds[family]
        expected_query_s += frequency * item.query_cost_s

    violated = tuple(sorted(family for family, margin in quality_margin.items() if margin < 0))
    return RealizedDesign(
        representations=design.representations,
        assignment=dict(design.assignment),
        storage_bytes_per_hour=sum(profiles[name].storage_bytes_per_hour for name in design.representations),
        construction_s_per_hour=sum(profiles[name].construction_s_per_hour for name in design.representations),
        expected_query_s=expected_query_s,
        family_quality=family_quality,
        quality_margin=quality_margin,
        violated_families=violated,
    )


def layout_stability(calibration: Design, oracle: Design) -> dict[str, float | int]:
    selected = set(calibration.representations)
    target = set(oracle.representations)
    union = selected | target
    representation_jaccard = 1.0 if not union else len(selected & target) / len(union)
    assignment_matches = sum(
        calibration.assignment.get(family) == representation
        for family, representation in oracle.assignment.items()
    )
    return {
        "representation_jaccard": representation_jaccard,
        "assignment_matches": assignment_matches,
        "assignment_total": len(oracle.assignment),
    }


def quality_violation_counts(rows: Iterable[RealizedDesign]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.violated_families)
    return counts
