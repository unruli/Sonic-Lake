from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import inf

from .schema import RepresentationProfile


@dataclass(frozen=True)
class Design:
    representations: tuple[str, ...]
    assignment: dict[str, str]
    storage_bytes_per_hour: float
    construction_s_per_hour: float
    expected_query_s: float
    objective: float

    def to_dict(self) -> dict:
        return asdict(self)


def optimize(
    profiles: dict[str, RepresentationProfile],
    frequencies: dict[str, float],
    thresholds: dict[str, float],
    *,
    archive_hours: float = 1.0,
    horizon_queries: float = 1.0,
    storage_weight: float = 1.0,
    construction_weight: float = 1.0,
    query_weight: float = 1.0,
    storage_budget_bytes: float | None = None,
    max_materializations: int | None = None,
) -> Design:
    if not profiles:
        raise ValueError("at least one representation profile is required")
    if archive_hours <= 0 or horizon_queries < 0:
        raise ValueError("archive_hours must be positive and horizon_queries nonnegative")
    if set(frequencies) != set(thresholds):
        raise ValueError("frequencies and thresholds must cover the same query families")
    if any(value < 0 for value in frequencies.values()):
        raise ValueError("query frequencies cannot be negative")

    names = sorted(profiles)
    limit = min(max_materializations or len(names), len(names))
    best: Design | None = None
    for width in range(1, limit + 1):
        for subset in combinations(names, width):
            storage = archive_hours * sum(profiles[name].storage_bytes_per_hour for name in subset)
            if storage_budget_bytes is not None and storage > storage_budget_bytes:
                continue
            construction = archive_hours * sum(profiles[name].construction_s_per_hour for name in subset)
            assignment: dict[str, str] = {}
            expected_query_s = 0.0
            feasible = True
            for family, frequency in frequencies.items():
                choices = [
                    (profiles[name].families[family].query_cost_s, name)
                    for name in subset
                    if family in profiles[name].families
                    and profiles[name].families[family].quality >= thresholds[family]
                ]
                if not choices:
                    feasible = False
                    break
                query_cost, chosen = min(choices)
                assignment[family] = chosen
                expected_query_s += frequency * query_cost * horizon_queries
            if not feasible:
                continue
            objective = (
                storage_weight * storage
                + construction_weight * construction
                + query_weight * expected_query_s
            )
            design = Design(
                representations=subset,
                assignment=assignment,
                storage_bytes_per_hour=storage / archive_hours,
                construction_s_per_hour=construction / archive_hours,
                expected_query_s=expected_query_s,
                objective=objective,
            )
            design_rank = (
                design.objective,
                len(design.representations),
                design.storage_bytes_per_hour,
                design.construction_s_per_hour,
                design.representations,
            )
            best_rank = None if best is None else (
                best.objective,
                len(best.representations),
                best.storage_bytes_per_hour,
                best.construction_s_per_hour,
                best.representations,
            )
            if best_rank is None or design_rank < best_rank:
                best = design
    if best is None:
        raise ValueError("no feasible physical design satisfies the supplied constraints")
    return best


def normalize_frequencies(frequencies: dict[str, float]) -> dict[str, float]:
    total = sum(frequencies.values())
    if total <= 0:
        raise ValueError("query frequencies must sum to a positive value")
    return {key: value / total for key, value in frequencies.items()}
