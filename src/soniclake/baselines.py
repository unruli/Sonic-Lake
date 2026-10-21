from __future__ import annotations

import hashlib

from .heldout import RealizedDesign
from .optimizer import Design, optimize
from .schema import RepresentationProfile


def fixed_representation_design(
    name: str,
    profiles: dict[str, RepresentationProfile],
    frequencies: dict[str, float],
    thresholds: dict[str, float],
    storage_budget_bytes: float,
) -> Design | None:
    if name not in profiles:
        raise ValueError(f"unknown representation: {name}")
    try:
        return optimize(
            {name: profiles[name]},
            frequencies,
            thresholds,
            storage_budget_bytes=storage_budget_bytes,
            max_materializations=1,
            storage_weight=0.0,
            construction_weight=0.0,
            query_weight=1.0,
        )
    except ValueError:
        return None


def deterministic_random_feasible_single(
    profiles: dict[str, RepresentationProfile],
    frequencies: dict[str, float],
    thresholds: dict[str, float],
    storage_budget_bytes: float,
    key: str,
) -> Design | None:
    feasible = [
        design
        for name in sorted(profiles)
        if (design := fixed_representation_design(
            name, profiles, frequencies, thresholds, storage_budget_bytes
        )) is not None
    ]
    if not feasible:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return feasible[int.from_bytes(digest[:8], "big") % len(feasible)]


def construction_break_even(
    baseline: RealizedDesign,
    candidate: RealizedDesign,
) -> dict[str, float | None | bool]:
    extra_construction = candidate.construction_s_per_hour - baseline.construction_s_per_hour
    extra_storage = candidate.storage_bytes_per_hour - baseline.storage_bytes_per_hour
    query_saving = baseline.expected_query_s - candidate.expected_query_s
    if query_saving <= 0:
        break_even = None
    else:
        break_even = max(0.0, extra_construction) / query_saving
    return {
        "extra_construction_s_per_audio_hour": extra_construction,
        "extra_storage_bytes_per_audio_hour": extra_storage,
        "query_latency_saving_s": query_saving,
        "candidate_is_faster": query_saving > 0,
        "construction_break_even_queries_per_audio_hour": break_even,
    }
