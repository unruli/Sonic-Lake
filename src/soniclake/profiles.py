from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable

from .schema import FamilyProfile, RepresentationProfile, RunResult


def aggregate_results(rows: Iterable[dict[str, Any]]) -> dict[str, RepresentationProfile]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for raw in rows:
        result = RunResult(**raw)
        result.validate()
        if result.status == "ok":
            key = f"{result.representation}@{result.operating_point}"
            grouped[key].append(result)

    profiles: dict[str, RepresentationProfile] = {}
    for name, results in grouped.items():
        total_hours = sum(row.duration_s for row in results) / 3600
        if total_hours <= 0:
            continue
        # Artifact storage and construction cost are repeated across query-family rows.
        unique_artifacts: dict[tuple[str, str], RunResult] = {}
        for row in results:
            unique_artifacts[(row.sample_id, row.artifact_path)] = row
        artifact_rows = list(unique_artifacts.values())
        artifact_hours = sum(row.duration_s for row in artifact_rows) / 3600
        families: dict[str, FamilyProfile] = {}
        for family in sorted({row.query_family for row in results}):
            selected = [row for row in results if row.query_family == family]
            families[family] = FamilyProfile(
                quality=statistics.fmean(row.quality for row in selected if row.quality is not None),
                query_cost_s=statistics.fmean(row.decode_s + row.query_s for row in selected),
            )
        profile = RepresentationProfile(
            name=name,
            storage_bytes_per_hour=sum(row.artifact_bytes for row in artifact_rows) / artifact_hours,
            construction_s_per_hour=sum(row.encode_s for row in artifact_rows) / artifact_hours,
            families=families,
        )
        profile.validate()
        profiles[name] = profile
    return profiles


def profiles_to_dict(profiles: dict[str, RepresentationProfile]) -> dict[str, Any]:
    return {
        name: {
            "storage_bytes_per_hour": profile.storage_bytes_per_hour,
            "construction_s_per_hour": profile.construction_s_per_hour,
            "families": {
                family: {"quality": item.quality, "query_cost_s": item.query_cost_s}
                for family, item in profile.families.items()
            },
        }
        for name, profile in sorted(profiles.items())
    }


def profiles_from_dict(payload: dict[str, Any]) -> dict[str, RepresentationProfile]:
    profiles: dict[str, RepresentationProfile] = {}
    for name, raw in payload.items():
        profiles[name] = RepresentationProfile(
            name=name,
            storage_bytes_per_hour=float(raw["storage_bytes_per_hour"]),
            construction_s_per_hour=float(raw["construction_s_per_hour"]),
            families={
                family: FamilyProfile(
                    quality=float(values["quality"]),
                    query_cost_s=float(values["query_cost_s"]),
                )
                for family, values in raw["families"].items()
            },
        )
        profiles[name].validate()
    return profiles

