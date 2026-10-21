from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


QUERY_FAMILIES = ("lexical", "speaker", "accent", "emotion", "acoustic_event")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    split: str
    audio_path: str
    duration_s: float
    sha256: str
    labels: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id or not self.dataset or not self.split:
            raise ValueError("sample_id, dataset, and split are required")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive for {self.sample_id}")
        if len(self.sha256) != 64:
            raise ValueError(f"sha256 must contain 64 hexadecimal characters for {self.sample_id}")
        int(self.sha256, 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    sample_id: str
    dataset: str
    representation: str
    operating_point: str
    query_family: str
    source_sha256: str
    artifact_path: str
    artifact_bytes: int
    duration_s: float
    encode_s: float
    decode_s: float
    query_s: float
    quality: float | None
    status: str = "ok"
    error_type: str = ""
    adapter_version: str = ""
    model_revision: str = ""
    environment: str = ""
    seed: int | None = None

    def validate(self) -> None:
        if self.query_family not in QUERY_FAMILIES:
            raise ValueError(f"unknown query family: {self.query_family}")
        if self.status not in {"ok", "failed", "skipped"}:
            raise ValueError(f"invalid status: {self.status}")
        if min(self.artifact_bytes, self.duration_s, self.encode_s, self.decode_s, self.query_s) < 0:
            raise ValueError("cost measurements cannot be negative")
        if self.status == "ok" and self.quality is None:
            raise ValueError("successful results require a quality value")
        if self.quality is not None and not 0 <= self.quality <= 1:
            raise ValueError("quality must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyProfile:
    quality: float
    query_cost_s: float


@dataclass(frozen=True)
class RepresentationProfile:
    name: str
    storage_bytes_per_hour: float
    construction_s_per_hour: float
    families: dict[str, FamilyProfile]

    def validate(self) -> None:
        if self.storage_bytes_per_hour < 0 or self.construction_s_per_hour < 0:
            raise ValueError(f"negative physical cost for {self.name}")
        unknown = set(self.families) - set(QUERY_FAMILIES)
        if unknown:
            raise ValueError(f"unknown query families for {self.name}: {sorted(unknown)}")


def stable_sample_id(dataset: str, split: str, relative_path: str) -> str:
    import hashlib

    logical_key = f"{dataset}\0{split}\0{Path(relative_path).as_posix()}"
    return hashlib.sha256(logical_key.encode("utf-8")).hexdigest()[:20]

