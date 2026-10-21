from __future__ import annotations

from pathlib import Path
from typing import Any


def artifact_paths(output_root: str | Path, sample_id: str) -> tuple[Path, Path]:
    """Return deterministic token and decoded-audio paths for one manifest row."""
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError(f"unsafe sample_id: {sample_id!r}")
    root = Path(output_root)
    return root / "tokens" / f"{sample_id}.sncltok", root / "decoded" / f"{sample_id}.wav"


def encoding_record(
    sample: dict[str, Any],
    *,
    representation: str,
    operating_point: str,
    token_path: str | Path,
    decoded_path: str | Path,
    encode_s: float,
    status: str,
    error_type: str = "",
    stderr: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common, auditable encoding record used by learned codecs."""
    token = Path(token_path)
    duration_s = float(sample["duration_s"])
    artifact_bytes = token.stat().st_size if status == "ok" and token.exists() else 0
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "source_sha256": sample["sha256"],
        "representation": representation,
        "operating_point": operating_point,
        "artifact_path": str(token),
        "decoded_path": str(decoded_path),
        "artifact_bytes": artifact_bytes,
        "duration_s": duration_s,
        "effective_kbps": artifact_bytes * 8.0 / duration_s / 1000.0 if duration_s > 0 else 0.0,
        "encode_s": encode_s,
        "status": status,
        "error_type": error_type,
        "stderr": stderr[-2000:],
    }
    if extra:
        row.update(extra)
    return row
