from __future__ import annotations

from pathlib import Path
from typing import Any

from soniclake.io import read_jsonl


def evaluation_items(
    source_manifest: str | Path,
    encoding_records: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Join immutable source labels to one successfully materialized representation."""
    samples = list(read_jsonl(source_manifest))
    if encoding_records is None:
        return [
            {
                "sample_id": sample["sample_id"],
                "audio_path": sample["audio_path"],
                "representation": "source",
                "operating_point": "source",
                "sample": sample,
            }
            for sample in samples
        ]

    records = list(read_jsonl(encoding_records))
    by_id = {record["sample_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("encoding records contain duplicate sample_id values")

    source_ids = {sample["sample_id"] for sample in samples}
    missing = source_ids - by_id.keys()
    extra = by_id.keys() - source_ids
    if missing or extra:
        raise ValueError(f"manifest/encoding ID mismatch: missing={len(missing)}, extra={len(extra)}")

    items = []
    for sample in samples:
        record = by_id[sample["sample_id"]]
        if record.get("status") != "ok":
            raise ValueError(f"failed encoding record for {sample['sample_id']}")
        audio_path = record.get("decoded_path") or record["artifact_path"]
        items.append(
            {
                "sample_id": sample["sample_id"],
                "audio_path": audio_path,
                "representation": record["representation"],
                "operating_point": record["operating_point"],
                "sample": sample,
            }
        )
    return items
