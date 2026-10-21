import json
from pathlib import Path

import pytest

from soniclake.evaluation import evaluation_items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluation_items_join_labels_and_decoded_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = tmp_path / "records.jsonl"
    write_jsonl(manifest, [{"sample_id": "a", "audio_path": "source.wav", "labels": {"x": 1}}])
    write_jsonl(records, [{
        "sample_id": "a", "status": "ok", "decoded_path": "decoded.wav",
        "artifact_path": "tokens.bin", "representation": "dac", "operating_point": "4cb",
    }])
    item = evaluation_items(manifest, records)[0]
    assert item["audio_path"] == "decoded.wav"
    assert item["sample"]["labels"] == {"x": 1}


def test_evaluation_items_reject_incomplete_representation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = tmp_path / "records.jsonl"
    write_jsonl(manifest, [{"sample_id": "a", "audio_path": "source.wav"}])
    write_jsonl(records, [])
    with pytest.raises(ValueError, match="mismatch"):
        evaluation_items(manifest, records)
