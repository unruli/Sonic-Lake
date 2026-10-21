from pathlib import Path

import pytest

from soniclake.codec_artifacts import artifact_paths, encoding_record


def test_artifact_paths_are_isolated(tmp_path: Path) -> None:
    token, decoded = artifact_paths(tmp_path, "abc123")
    assert token == tmp_path / "tokens" / "abc123.sncltok"
    assert decoded == tmp_path / "decoded" / "abc123.wav"


def test_artifact_paths_reject_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        artifact_paths(tmp_path, "../escape")


def test_failed_record_does_not_count_stale_artifact(tmp_path: Path) -> None:
    token = tmp_path / "stale.sncltok"
    token.write_bytes(b"stale")
    row = encoding_record(
        {"sample_id": "x", "sha256": "abc", "duration_s": 2.0},
        representation="dac",
        operating_point="4cb",
        token_path=token,
        decoded_path=tmp_path / "x.wav",
        encode_s=0.2,
        status="failed",
    )
    assert row["artifact_bytes"] == 0
    assert row["effective_kbps"] == 0.0
