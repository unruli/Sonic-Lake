from __future__ import annotations

import time
from pathlib import Path

import pytest

from soniclake import systems


def test_percentile_interpolates() -> None:
    assert systems.percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)


def test_run_closed_loop_reports_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(b"audio" * 20)
        paths.append(path)
    items = [{"sample_id": str(index), "artifact_path": str(path)} for index, path in enumerate(paths)]

    def fake_decode(payload: bytes, codec: str, ffmpeg: str = "ffmpeg") -> None:
        assert payload
        assert codec == "pcm"
        time.sleep(0.001)

    monkeypatch.setattr(systems, "decode_bytes", fake_decode)
    measurements, summary = systems.run_closed_loop(
        items, codec="pcm", concurrency=2, cache_mode="warm"
    )
    assert len(measurements) == 4
    assert summary["requests"] == 4
    assert summary["throughput_queries_per_s"] > 0
    assert summary["response_p95_ms"] >= summary["decode_median_ms"]
    assert all(row.artifact_bytes == 100 for row in measurements)


def test_run_closed_loop_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        systems.run_closed_loop([], codec="pcm", concurrency=1, cache_mode="warm")
    with pytest.raises(ValueError):
        systems.run_closed_loop([{"sample_id": "x", "artifact_path": "x"}], codec="pcm", concurrency=0, cache_mode="warm")
