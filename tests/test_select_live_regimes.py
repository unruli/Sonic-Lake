import pytest

from scripts.select_live_regimes import select


def row(policy: str, gain: float, index: int) -> dict:
    latency = 1.0 if policy == "single" else 1.0 - gain
    assignment = {"lexical": "opus@24kbps", "acoustic_event": "flac@lossless_c8", "emotion": "flac@lossless_c8"}
    return {
        "status": "ok", "policy": policy, "workload": f"w{index}",
        "quality_retention": 0.9, "budget_pcm_multiplier": 1.0,
        "expected_query_s": latency, "representations": sorted(set(assignment.values())),
        "assignment": assignment,
    }


def test_select_spans_gain_range() -> None:
    rows = []
    for index, gain in enumerate((0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
        rows.extend((row("single", gain, index), row("adaptive", gain, index)))
    payload = select(rows, 3)
    gains = [item["predicted_latency_reduction"] for item in payload["regimes"]]
    assert gains == pytest.approx([0.0, 0.2, 0.5])
