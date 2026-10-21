import pytest

from scripts.benchmark_adaptive_layout import select_layout


def test_select_layout_returns_frozen_policy():
    payload = {"regimes": [{"id": "r1", "single": {"assignment": {"lexical": "flac"}},
                             "adaptive": {"assignment": {"lexical": "opus"}}}]}
    regime, layout = select_layout(payload, "r1", "adaptive")
    assert regime["id"] == "r1"
    assert layout["assignment"]["lexical"] == "opus"


def test_select_layout_rejects_unknown_values():
    payload = {"regimes": [{"id": "r1", "single": {}, "adaptive": {}}]}
    with pytest.raises(ValueError):
        select_layout(payload, "missing", "single")
    with pytest.raises(ValueError):
        select_layout(payload, "r1", "oracle")
