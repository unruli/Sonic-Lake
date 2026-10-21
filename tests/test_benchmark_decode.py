import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_decode.py"
SPEC = importlib.util.spec_from_file_location("benchmark_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_evenly_spaced_includes_endpoints() -> None:
    rows = [{"i": i} for i in range(10)]
    assert MODULE.evenly_spaced(rows, 4) == [rows[0], rows[3], rows[6], rows[9]]


def test_evenly_spaced_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        MODULE.evenly_spaced([{"i": 1}], 0)


def test_percentile_interpolates() -> None:
    assert MODULE.percentile([1.0, 2.0, 3.0], 0.75) == 2.5
