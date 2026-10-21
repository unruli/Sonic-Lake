import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "bootstrap_quality.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_macro_f1() -> None:
    gold = np.asarray(["a", "a", "b", "b"])
    predicted = np.asarray(["a", "b", "b", "b"])
    assert np.isclose(MODULE.macro_f1(gold, predicted), (2 / 3 + 4 / 5) / 2)


def test_paired_bootstrap_identity_has_zero_delta() -> None:
    rows = [
        {"sample_id": "a", "status": "ok", "correct": True},
        {"sample_id": "b", "status": "ok", "correct": False},
    ]
    _, deltas = MODULE.paired_bootstrap(
        "acoustic_event", rows, rows, replicates=100, seed=7, batch_size=10
    )
    assert np.all(deltas == 0)
