import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_production_profiles.py"
SPEC = importlib.util.spec_from_file_location("build_production_profiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quality_conventions() -> None:
    assert MODULE.quality_from_summary("lexical", {"corpus_wer": 0.2}) == 0.8
    assert MODULE.quality_from_summary("acoustic_event", {"macro_accuracy": 0.7}) == 0.7
    assert MODULE.quality_from_summary("emotion", {"macro_f1": 0.6}) == 0.6


def test_emotion_query_includes_feature_extraction() -> None:
    row = {"inference_s": 0.1, "feature_extraction_s": 0.4}
    assert MODULE.query_seconds("emotion", row) == 0.5
    assert MODULE.query_seconds("lexical", row) == 0.1
