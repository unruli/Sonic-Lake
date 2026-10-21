from soniclake.heldout import (
    calibration_ids,
    holdout_ids,
    layout_stability,
    paired_quality_margin,
    quality_from_rows,
    realize_design,
)
from soniclake.optimizer import Design
from soniclake.schema import FamilyProfile, RepresentationProfile


def test_calibration_splits_are_nested_and_disjoint_from_holdout() -> None:
    ids = {f"sample-{index}" for index in range(10_000)}
    small = calibration_ids(ids, 0.1, seed=7)
    large = calibration_ids(ids, 0.2, seed=7)
    test = holdout_ids(ids, seed=7)
    assert small < large
    assert large.isdisjoint(test)
    assert 850 <= len(small) <= 1150
    assert 1850 <= len(large) <= 2150
    assert 4800 <= len(test) <= 5200


def test_quality_metrics_match_family_conventions() -> None:
    lexical = [
        {"status": "ok", "word_errors": 1, "reference_words": 4},
        {"status": "ok", "word_errors": 0, "reference_words": 6},
    ]
    events = [{"status": "ok", "correct": value} for value in (1, 0, 1, 1)]
    emotion = [
        {"status": "ok", "gold": "a", "prediction": "a"},
        {"status": "ok", "gold": "a", "prediction": "b"},
        {"status": "ok", "gold": "b", "prediction": "b"},
        {"status": "ok", "gold": "b", "prediction": "b"},
    ]
    assert quality_from_rows("lexical", lexical) == 0.9
    assert quality_from_rows("acoustic_event", events) == 0.75
    assert quality_from_rows("emotion", emotion) > 0.73


def test_realize_design_uses_fixed_assignment_and_detects_violation() -> None:
    profiles = {
        "a": RepresentationProfile(
            name="a",
            storage_bytes_per_hour=10,
            construction_s_per_hour=2,
            families={
                "lexical": FamilyProfile(quality=0.89, query_cost_s=1),
                "emotion": FamilyProfile(quality=0.95, query_cost_s=4),
            },
        ),
        "b": RepresentationProfile(
            name="b",
            storage_bytes_per_hour=20,
            construction_s_per_hour=3,
            families={
                "lexical": FamilyProfile(quality=0.99, query_cost_s=3),
                "emotion": FamilyProfile(quality=0.91, query_cost_s=2),
            },
        ),
    }
    design = Design(
        representations=("a", "b"),
        assignment={"lexical": "a", "emotion": "b"},
        storage_bytes_per_hour=30,
        construction_s_per_hour=5,
        expected_query_s=0,
        objective=0,
    )
    realized = realize_design(
        design,
        profiles,
        frequencies={"lexical": 0.75, "emotion": 0.25},
        thresholds={"lexical": 0.9, "emotion": 0.9},
    )
    assert realized.expected_query_s == 1.25
    assert realized.violated_families == ("lexical",)
    assert realized.storage_bytes_per_hour == 30


def test_layout_stability_reports_set_and_assignment_agreement() -> None:
    calibration = Design(("a", "b"), {"x": "a", "y": "b"}, 0, 0, 0, 0)
    oracle = Design(("b", "c"), {"x": "c", "y": "b"}, 0, 0, 0, 0)
    result = layout_stability(calibration, oracle)
    assert result["representation_jaccard"] == 1 / 3
    assert result["assignment_matches"] == 1
    assert result["assignment_total"] == 2


def test_paired_quality_margin_rewards_consistent_retention() -> None:
    source = [
        {"sample_id": str(index), "status": "ok", "correct": 1}
        for index in range(20)
    ]
    candidate = [
        {"sample_id": str(index), "status": "ok", "correct": int(index != 0)}
        for index in range(20)
    ]
    estimate = paired_quality_margin(
        "acoustic_event", source, candidate, retention=0.8, confidence=0.95
    )
    assert estimate.samples == 20
    assert estimate.point_margin > 0
    assert estimate.lower_bound is not None
    assert estimate.lower_bound > 0


def test_paired_quality_margin_does_not_certify_one_sample() -> None:
    source = [{"sample_id": "only", "status": "ok", "correct": 1}]
    estimate = paired_quality_margin(
        "acoustic_event", source, source, retention=0.9, confidence=0.95
    )
    assert estimate.samples == 1
    assert estimate.standard_error is None
    assert estimate.lower_bound is None
