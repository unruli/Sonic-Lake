from soniclake.baselines import construction_break_even, deterministic_random_feasible_single
from soniclake.heldout import RealizedDesign
from soniclake.schema import FamilyProfile, RepresentationProfile


def test_random_feasible_single_is_deterministic_and_respects_constraints() -> None:
    profiles = {
        name: RepresentationProfile(
            name=name,
            storage_bytes_per_hour=storage,
            construction_s_per_hour=1,
            families={"lexical": FamilyProfile(quality=quality, query_cost_s=cost)},
        )
        for name, storage, quality, cost in (
            ("a", 10, 0.95, 1),
            ("b", 20, 0.99, 2),
            ("bad", 5, 0.5, 0.1),
        )
    }
    first = deterministic_random_feasible_single(
        profiles, {"lexical": 1}, {"lexical": 0.9}, 30, "same-key"
    )
    second = deterministic_random_feasible_single(
        profiles, {"lexical": 1}, {"lexical": 0.9}, 30, "same-key"
    )
    assert first == second
    assert first is not None
    assert first.representations[0] in {"a", "b"}


def test_construction_break_even_reports_query_count() -> None:
    baseline = RealizedDesign(("a",), {}, 10, 5, 3, {}, {}, ())
    candidate = RealizedDesign(("a", "b"), {}, 15, 9, 2, {}, {}, ())
    result = construction_break_even(baseline, candidate)
    assert result["extra_storage_bytes_per_audio_hour"] == 5
    assert result["construction_break_even_queries_per_audio_hour"] == 4
