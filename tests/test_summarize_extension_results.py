from scripts.summarize_extension_results import summarize_sweep


def test_summarize_sweep_counts_adaptive_win() -> None:
    common = {
        "status": "ok",
        "workload": "balanced",
        "quality_retention": 0.9,
        "budget_pcm_multiplier": 0.5,
        "assignment": {"lexical": "flac@lossless_c8"},
        "representations": ["flac@lossless_c8"],
    }
    rows = [
        {**common, "policy": "single", "expected_query_s": 2.0},
        {**common, "policy": "adaptive", "expected_query_s": 1.0},
    ]
    summary = summarize_sweep(rows)
    assert summary["both_feasible"] == 1
    assert summary["adaptive_wins"] == 1
    assert summary["winning_reduction_median"] == 0.5
    assert summary["materialization_counts"]["adaptive"]["flac@lossless_c8"] == 1
