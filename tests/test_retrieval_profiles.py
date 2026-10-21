import json

from scripts.build_retrieval_profiles import build


def test_retrieval_profile_adds_index_and_replaces_lexical_cost(tmp_path) -> None:
    profiles = {
        "opus@48kbps": {
            "storage_bytes_per_hour": 10.0,
            "construction_s_per_hour": 2.0,
            "families": {
                "lexical": {"quality": 0.8, "query_cost_s": 1.0},
                "emotion": {"quality": 0.5, "query_cost_s": 2.0},
            },
        }
    }
    retrieval = {
        "index_bytes_per_audio_hour": 3.0,
        "transcription_s_per_audio_hour": 4.0,
        "index_build_s": 2.0,
        "audio_hours": 2.0,
        "macro_f1": 0.9,
        "median_lookup_ns": 1000.0,
        "queries": 50,
    }
    (tmp_path / "opus48.summary.json").write_text(json.dumps(retrieval), encoding="utf-8")
    result = build(profiles, tmp_path)["opus@48kbps"]
    assert result["storage_bytes_per_hour"] == 13.0
    assert result["construction_s_per_hour"] == 7.0
    assert result["families"]["lexical"]["quality"] == 0.9
    assert result["families"]["lexical"]["query_cost_s"] == 1e-6
