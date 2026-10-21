from soniclake.router import CatalogRouter


def test_router_dispatches_and_cycles_measurements() -> None:
    rows = {
        ("opus@24kbps", "lexical"): [
            {"sample_id": "a", "decode_s": 0.01, "query_s": 9.0},
            {"sample_id": "b", "decode_s": 0.02, "query_s": 9.0},
        ],
        ("flac@lossless_c8", "emotion"): [
            {"sample_id": "c", "decode_s": 0.03, "query_s": 0.2},
        ],
    }
    router = CatalogRouter(
        {"lexical": "opus@24kbps", "emotion": "flac@lossless_c8"},
        rows,
        {"opus@24kbps": [1e-7, 2e-7]},
    )
    first = router.route("lexical")
    second = router.route("lexical")
    emotion = router.route("emotion")
    assert first.sample_id == "a"
    assert second.sample_id == "b"
    assert first.measured_execution_s == 0.0100001
    assert second.measured_execution_s == 0.0200002
    assert emotion.representation == "flac@lossless_c8"
    assert emotion.measured_execution_s == 0.23
    assert first.router_ns >= 0


def test_router_rejects_missing_catalog_measurements() -> None:
    try:
        CatalogRouter({"emotion": "missing@point"}, {})
    except ValueError as error:
        assert "missing measurements" in str(error)
    else:
        raise AssertionError("expected missing measurements to fail")
