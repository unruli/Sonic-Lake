from scripts.evaluate_keyword_retrieval import evaluate, inverted_index, select_queries


def row(sample_id: str, reference: str, hypothesis: str) -> dict:
    return {
        "sample_id": sample_id,
        "status": "ok",
        "normalized_reference": reference,
        "normalized_hypothesis": hypothesis,
        "representation": "opus",
        "operating_point": "48kbps",
        "duration_s": 10.0,
        "inference_s": 0.1,
    }


def test_keyword_retrieval_reports_false_negative() -> None:
    baseline = [
        row("a", "bright morning river", "bright morning river"),
        row("b", "bright evening river", "evening river"),
    ]
    rows, summary = evaluate(
        baseline,
        baseline,
        min_document_frequency=2,
        max_document_frequency_fraction=1.0,
        max_queries=20,
        seed=1,
        latency_repetitions=2,
    )
    bright = next(item for item in rows if item["query"] == "bright")
    assert bright["recall"] == 0.5
    assert summary["micro_recall"] < 1.0
    assert summary["index_bytes"] > 0
    assert summary["transcription_s_per_audio_hour"] > 0


def test_query_selection_is_deterministic() -> None:
    rows = [row(str(i), "alpha beta gamma", "alpha beta gamma") for i in range(5)]
    index = inverted_index(rows, "normalized_reference")
    first = select_queries(
        index,
        documents=5,
        min_document_frequency=1,
        max_document_frequency_fraction=1.0,
        max_queries=2,
        seed=7,
    )
    second = select_queries(
        index,
        documents=5,
        min_document_frequency=1,
        max_document_frequency_fraction=1.0,
        max_queries=2,
        seed=7,
    )
    assert first == second
