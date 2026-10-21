#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from soniclake.io import read_jsonl, write_jsonl


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "not", "of", "on", "or", "she", "that", "the", "their", "they",
    "this", "to", "was", "were", "will", "with", "you",
}


def terms(text: str) -> set[str]:
    return {
        token.strip("'")
        for token in text.lower().split()
        if len(token.strip("'")) >= 3 and token.strip("'") not in STOPWORDS
    }


def inverted_index(rows: list[dict], field: str) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for term in terms(row[field]):
            index[term].add(row["sample_id"])
    return dict(index)


def select_queries(
    gold: dict[str, set[str]],
    *,
    documents: int,
    min_document_frequency: int,
    max_document_frequency_fraction: float,
    max_queries: int,
    seed: int,
) -> list[str]:
    maximum = max(min_document_frequency, int(documents * max_document_frequency_fraction))
    eligible = [
        term for term, postings in gold.items()
        if min_document_frequency <= len(postings) <= maximum
    ]
    ranked = sorted(
        eligible,
        key=lambda term: hashlib.sha256(f"{seed}\0{term}".encode("utf-8")).digest(),
    )
    return ranked[:max_queries]


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def serialized_index_bytes(index: dict[str, set[str]]) -> int:
    payload = {term: sorted(postings) for term, postings in sorted(index.items())}
    return len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def evaluate(
    baseline: list[dict],
    target: list[dict],
    *,
    min_document_frequency: int,
    max_document_frequency_fraction: float,
    max_queries: int,
    seed: int,
    latency_repetitions: int,
) -> tuple[list[dict], dict]:
    base = {row["sample_id"]: row for row in baseline if row["status"] == "ok"}
    candidate = {row["sample_id"]: row for row in target if row["status"] == "ok"}
    if set(base) != set(candidate):
        raise ValueError(f"sample mismatch: baseline={len(base)}, target={len(candidate)}")
    ordered_ids = sorted(base)
    base_rows = [base[sample_id] for sample_id in ordered_ids]
    target_rows = [candidate[sample_id] for sample_id in ordered_ids]

    gold = inverted_index(base_rows, "normalized_reference")
    build_started = time.perf_counter()
    predicted = inverted_index(target_rows, "normalized_hypothesis")
    build_s = time.perf_counter() - build_started
    queries = select_queries(
        gold,
        documents=len(base_rows),
        min_document_frequency=min_document_frequency,
        max_document_frequency_fraction=max_document_frequency_fraction,
        max_queries=max_queries,
        seed=seed,
    )
    if not queries:
        raise ValueError("no terms satisfy the query-frequency constraints")

    rows = []
    total_true = total_retrieved = total_tp = 0
    for term in queries:
        relevant = gold[term]
        retrieved = predicted.get(term, set())
        started = time.perf_counter_ns()
        for _ in range(latency_repetitions):
            predicted.get(term, set())
        lookup_ns = (time.perf_counter_ns() - started) / latency_repetitions
        true_positive = len(relevant & retrieved)
        precision = safe_ratio(true_positive, len(retrieved))
        recall = safe_ratio(true_positive, len(relevant))
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        rows.append({
            "query": term,
            "relevant_documents": len(relevant),
            "retrieved_documents": len(retrieved),
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "lookup_ns": lookup_ns,
        })
        total_true += len(relevant)
        total_retrieved += len(retrieved)
        total_tp += true_positive

    micro_precision = safe_ratio(total_tp, total_retrieved)
    micro_recall = safe_ratio(total_tp, total_true)
    representation = target_rows[0]["representation"]
    operating_point = target_rows[0]["operating_point"]
    audio_hours = sum(float(row["duration_s"]) for row in target_rows) / 3600
    transcription_s = sum(float(row["inference_s"]) for row in target_rows)
    summary = {
        "representation": representation,
        "operating_point": operating_point,
        "documents": len(base_rows),
        "queries": len(rows),
        "min_document_frequency": min_document_frequency,
        "max_document_frequency_fraction": max_document_frequency_fraction,
        "seed": seed,
        "macro_precision": sum(row["precision"] for row in rows) / len(rows),
        "macro_recall": sum(row["recall"] for row in rows) / len(rows),
        "macro_f1": sum(row["f1"] for row in rows) / len(rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "index_terms": len(predicted),
        "index_postings": sum(len(postings) for postings in predicted.values()),
        "index_bytes": serialized_index_bytes(predicted),
        "index_bytes_per_audio_hour": serialized_index_bytes(predicted) / audio_hours,
        "index_build_s": build_s,
        "audio_hours": audio_hours,
        "transcription_s_per_audio_hour": transcription_s / audio_hours,
        "median_lookup_ns": sorted(row["lookup_ns"] for row in rows)[len(rows) // 2],
        "latency_repetitions": latency_repetitions,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic keyword retrieval over ASR transcripts.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-document-frequency", type=int, default=5)
    parser.add_argument("--max-document-frequency-fraction", type=float, default=0.20)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--latency-repetitions", type=int, default=1000)
    args = parser.parse_args()

    if not 0 < args.max_document_frequency_fraction <= 1:
        raise ValueError("max-document-frequency-fraction must be in (0, 1]")
    if min(args.min_document_frequency, args.max_queries, args.latency_repetitions) < 1:
        raise ValueError("frequency, query, and repetition arguments must be positive")

    rows, summary = evaluate(
        list(read_jsonl(args.baseline)),
        list(read_jsonl(args.target)),
        min_document_frequency=args.min_document_frequency,
        max_document_frequency_fraction=args.max_document_frequency_fraction,
        max_queries=args.max_queries,
        seed=args.seed,
        latency_repetitions=args.latency_repetitions,
    )
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
