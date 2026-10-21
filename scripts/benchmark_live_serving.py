#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from soniclake.evaluation import evaluation_items
from soniclake.io import read_jsonl, write_jsonl
from soniclake.systems import percentile


def run_threaded(items: list[dict], concurrency: int, execute: Callable[[dict], dict]) -> tuple[list[dict], float]:
    """Run a closed-loop workload with at most ``concurrency`` requests in flight."""
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        remaining = iter(items)
        pending = {}

        def submit(item: dict) -> None:
            submitted = time.perf_counter_ns()
            pending[executor.submit(execute, item)] = submitted

        for _ in range(min(concurrency, len(items))):
            submit(next(remaining))
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                submitted = pending.pop(future)
                row = future.result()
                row["submitted_ns"] = submitted
                row["response_s"] = (row["finished_ns"] - submitted) / 1e9
                rows.append(row)
                try:
                    submit(next(remaining))
                except StopIteration:
                    pass
    return rows, time.perf_counter() - started


def summarize(rows: list[dict], wall_s: float, family: str, concurrency: int) -> dict:
    response = [row["response_s"] for row in rows]
    service = [row["service_s"] for row in rows]
    return {
        "method": "live query execution",
        "family": family,
        "concurrency": concurrency,
        "requests": len(rows),
        "wall_s": wall_s,
        "throughput_queries_per_s": len(rows) / wall_s,
        "service_median_ms": statistics.median(service) * 1000,
        "service_p95_ms": percentile(service, 0.95) * 1000,
        "response_median_ms": statistics.median(response) * 1000,
        "response_p95_ms": percentile(response, 0.95) * 1000,
    }


def lexical_runner(target: Path, requests: int, seed: int) -> tuple[list[dict], Callable[[dict], dict]]:
    from scripts.evaluate_keyword_retrieval import inverted_index, terms

    records = [row for row in read_jsonl(target) if row.get("status") == "ok"]
    index = inverted_index(records, "normalized_hypothesis")
    vocabulary = sorted(index)
    selected = random.Random(seed).choices(vocabulary, k=requests)
    items = [{"query_id": index, "term": term} for index, term in enumerate(selected)]

    def execute(item: dict) -> dict:
        started = time.perf_counter_ns()
        postings = index.get(item["term"], set())
        finished = time.perf_counter_ns()
        return {
            "query_id": item["query_id"], "result_count": len(postings),
            "service_s": (finished - started) / 1e9, "finished_ns": finished,
        }

    return items, execute


def emotion_runner(manifest: Path, records: Path, requests: int, seed: int) -> tuple[list[dict], Callable[[dict], dict], float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scripts.evaluate_emotion_prosody import extract_features

    setup_started = time.perf_counter()
    source_items = evaluation_items(manifest)
    target_by_id = {item["sample_id"]: item for item in evaluation_items(manifest, records)}
    source_x = np.nan_to_num(np.asarray([extract_features(item["audio_path"]) for item in source_items]))
    labels = np.asarray([item["sample"]["labels"]["emotion"] for item in source_items])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=0))
    classifier.fit(source_x, labels)
    chosen = random.Random(seed).choices(source_items, k=requests)
    items = [target_by_id[item["sample_id"]] for item in chosen]

    def execute(item: dict) -> dict:
        started = time.perf_counter_ns()
        features = np.nan_to_num(np.asarray([extract_features(item["audio_path"])]))
        prediction = str(classifier.predict(features)[0])
        finished = time.perf_counter_ns()
        return {
            "sample_id": item["sample_id"], "prediction": prediction,
            "service_s": (finished - started) / 1e9, "finished_ns": finished,
        }

    return items, execute, time.perf_counter() - setup_started


def event_batched(manifest: Path, records: Path, requests: int, batch_size: int, seed: int) -> tuple[list[dict], float, float]:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.transforms as transforms
    from msclap import CLAP
    from msclap.CLAPWrapper import CLAPWrapper

    def read_audio_soundfile(self, audio_path, resample=True):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
        target_rate = self.args.sampling_rate
        if resample and target_rate != sample_rate:
            tensor = transforms.Resample(sample_rate, target_rate)(tensor)
        return tensor, target_rate

    CLAPWrapper.read_audio = read_audio_soundfile
    available = evaluation_items(manifest, records)
    chosen = random.Random(seed).choices(available, k=requests)
    load_started = time.perf_counter()
    model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_started
    # Warm up kernels and allocations outside the measured interval.
    model.get_audio_embeddings([item["audio_path"] for item in chosen[:batch_size]])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rows = []
    run_started = time.perf_counter()
    for offset in range(0, len(chosen), batch_size):
        batch = chosen[offset:offset + batch_size]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        submitted = time.perf_counter_ns()
        model.get_audio_embeddings([item["audio_path"] for item in batch])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        service_s = (finished - submitted) / 1e9
        for item in batch:
            rows.append({
                "sample_id": item["sample_id"], "service_s": service_s,
                "response_s": service_s, "submitted_ns": submitted, "finished_ns": finished,
            })
    return rows, time.perf_counter() - run_started, model_load_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live SonicLake query evaluators.")
    parser.add_argument("--family", choices=("lexical", "emotion", "acoustic_event"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--asr-results", type=Path)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        raise ValueError("concurrency and requests must be positive")

    model_load_s = 0.0
    if args.family == "lexical":
        if args.asr_results is None:
            raise ValueError("lexical requires --asr-results")
        items, execute = lexical_runner(args.asr_results, args.requests, args.seed)
        rows, wall_s = run_threaded(items, args.concurrency, execute)
    elif args.family == "emotion":
        if args.manifest is None or args.records is None:
            raise ValueError("emotion requires --manifest and --records")
        items, execute, model_load_s = emotion_runner(args.manifest, args.records, args.requests, args.seed)
        rows, wall_s = run_threaded(items, args.concurrency, execute)
    else:
        if args.manifest is None or args.records is None:
            raise ValueError("acoustic_event requires --manifest and --records")
        rows, wall_s, model_load_s = event_batched(
            args.manifest, args.records, args.requests, args.concurrency, args.seed
        )

    summary = summarize(rows, wall_s, args.family, args.concurrency)
    summary.update(representation=args.representation, seed=args.seed, model_load_s=model_load_s)
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
