#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from soniclake.evaluation import evaluation_items
from soniclake.io import read_jsonl, write_jsonl
from soniclake.systems import percentile


REPRESENTATIONS = {
    "pcm_s16le@source_sr": "pcm",
    "flac@lossless_c8": "flac",
    "opus@24kbps": "opus24",
}


def select_layout(regimes: dict, regime_id: str, policy: str) -> tuple[dict, dict]:
    regime = next((item for item in regimes["regimes"] if item["id"] == regime_id), None)
    if regime is None:
        raise ValueError(f"unknown regime {regime_id!r}")
    if policy not in {"single", "adaptive"}:
        raise ValueError("policy must be single or adaptive")
    return regime, regime[policy]


def build_event_model():
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
    return CLAP(version="2023", use_cuda=torch.cuda.is_available()), torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute frozen SonicLake adaptive and single layouts live.")
    parser.add_argument("--regimes", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--policy", choices=("single", "adaptive"), required=True)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--librispeech-manifest", type=Path, default=Path("data/librispeech.jsonl"))
    parser.add_argument("--cremad-manifest", type=Path, default=Path("data/cremad_subset.jsonl"))
    parser.add_argument("--esc50-manifest", type=Path, default=Path("data/esc50.jsonl"))
    args = parser.parse_args()
    if args.queries < 1:
        raise ValueError("queries must be positive")

    regimes = json.loads(args.regimes.read_text(encoding="utf-8"))
    regime, layout = select_layout(regimes, args.regime, args.policy)
    assignment = layout["assignment"]
    workload = json.loads(args.workloads.read_text(encoding="utf-8"))["workloads"][regime["workload"]]
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    storage_bytes_per_hour = sum(profiles[name]["storage_bytes_per_hour"] for name in layout["representations"])
    rng = random.Random(args.seed)
    families = sorted(workload)
    trace = rng.choices(families, weights=[workload[name] for name in families], k=args.queries)

    setup_started = time.perf_counter()
    suffixes = {"pcm": "pcm_s16le", "flac": "flac_lossless_c8", "opus24": "opus_24kbps"}
    asr_names = {"pcm": "asr_pcm.jsonl", "flac": "asr_flac.jsonl", "opus24": "asr_opus24.jsonl"}

    from scripts.evaluate_keyword_retrieval import inverted_index
    indexes = {}
    for rep in set(REPRESENTATIONS.values()):
        records = [row for row in read_jsonl(Path("results/production/evaluation") / asr_names[rep])
                   if row.get("status") == "ok"]
        indexes[rep] = inverted_index(records, "normalized_hypothesis")
    vocabulary = sorted(set.intersection(*(set(index) for index in indexes.values())))
    lexical_terms = iter(rng.choices(vocabulary, k=trace.count("lexical")))

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scripts.evaluate_emotion_prosody import extract_features
    emotion_source = evaluation_items(args.cremad_manifest)
    source_x = np.nan_to_num(np.asarray([extract_features(item["audio_path"]) for item in emotion_source]))
    labels = np.asarray([item["sample"]["labels"]["emotion"] for item in emotion_source])
    emotion_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=0))
    emotion_model.fit(source_x, labels)
    emotion_targets = {}
    for rep, suffix in suffixes.items():
        records = Path(f"results/production/encoding/crema_d_{suffix}.jsonl")
        emotion_targets[rep] = {item["sample_id"]: item for item in evaluation_items(args.cremad_manifest, records)}
    emotion_ids = iter(rng.choices(sorted(set.intersection(*(set(items) for items in emotion_targets.values()))),
                                   k=trace.count("emotion")))

    event_model, torch = build_event_model()
    event_targets = {}
    for rep, suffix in suffixes.items():
        records = Path(f"results/production/encoding/esc50_{suffix}.jsonl")
        event_targets[rep] = {item["sample_id"]: item for item in evaluation_items(args.esc50_manifest, records)}
    event_ids = iter(rng.choices(sorted(set.intersection(*(set(items) for items in event_targets.values()))),
                                 k=trace.count("acoustic_event")))
    for rep in sorted(set(REPRESENTATIONS[name] for name in assignment.values())):
        sample_id = next(iter(event_targets[rep]))
        event_model.get_audio_embeddings([event_targets[rep][sample_id]["audio_path"]])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_started

    rows = []
    run_started = time.perf_counter()
    for query_id, family in enumerate(trace):
        route_started = time.perf_counter_ns()
        representation = assignment[family]
        rep = REPRESENTATIONS[representation]
        routed_ns = time.perf_counter_ns()
        if family == "lexical":
            term = next(lexical_terms)
            result_count = len(indexes[rep].get(term, set()))
            result = {"term": term, "result_count": result_count}
        elif family == "emotion":
            sample_id = next(emotion_ids)
            features = np.nan_to_num(np.asarray([extract_features(emotion_targets[rep][sample_id]["audio_path"])]))
            result = {"sample_id": sample_id, "prediction": str(emotion_model.predict(features)[0])}
        else:
            sample_id = next(event_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            event_model.get_audio_embeddings([event_targets[rep][sample_id]["audio_path"]])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result = {"sample_id": sample_id}
        finished_ns = time.perf_counter_ns()
        rows.append({
            "query_id": query_id, "family": family, "representation": representation,
            "router_s": (routed_ns - route_started) / 1e9,
            "response_s": (finished_ns - route_started) / 1e9,
            **result,
        })
    wall_s = time.perf_counter() - run_started
    response = [row["response_s"] for row in rows]
    router = [row["router_s"] for row in rows]
    summary = {
        "method": "live serial mixed-workload execution",
        "regime": args.regime,
        "policy": args.policy,
        "workload": regime["workload"],
        "quality_retention": regime["quality_retention"],
        "budget_pcm_multiplier": regime["budget_pcm_multiplier"],
        "predicted_latency_reduction": regime["predicted_latency_reduction"],
        "assignment": assignment,
        "representations": layout["representations"],
        "storage_bytes_per_hour": storage_bytes_per_hour,
        "queries": len(rows),
        "seed": args.seed,
        "setup_s": setup_s,
        "wall_s": wall_s,
        "throughput_queries_per_s": len(rows) / wall_s,
        "response_median_ms": statistics.median(response) * 1000,
        "response_p95_ms": percentile(response, 0.95) * 1000,
        "router_median_us": statistics.median(router) * 1e6,
        "family_counts": {family: trace.count(family) for family in families},
    }
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
