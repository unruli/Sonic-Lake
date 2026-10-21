#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from soniclake.evaluation import evaluation_items
from soniclake.io import write_jsonl


FEATURE_NAMES = (
    ["f0_mean", "f0_std", "voiced_frac", "rms_mean", "rms_std", "zcr_mean",
     "centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean", "tempo"]
    + [f"mfcc{i}_mean" for i in range(13)]
    + [f"mfcc{i}_std" for i in range(13)]
)


def extract_features(path: str) -> list[float]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(audio, fmin=65, fmax=400, sr=sample_rate)
    voiced_f0 = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    tempo = float(librosa.feature.tempo(y=audio, sr=sample_rate)[0])
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    values = [
        float(voiced_f0.mean()) if voiced_f0.size else 0.0,
        float(voiced_f0.std()) if voiced_f0.size else 0.0,
        float(np.mean(voiced)) if voiced is not None else 0.0,
        float(rms.mean()),
        float(rms.std()),
        float(zcr.mean()),
        float(centroid.mean()),
        float(centroid.std()),
        float(bandwidth.mean()),
        float(rolloff.mean()),
        tempo,
    ]
    values += [float(value) for value in mfcc.mean(axis=1)]
    values += [float(value) for value in mfcc.std(axis=1)]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CREMA-D with source-trained frozen-fold prosody classifiers."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--encoding-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--group-label",
        help="Manifest label used to keep groups disjoint across folds (for example, actor_id).",
    )
    args = parser.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    items = evaluation_items(args.source_manifest, args.encoding_records)
    ids = []
    labels = []
    source_matrix = []
    target_matrix = []
    source_extraction_s = []
    extraction_s = []
    for item in items:
        gold = item["sample"]["labels"]["emotion"]
        ids.append(item["sample_id"])
        labels.append(gold)
        started = time.perf_counter()
        source_matrix.append(extract_features(item["sample"]["audio_path"]))
        source_extraction_s.append(time.perf_counter() - started)
        if item["representation"] == "source":
            target_matrix.append(source_matrix[-1])
            extraction_s.append(0.0)
        else:
            started = time.perf_counter()
            target_matrix.append(extract_features(item["audio_path"]))
            extraction_s.append(time.perf_counter() - started)

    source_x = np.nan_to_num(np.asarray(source_matrix, dtype=float))
    target_x = np.nan_to_num(np.asarray(target_matrix, dtype=float))
    y = np.asarray(labels)
    predictions = np.empty(len(y), dtype=object)
    fold_ids = np.empty(len(y), dtype=int)
    inference_s = np.zeros(len(y), dtype=float)
    if args.group_label:
        missing = [item["sample_id"] for item in items if args.group_label not in item["sample"]["labels"]]
        if missing:
            raise ValueError(
                f"manifest is missing group label {args.group_label!r} for {len(missing)} samples"
            )
        groups = np.asarray([item["sample"]["labels"][args.group_label] for item in items])
        unique_groups = np.unique(groups)
        if len(unique_groups) < args.folds:
            raise ValueError(
                f"group label {args.group_label!r} has {len(unique_groups)} groups, fewer than {args.folds} folds"
            )
        split_iterator = GroupKFold(n_splits=args.folds).split(source_x, y, groups)
        split_method = "group-k-fold"
    else:
        groups = None
        split_iterator = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=0
        ).split(source_x, y)
        split_method = "stratified-k-fold"

    for fold, (train_indices, test_indices) in enumerate(split_iterator):
        if groups is not None:
            overlap = set(groups[train_indices]) & set(groups[test_indices])
            if overlap:
                raise RuntimeError(f"group leakage in fold {fold}: {sorted(overlap)[:10]}")
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=0),
        )
        classifier.fit(source_x[train_indices], y[train_indices])
        started = time.perf_counter()
        predictions[test_indices] = classifier.predict(target_x[test_indices])
        elapsed = time.perf_counter() - started
        inference_s[test_indices] = elapsed / len(test_indices)
        fold_ids[test_indices] = fold

    rows = [
        {
            "sample_id": sample_id,
            "representation": items[index]["representation"],
            "operating_point": items[index]["operating_point"],
            "gold": labels[index],
            "prediction": str(predictions[index]),
            "correct": bool(predictions[index] == labels[index]),
            "fold": int(fold_ids[index]),
            "group": str(groups[index]) if groups is not None else None,
            "feature_extraction_s": extraction_s[index],
            "inference_s": float(inference_s[index]),
            "duration_s": items[index]["sample"]["duration_s"],
            "status": "ok",
        }
        for index, sample_id in enumerate(ids)
    ]
    write_jsonl(args.output, rows)
    total_audio_s = sum(row["duration_s"] for row in rows)
    total_compute_s = sum(extraction_s) + float(inference_s.sum())
    summary = {
        "evaluator": "source-trained-prosody-logistic-regression",
        "features": len(FEATURE_NAMES),
        "folds": args.folds,
        "fold_seed": 0,
        "split_method": split_method,
        "group_label": args.group_label,
        "groups": int(len(np.unique(groups))) if groups is not None else None,
        "training_representation": "source",
        "representation": items[0]["representation"],
        "operating_point": items[0]["operating_point"],
        "samples": len(rows),
        "failed": 0,
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "audio_hours": total_audio_s / 3600.0,
        "source_feature_extraction_s": sum(source_extraction_s),
        "feature_extraction_s": sum(extraction_s),
        "inference_s": float(inference_s.sum()),
        "rtf": total_compute_s / total_audio_s if total_audio_s else None,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
