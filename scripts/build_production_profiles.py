#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soniclake.io import read_jsonl, write_jsonl
from soniclake.profiles import aggregate_results, profiles_to_dict


REPRESENTATIONS = {
    "pcm": "pcm_s16le",
    "flac": "flac_lossless_c8",
    "opus24": "opus_24kbps",
    "opus48": "opus_48kbps",
    "opus96": "opus_96kbps",
    "dac4": "dac_4",
    "dac8": "dac_8",
    "dac16": "dac_16",
    "encodec3": "encodec_3",
    "encodec6": "encodec_6",
    "encodec12": "encodec_12",
    "mimi8": "mimi_8",
    "mimi16": "mimi_16",
    "mimi32": "mimi_32",
}

FAMILIES = {
    "lexical": ("librispeech", "asr"),
    "acoustic_event": ("esc50", "events"),
    "emotion": ("crema_d", "emotion"),
}


def quality_from_summary(family: str, summary: dict) -> float:
    if family == "lexical":
        return 1.0 - float(summary["corpus_wer"])
    if family == "acoustic_event":
        return float(summary["macro_accuracy"])
    if family == "emotion":
        return float(summary["macro_f1"])
    raise ValueError(f"unsupported family: {family}")


def query_seconds(family: str, row: dict) -> float:
    elapsed = float(row["inference_s"])
    if family == "emotion":
        elapsed += float(row["feature_extraction_s"])
    return elapsed


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(root: Path) -> list[dict]:
    evaluation_root = root / "results" / "production" / "evaluation"
    encoding_root = root / "results" / "production" / "encoding"
    decode_root = root / "results" / "production" / "decode"
    output = []
    for name, encoding_suffix in REPRESENTATIONS.items():
        for family, (dataset, evaluation_prefix) in FAMILIES.items():
            decode_path = decode_root / f"{dataset}_{name}.summary.json"
            if dataset == "librispeech" and not decode_path.exists():
                # Backward-compatible name used by the first LibriSpeech latency run.
                decode_path = decode_root / f"{name}.summary.json"
            decode_summary = load_json(decode_path)
            decode_rtf = float(decode_summary["aggregate_decode_rtf"])
            encoded = {
                row["sample_id"]: row
                for row in read_jsonl(encoding_root / f"{dataset}_{encoding_suffix}.jsonl")
                if row["status"] == "ok"
            }
            evaluation_path = evaluation_root / f"{evaluation_prefix}_{name}.jsonl"
            evaluation_rows = list(read_jsonl(evaluation_path))
            summary = load_json(evaluation_root / f"{evaluation_prefix}_{name}.summary.json")
            quality = quality_from_summary(family, summary)
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"invalid quality for {family}/{name}: {quality}")
            if {row["sample_id"] for row in evaluation_rows} != set(encoded):
                raise ValueError(f"evaluation/encoding mismatch for {family}/{name}")
            for row in evaluation_rows:
                artifact = encoded[row["sample_id"]]
                status = row["status"]
                output.append(
                    {
                        "run_id": f"production-{family}-{name}",
                        "sample_id": row["sample_id"],
                        "dataset": dataset,
                        "representation": artifact["representation"],
                        "operating_point": artifact["operating_point"],
                        "query_family": family,
                        "source_sha256": artifact["source_sha256"],
                        "artifact_path": artifact["artifact_path"],
                        "artifact_bytes": artifact["artifact_bytes"],
                        "duration_s": artifact["duration_s"],
                        "encode_s": artifact["encode_s"],
                        "decode_s": float(artifact["duration_s"]) * decode_rtf,
                        "query_s": query_seconds(family, row),
                        "quality": quality if status == "ok" else None,
                        "status": status,
                        "error_type": row.get("error_type", ""),
                        "adapter_version": summary["evaluator"],
                        "model_revision": str(summary.get("model", summary.get("version", ""))),
                        "environment": "hpc-cluster-production-2026-10",
                        "seed": summary.get("fold_seed"),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Join production measurements into optimizer-ready profiles.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-profiles", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(args.root)
    write_jsonl(args.output_results, rows)
    payload = profiles_to_dict(aggregate_results(rows))
    args.output_profiles.parent.mkdir(parents=True, exist_ok=True)
    args.output_profiles.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} run rows and {len(payload)} profiles")


if __name__ == "__main__":
    main()
