#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SHORT_NAMES = {
    "pcm_s16le@source_sr": "pcm",
    "flac@lossless_c8": "flac",
    "opus@24kbps": "opus24",
    "opus@48kbps": "opus48",
    "opus@96kbps": "opus96",
    "dac@4cb": "dac4",
    "dac@8cb": "dac8",
    "dac@16cb": "dac16",
    "encodec@3kbps": "encodec3",
    "encodec@6kbps": "encodec6",
    "encodec@12kbps": "encodec12",
    "mimi@8cb": "mimi8",
    "mimi@16cb": "mimi16",
    "mimi@32cb": "mimi32",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(profiles: dict, retrieval_root: Path) -> dict:
    result = {}
    for name, profile in profiles.items():
        short_name = SHORT_NAMES[name]
        retrieval = load_json(retrieval_root / f"{short_name}.summary.json")
        updated = json.loads(json.dumps(profile))
        updated["storage_bytes_per_hour"] += float(retrieval["index_bytes_per_audio_hour"])
        updated["construction_s_per_hour"] += (
            float(retrieval["transcription_s_per_audio_hour"])
            + float(retrieval["index_build_s"]) / float(retrieval["audio_hours"])
        )
        updated["families"]["lexical"] = {
            "quality": float(retrieval["macro_f1"]),
            "query_cost_s": float(retrieval["median_lookup_ns"]) / 1_000_000_000,
        }
        updated["derived_representation"] = "asr_transcript_inverted_index"
        updated["retrieval_queries"] = int(retrieval["queries"])
        result[name] = updated
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach ASR transcript indexes to audio profiles for retrieval-aware optimization."
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--output-source-quality", type=Path)
    args = parser.parse_args()

    payload = build(load_json(args.profiles), args.retrieval_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (args.analysis_root is None) != (args.output_source_quality is None):
        raise ValueError("analysis-root and output-source-quality must be supplied together")
    if args.analysis_root is not None and args.output_source_quality is not None:
        source_retrieval = load_json(args.retrieval_root / "source.summary.json")
        event_quality = load_json(args.analysis_root / "quality_events.json")
        emotion_quality = load_json(args.analysis_root / "quality_emotion.json")
        source_quality = {
            "lexical": float(source_retrieval["macro_f1"]),
            "acoustic_event": float(event_quality["baseline_estimate"]),
            "emotion": float(emotion_quality["baseline_estimate"]),
        }
        args.output_source_quality.parent.mkdir(parents=True, exist_ok=True)
        args.output_source_quality.write_text(
            json.dumps(source_quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(payload)} retrieval-aware representation profiles")


if __name__ == "__main__":
    main()
