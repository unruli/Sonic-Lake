#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

from soniclake.evaluation import evaluation_items
from soniclake.io import write_jsonl


def fallback_normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one audio representation with frozen Faster-Whisper ASR.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--encoding-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    from rapidfuzz.distance import Levenshtein

    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
    except Exception:
        normalizer = fallback_normalize

    items = evaluation_items(args.source_manifest, args.encoding_records)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("evaluation set is empty")

    model_started = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    model_load_s = time.perf_counter() - model_started

    rows = []
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        try:
            segments, _ = model.transcribe(
                item["audio_path"], language="en", beam_size=args.beam_size, vad_filter=False
            )
            hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
            reference = item["sample"]["labels"]["transcript"]
            normalized_reference = normalizer(reference)
            normalized_hypothesis = normalizer(hypothesis)
            reference_words = normalized_reference.split()
            hypothesis_words = normalized_hypothesis.split()
            word_errors = Levenshtein.distance(reference_words, hypothesis_words)
            reference_chars = normalized_reference.replace(" ", "")
            hypothesis_chars = normalized_hypothesis.replace(" ", "")
            char_errors = Levenshtein.distance(reference_chars, hypothesis_chars)
            row = {
                "sample_id": item["sample_id"],
                "representation": item["representation"],
                "operating_point": item["operating_point"],
                "audio_path": item["audio_path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalized_reference,
                "normalized_hypothesis": normalized_hypothesis,
                "word_errors": word_errors,
                "reference_words": len(reference_words),
                "char_errors": char_errors,
                "reference_chars": len(reference_chars),
                "inference_s": time.perf_counter() - started,
                "duration_s": item["sample"]["duration_s"],
                "status": "ok",
                "error_type": "",
                "stderr": "",
            }
        except Exception as exc:
            row = {
                "sample_id": item["sample_id"],
                "representation": item["representation"],
                "operating_point": item["operating_point"],
                "audio_path": item["audio_path"],
                "inference_s": time.perf_counter() - started,
                "duration_s": item["sample"]["duration_s"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "stderr": traceback.format_exc()[-2000:],
            }
        rows.append(row)
        if index % args.checkpoint_every == 0:
            write_jsonl(args.output, rows)

    write_jsonl(args.output, rows)
    successful = [row for row in rows if row["status"] == "ok"]
    failed = len(rows) - len(successful)
    total_words = sum(row["reference_words"] for row in successful)
    total_chars = sum(row["reference_chars"] for row in successful)
    total_audio_s = sum(row["duration_s"] for row in successful)
    total_inference_s = sum(row["inference_s"] for row in successful)
    summary = {
        "evaluator": "faster-whisper",
        "model": args.model,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "representation": items[0]["representation"],
        "operating_point": items[0]["operating_point"],
        "samples": len(rows),
        "successful": len(successful),
        "failed": failed,
        "corpus_wer": sum(row["word_errors"] for row in successful) / total_words if total_words else None,
        "corpus_cer": sum(row["char_errors"] for row in successful) / total_chars if total_chars else None,
        "audio_hours": total_audio_s / 3600.0,
        "inference_s": total_inference_s,
        "rtf": total_inference_s / total_audio_s if total_audio_s else None,
        "model_load_s": model_load_s,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
