#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

from soniclake.evaluation import evaluation_items
from soniclake.io import write_jsonl


def class_prompt(category: str) -> str:
    return f"a recording of the sound of {category.replace('_', ' ')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ESC-50 with frozen zero-shot CLAP prompts.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--encoding-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--version", default="2023")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

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
    items = evaluation_items(args.source_manifest, args.encoding_records)
    categories = sorted({item["sample"]["labels"]["category"] for item in items})
    if len(categories) != 50:
        raise ValueError(f"ESC-50 evaluator requires 50 categories, found {len(categories)}")
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("evaluation set is empty")

    model_started = time.perf_counter()
    model = CLAP(version=args.version, use_cuda=torch.cuda.is_available())
    text_embeddings = torch.nn.functional.normalize(
        model.get_text_embeddings([class_prompt(category) for category in categories]).cpu(), dim=-1
    )
    model_load_s = time.perf_counter() - model_started

    rows = []
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        gold = item["sample"]["labels"]["category"]
        try:
            audio_embedding = torch.nn.functional.normalize(
                model.get_audio_embeddings([item["audio_path"]]).cpu(), dim=-1
            )
            similarities = (audio_embedding @ text_embeddings.T)[0]
            predicted_index = int(similarities.argmax())
            prediction = categories[predicted_index]
            row = {
                "sample_id": item["sample_id"],
                "representation": item["representation"],
                "operating_point": item["operating_point"],
                "audio_path": item["audio_path"],
                "gold": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "confidence": float(similarities.softmax(dim=0)[predicted_index]),
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
                "gold": gold,
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
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row in successful:
        by_class[row["gold"]].append(row["correct"])
    total_audio_s = sum(row["duration_s"] for row in successful)
    total_inference_s = sum(row["inference_s"] for row in successful)
    summary = {
        "evaluator": "msclap-zero-shot",
        "version": args.version,
        "prompt_template": class_prompt("{category}"),
        "representation": items[0]["representation"],
        "operating_point": items[0]["operating_point"],
        "samples": len(rows),
        "successful": len(successful),
        "failed": failed,
        "accuracy": sum(row["correct"] for row in successful) / len(successful) if successful else None,
        "macro_accuracy": (
            sum(sum(values) / len(values) for values in by_class.values()) / len(by_class) if by_class else None
        ),
        "classes_observed": len(by_class),
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
