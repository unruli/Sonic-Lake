#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Callable

from soniclake.io import read_jsonl, write_jsonl
from soniclake.tokenpack import read_token_matrix


def evenly_spaced(rows: list[dict], limit: int | None) -> list[dict]:
    """Choose deterministic samples spanning the full manifest order."""
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_items(manifest: Path, records: Path | None) -> list[dict]:
    sources = list(read_jsonl(manifest))
    if records is None:
        return [
            {
                "sample_id": row["sample_id"],
                "duration_s": row["duration_s"],
                "artifact_path": row["audio_path"],
            }
            for row in sources
        ]
    encoded = {row["sample_id"]: row for row in read_jsonl(records) if row.get("status") == "ok"}
    source_ids = {row["sample_id"] for row in sources}
    if set(encoded) != source_ids:
        raise ValueError(
            f"manifest/encoding mismatch: source={len(source_ids)}, encoded={len(encoded)}, "
            f"missing={len(source_ids - set(encoded))}, extra={len(set(encoded) - source_ids)}"
        )
    return [encoded[row["sample_id"]] for row in sources]


def sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def build_decoder(codec: str, device: str) -> Callable[[str], tuple[float, float]]:
    if codec in {"source", "pcm", "flac"}:
        import soundfile as sf

        def decode_audio(path: str) -> tuple[float, float]:
            started = time.perf_counter()
            sf.read(path, always_2d=False)
            elapsed = time.perf_counter() - started
            return elapsed, 0.0

        return decode_audio

    if codec == "opus":
        def decode_opus(path: str) -> tuple[float, float]:
            started = time.perf_counter()
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "null", "-"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            elapsed = time.perf_counter() - started
            return elapsed, 0.0

        return decode_opus

    import torch

    target = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if codec == "dac":
        import dac
        from dac import DAC

        model = DAC.load(dac.utils.download(model_type="24khz")).eval().to(target)

        def decode_dac(path: str) -> tuple[float, float]:
            read_started = time.perf_counter()
            _, matrix = read_token_matrix(path)
            codes = torch.tensor(matrix, dtype=torch.long, device=target).unsqueeze(0)
            read_s = time.perf_counter() - read_started
            sync_cuda(torch)
            compute_started = time.perf_counter()
            with torch.inference_mode():
                quantized = model.quantizer.from_codes(codes)[0]
                model.decode(quantized)
            sync_cuda(torch)
            return read_s, time.perf_counter() - compute_started

        return decode_dac

    if codec == "encodec":
        from encodec import EncodecModel

        model = EncodecModel.encodec_model_24khz().eval().to(target)

        def decode_encodec(path: str) -> tuple[float, float]:
            read_started = time.perf_counter()
            _, matrix = read_token_matrix(path)
            codes = torch.tensor(matrix, dtype=torch.long, device=target).unsqueeze(0)
            read_s = time.perf_counter() - read_started
            sync_cuda(torch)
            compute_started = time.perf_counter()
            with torch.inference_mode():
                model.decode([(codes, None)])
            sync_cuda(torch)
            return read_s, time.perf_counter() - compute_started

        return decode_encodec

    if codec == "mimi":
        from transformers import MimiModel

        model = MimiModel.from_pretrained("kyutai/mimi").eval().to(target)

        def decode_mimi(path: str) -> tuple[float, float]:
            read_started = time.perf_counter()
            _, matrix = read_token_matrix(path)
            codes = torch.tensor(matrix, dtype=torch.long, device=target).unsqueeze(0)
            read_s = time.perf_counter() - read_started
            sync_cuda(torch)
            compute_started = time.perf_counter()
            with torch.inference_mode():
                model.decode(codes)
            sync_cuda(torch)
            return read_s, time.perf_counter() - compute_started

        return decode_mimi

    raise ValueError(f"unsupported codec: {codec}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark artifact read plus codec decode latency.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoding-records", type=Path)
    parser.add_argument(
        "--codec",
        required=True,
        choices=("source", "pcm", "flac", "opus", "dac", "encodec", "mimi"),
    )
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    items = evenly_spaced(load_items(args.manifest, args.encoding_records), args.limit)
    model_started = time.perf_counter()
    decoder = build_decoder(args.codec, args.device)
    model_load_s = time.perf_counter() - model_started

    # One unmeasured warm-up removes lazy kernel/library initialization from the benchmark.
    decoder(items[0]["artifact_path"])

    rows = []
    for item in items:
        try:
            read_s, compute_s = decoder(item["artifact_path"])
            duration_s = float(item["duration_s"])
            decode_s = read_s + compute_s
            row = {
                "sample_id": item["sample_id"],
                "representation": args.representation,
                "codec": args.codec,
                "artifact_path": item["artifact_path"],
                "duration_s": duration_s,
                "artifact_read_s": read_s,
                "decode_compute_s": compute_s,
                "decode_s": decode_s,
                "decode_rtf": decode_s / duration_s,
                "status": "ok",
                "error_type": "",
            }
        except Exception as exc:
            row = {
                "sample_id": item["sample_id"],
                "representation": args.representation,
                "codec": args.codec,
                "artifact_path": item["artifact_path"],
                "duration_s": float(item["duration_s"]),
                "artifact_read_s": 0.0,
                "decode_compute_s": 0.0,
                "decode_s": 0.0,
                "decode_rtf": 0.0,
                "status": "failed",
                "error_type": type(exc).__name__,
                "stderr": traceback.format_exc()[-2000:],
            }
        rows.append(row)

    write_jsonl(args.output, rows)
    good = [row for row in rows if row["status"] == "ok"]
    if not good:
        raise RuntimeError("every decode failed")
    latencies = [row["decode_s"] for row in good]
    total_audio_s = sum(row["duration_s"] for row in good)
    summary = {
        "codec": args.codec,
        "representation": args.representation,
        "samples": len(rows),
        "failed": len(rows) - len(good),
        "model_load_s": model_load_s,
        "mean_decode_s": statistics.fmean(latencies),
        "median_decode_s": statistics.median(latencies),
        "p95_decode_s": percentile(latencies, 0.95),
        "aggregate_decode_rtf": sum(latencies) / total_audio_s,
        "mean_artifact_read_s": statistics.fmean(row["artifact_read_s"] for row in good),
        "mean_decode_compute_s": statistics.fmean(row["decode_compute_s"] for row in good),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if len(good) != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
