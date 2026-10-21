#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from soniclake.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a manifest to Opus and emit provenance/timing rows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bitrate-kbps", type=int, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--encoder", default="libopus", choices=("libopus", "opus"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = []
    for index, sample in enumerate(read_jsonl(args.manifest)):
        if args.limit is not None and index >= args.limit:
            break
        output = args.output_root / f"{sample['sample_id']}.opus"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        process = subprocess.run(
            [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", sample["audio_path"],
             "-c:a", args.encoder, "-strict", "experimental", "-b:a", f"{args.bitrate_kbps}k", str(output)],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        rows.append({
            "sample_id": sample["sample_id"],
            "source_sha256": sample["sha256"],
            "representation": "opus",
            "operating_point": f"{args.bitrate_kbps}kbps",
            "artifact_path": str(output),
            "artifact_bytes": output.stat().st_size if process.returncode == 0 else 0,
            "duration_s": sample["duration_s"],
            "encode_s": elapsed,
            "status": "ok" if process.returncode == 0 else "failed",
            "error_type": "" if process.returncode == 0 else "ffmpeg",
            "stderr": process.stderr[-2000:],
        })
    write_jsonl(args.records, rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"encoded {len(rows) - failed}/{len(rows)} files; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
