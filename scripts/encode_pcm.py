#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from soniclake.codec_artifacts import encoding_record
from soniclake.io import read_jsonl, write_jsonl


def ffmpeg_command(ffmpeg: str, source: str, output: Path) -> list[str]:
    """Build the lossless PCM-S16LE materialization command."""
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source,
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a lossless PCM-S16LE baseline.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = []
    for index, sample in enumerate(read_jsonl(args.manifest)):
        if args.limit is not None and index >= args.limit:
            break
        output = args.output_root / f"{sample['sample_id']}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        process = subprocess.run(
            ffmpeg_command(args.ffmpeg, sample["audio_path"], output),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            encoding_record(
                sample,
                representation="pcm_s16le",
                operating_point="source_sr",
                token_path=output,
                decoded_path=output,
                encode_s=elapsed,
                status="ok" if process.returncode == 0 else "failed",
                error_type="" if process.returncode == 0 else "ffmpeg",
                stderr=process.stderr,
                extra={"sample_format": "signed_16_bit_little_endian"},
            )
        )

    write_jsonl(args.records, rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"encoded {len(rows) - failed}/{len(rows)} files; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
