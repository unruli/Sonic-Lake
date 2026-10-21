#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from soniclake.manifest import build_manifest, load_label_map, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a checksummed SonicLake JSONL manifest from WAV files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.wav", dest="pattern")
    parser.add_argument("--labels", type=Path, help="JSON object keyed by audio path relative to --root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audio_paths = list(args.root.glob(args.pattern))
    if not audio_paths:
        raise SystemExit(f"no audio files matched {args.pattern!r} under {args.root}")
    samples = build_manifest(
        dataset=args.dataset,
        split=args.split,
        root=args.root,
        audio_paths=audio_paths,
        label_rows=load_label_map(args.labels),
    )
    save_manifest(args.output, samples)
    total_hours = sum(sample.duration_s for sample in samples) / 3600
    print(f"wrote {len(samples)} samples ({total_hours:.3f} h) to {args.output}")


if __name__ == "__main__":
    main()

