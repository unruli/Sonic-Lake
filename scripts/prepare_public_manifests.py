#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from soniclake.datasets import crema_subset_manifest, esc50_manifest, librispeech_manifest
from soniclake.manifest import save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled manifests for SonicLake public datasets.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for name in ("librispeech", "esc50", "crema_d"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "crema_d":
            child.add_argument("--actor-map", type=Path)
    args = parser.parse_args()

    if args.dataset == "librispeech":
        samples = librispeech_manifest(args.root)
    elif args.dataset == "esc50":
        samples = esc50_manifest(args.root)
    else:
        samples = crema_subset_manifest(args.root, actor_map=args.actor_map)
    if not samples:
        raise SystemExit(f"no samples found for {args.dataset} under {args.root}")
    save_manifest(args.output, samples)
    hours = sum(sample.duration_s for sample in samples) / 3600
    print(f"wrote {len(samples)} labeled {args.dataset} samples ({hours:.3f} h) to {args.output}")


if __name__ == "__main__":
    main()
