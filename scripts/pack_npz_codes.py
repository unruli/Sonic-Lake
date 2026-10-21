#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from soniclake.tokenpack import write_token_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a codec NPZ code matrix into deterministic packed tokens.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--frame-rate-hz", type=float, required=True)
    parser.add_argument("--key", default="codes")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required to read NPZ codec outputs") from exc
    with np.load(args.input, allow_pickle=False) as archive:
        codes = archive[args.key]
    if codes.ndim == 1:
        codes = codes[None, :]
    if codes.ndim != 2:
        raise SystemExit(f"expected a 1D/2D token matrix, received shape {codes.shape}")
    header = write_token_matrix(
        args.output,
        codes.astype(int).tolist(),
        vocab_size=args.vocab_size,
        sample_rate=args.sample_rate,
        frame_rate_hz=args.frame_rate_hz,
        codec=args.codec,
        model_revision=args.model_revision,
    )
    print(f"wrote {args.output.stat().st_size} bytes: {header}")


if __name__ == "__main__":
    main()

