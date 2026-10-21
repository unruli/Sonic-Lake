#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from soniclake.codec_artifacts import artifact_paths, encoding_record
from soniclake.io import read_jsonl, write_jsonl
from soniclake.tokenpack import write_token_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize packed DAC tokens and decoded WAVs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, required=True)
    parser.add_argument("--model-type", default="24khz", choices=("16khz", "24khz", "44khz"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    import dac
    import soundfile as sf
    import torch
    import torchaudio
    from dac import DAC

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = dac.utils.download(model_type=args.model_type)
    model = DAC.load(checkpoint).eval().to(device)
    if args.codebooks < 1 or args.codebooks > int(model.n_codebooks):
        raise ValueError(f"codebooks must be in [1, {model.n_codebooks}]")

    rows = []
    for index, sample in enumerate(read_jsonl(args.manifest)):
        if args.limit is not None and index >= args.limit:
            break
        token_path, decoded_path = artifact_paths(args.output_root, sample["sample_id"])
        started = time.perf_counter()
        try:
            wav, source_sr = sf.read(sample["audio_path"], always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            audio = torch.as_tensor(wav, dtype=torch.float32)
            if source_sr != model.sample_rate:
                audio = torchaudio.functional.resample(audio.unsqueeze(0), source_sr, model.sample_rate).squeeze(0)
            input_values = audio.unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                _, all_codes, _, _, _ = model.encode(input_values)
                codes = all_codes[:, : args.codebooks, :]
                quantized = model.quantizer.from_codes(codes)[0]
                decoded = model.decode(quantized)[..., : input_values.shape[-1]]

            matrix = codes[0].detach().cpu().to(torch.int64).tolist()
            frames = len(matrix[0])
            duration = input_values.shape[-1] / float(model.sample_rate)
            write_token_matrix(
                token_path,
                matrix,
                vocab_size=1024,
                sample_rate=int(model.sample_rate),
                frame_rate_hz=frames / duration,
                codec="dac",
                model_revision=args.model_type,
            )
            decoded_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(decoded_path, decoded[0, 0].detach().cpu().numpy(), model.sample_rate)
            row = encoding_record(
                sample,
                representation="dac",
                operating_point=f"{args.codebooks}cb",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="ok",
                extra={
                    "model_type": args.model_type,
                    "codebooks": args.codebooks,
                    "frames": frames,
                    "frame_rate_hz": frames / duration,
                    "theoretical_kbps": args.codebooks * 10.0 * frames / duration / 1000.0,
                },
            )
        except Exception as exc:
            row = encoding_record(
                sample,
                representation="dac",
                operating_point=f"{args.codebooks}cb",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="failed",
                error_type=type(exc).__name__,
                stderr=traceback.format_exc(),
                extra={"model_type": args.model_type, "codebooks": args.codebooks},
            )
        rows.append(row)

    write_jsonl(args.records, rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"encoded {len(rows) - failed}/{len(rows)} files; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
