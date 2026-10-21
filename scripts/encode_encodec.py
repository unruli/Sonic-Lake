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
    parser = argparse.ArgumentParser(description="Materialize packed EnCodec tokens and decoded WAVs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bandwidth-kbps", type=float, required=True, choices=(1.5, 3.0, 6.0, 12.0, 24.0))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(args.bandwidth_kbps)
    model.eval().to(device)

    rows = []
    for index, sample in enumerate(read_jsonl(args.manifest)):
        if args.limit is not None and index >= args.limit:
            break
        token_path, decoded_path = artifact_paths(args.output_root, sample["sample_id"])
        started = time.perf_counter()
        try:
            wav, source_sr = sf.read(sample["audio_path"], always_2d=False)
            audio = torch.as_tensor(wav, dtype=torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.transpose(0, 1)
            input_values = convert_audio(audio, source_sr, model.sample_rate, model.channels).unsqueeze(0).to(device)
            with torch.inference_mode():
                encoded_frames = model.encode(input_values)
                decoded = model.decode(encoded_frames)[..., : input_values.shape[-1]]

            if any(scale is not None for _, scale in encoded_frames):
                raise RuntimeError("normalized EnCodec frames require scale side data; unsupported for packed-token accounting")
            codes = torch.cat([frame_codes for frame_codes, _ in encoded_frames], dim=-1)
            matrix = codes[0].detach().cpu().to(torch.int64).tolist()
            frames = len(matrix[0])
            duration = input_values.shape[-1] / float(model.sample_rate)
            write_token_matrix(
                token_path,
                matrix,
                vocab_size=1024,
                sample_rate=int(model.sample_rate),
                frame_rate_hz=frames / duration,
                codec="encodec",
                model_revision="encodec_24khz",
            )
            decoded_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(decoded_path, decoded[0].transpose(0, 1).detach().cpu().numpy(), model.sample_rate)
            row = encoding_record(
                sample,
                representation="encodec",
                operating_point=f"{args.bandwidth_kbps:g}kbps",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="ok",
                extra={
                    "bandwidth_kbps": args.bandwidth_kbps,
                    "codebooks": len(matrix),
                    "frames": frames,
                    "frame_rate_hz": frames / duration,
                    "theoretical_kbps": len(matrix) * 10.0 * frames / duration / 1000.0,
                },
            )
        except Exception as exc:
            row = encoding_record(
                sample,
                representation="encodec",
                operating_point=f"{args.bandwidth_kbps:g}kbps",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="failed",
                error_type=type(exc).__name__,
                stderr=traceback.format_exc(),
                extra={"bandwidth_kbps": args.bandwidth_kbps},
            )
        rows.append(row)

    write_jsonl(args.records, rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"encoded {len(rows) - failed}/{len(rows)} files; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
