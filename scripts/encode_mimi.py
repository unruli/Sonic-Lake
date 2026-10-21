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
    parser = argparse.ArgumentParser(description="Materialize packed Mimi tokens and decoded WAVs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, required=True)
    parser.add_argument("--model-id", default="kyutai/mimi")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, MimiModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = MimiModel.from_pretrained(args.model_id).eval().to(device)
    if args.codebooks < int(model.config.num_semantic_quantizers) or args.codebooks > int(model.config.num_quantizers):
        raise ValueError(
            f"codebooks must be in [{model.config.num_semantic_quantizers}, {model.config.num_quantizers}]"
        )
    target_sr = int(extractor.sampling_rate)
    vocab_size = int(model.config.codebook_size)

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
            if source_sr != target_sr:
                audio = torchaudio.functional.resample(audio.unsqueeze(0), source_sr, target_sr).squeeze(0)
            inputs = extractor(raw_audio=audio.numpy(), sampling_rate=target_sr, return_tensors="pt")
            input_values = inputs["input_values"].to(device)
            with torch.inference_mode():
                encoded = model.encode(input_values, num_quantizers=args.codebooks)
                codes = encoded.audio_codes if hasattr(encoded, "audio_codes") else encoded[0]
                decoded_output = model.decode(codes)
                decoded = decoded_output.audio_values if hasattr(decoded_output, "audio_values") else decoded_output[0]
                decoded = decoded[..., : input_values.shape[-1]]

            matrix = codes[0].detach().cpu().to(torch.int64).tolist()
            frames = len(matrix[0])
            duration = input_values.shape[-1] / float(target_sr)
            write_token_matrix(
                token_path,
                matrix,
                vocab_size=vocab_size,
                sample_rate=target_sr,
                frame_rate_hz=frames / duration,
                codec="mimi",
                model_revision=args.model_id,
            )
            decoded_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(decoded_path, decoded[0, 0].detach().cpu().numpy(), target_sr)
            bits_per_token = (vocab_size - 1).bit_length()
            row = encoding_record(
                sample,
                representation="mimi",
                operating_point=f"{args.codebooks}cb",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="ok",
                extra={
                    "model_id": args.model_id,
                    "codebooks": args.codebooks,
                    "frames": frames,
                    "frame_rate_hz": frames / duration,
                    "theoretical_kbps": args.codebooks * bits_per_token * frames / duration / 1000.0,
                },
            )
        except Exception as exc:
            row = encoding_record(
                sample,
                representation="mimi",
                operating_point=f"{args.codebooks}cb",
                token_path=token_path,
                decoded_path=decoded_path,
                encode_s=time.perf_counter() - started,
                status="failed",
                error_type=type(exc).__name__,
                stderr=traceback.format_exc(),
                extra={"model_id": args.model_id, "codebooks": args.codebooks},
            )
        rows.append(row)

    write_jsonl(args.records, rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"encoded {len(rows) - failed}/{len(rows)} files; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
