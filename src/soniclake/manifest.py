from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Iterable

from .io import sha256_file, write_jsonl
from .schema import Sample, stable_sample_id


def audio_duration_s(path: str | Path) -> float:
    source = Path(path)
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    try:
        import soundfile as sf

        info = sf.info(str(source))
        return float(info.frames / info.samplerate)
    except (ImportError, RuntimeError):
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(source)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(process.stdout.strip())


def build_manifest(
    dataset: str,
    split: str,
    root: str | Path,
    audio_paths: Iterable[str | Path],
    label_rows: dict[str, dict] | None = None,
) -> list[Sample]:
    root_path = Path(root).resolve()
    labels = label_rows or {}
    samples: list[Sample] = []
    seen: set[str] = set()
    for source in sorted((Path(item).resolve() for item in audio_paths), key=str):
        relative = source.relative_to(root_path).as_posix()
        sample_id = stable_sample_id(dataset, split, relative)
        if sample_id in seen:
            raise ValueError(f"duplicate logical sample: {relative}")
        seen.add(sample_id)
        sample = Sample(
            sample_id=sample_id,
            dataset=dataset,
            split=split,
            audio_path=str(source),
            duration_s=audio_duration_s(source),
            sha256=sha256_file(source),
            labels=labels.get(relative, {}),
        )
        sample.validate()
        samples.append(sample)
    return samples


def load_label_map(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("label map must be a JSON object keyed by relative audio path")
    return payload


def save_manifest(path: str | Path, samples: Iterable[Sample]) -> None:
    write_jsonl(path, (sample.to_dict() for sample in samples))
