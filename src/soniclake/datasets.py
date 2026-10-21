from __future__ import annotations

import csv
from pathlib import Path

from .io import sha256_file
from .manifest import audio_duration_s
from .schema import Sample, stable_sample_id


def _sample(dataset: str, split: str, root: Path, audio: Path, labels: dict) -> Sample:
    relative = audio.relative_to(root).as_posix()
    sample = Sample(
        sample_id=stable_sample_id(dataset, split, relative),
        dataset=dataset,
        split=split,
        audio_path=str(audio.resolve()),
        duration_s=audio_duration_s(audio),
        sha256=sha256_file(audio),
        labels=labels,
    )
    sample.validate()
    return sample


def librispeech_manifest(root: str | Path, split: str = "test-clean") -> list[Sample]:
    root_path = Path(root).resolve()
    transcripts: dict[str, str] = {}
    for transcript_file in root_path.glob("**/*.trans.txt"):
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(maxsplit=1)
            transcripts[utterance_id] = text
    samples: list[Sample] = []
    for audio in sorted(root_path.glob("**/*.flac")):
        utterance_id = audio.stem
        parts = utterance_id.split("-")
        if len(parts) < 3 or utterance_id not in transcripts:
            raise ValueError(f"cannot resolve LibriSpeech metadata for {audio}")
        samples.append(_sample("librispeech", split, root_path, audio, {
            "speaker_id": parts[0],
            "chapter_id": parts[1],
            "utterance_id": utterance_id,
            "transcript": transcripts[utterance_id],
        }))
    return samples


def esc50_manifest(root: str | Path) -> list[Sample]:
    root_path = Path(root).resolve()
    metadata = root_path / "meta" / "esc50.csv"
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    samples: list[Sample] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            audio = root_path / "audio" / row["filename"]
            split = f"fold{row['fold']}"
            samples.append(_sample("esc50", split, root_path, audio, {
                "target": int(row["target"]),
                "category": row["category"],
                "esc10": row["esc10"].strip().lower() == "true",
                "source_file": row["src_file"],
                "take": row["take"],
                "group_id": row["src_file"],
            }))
    return samples


def crema_subset_manifest(
    root: str | Path,
    split: str = "evaluation",
    actor_map: str | Path | None = None,
) -> list[Sample]:
    root_path = Path(root).resolve()
    known = {"anger", "disgust", "fear", "happy", "neutral", "sad"}
    actors: dict[str, str] = {}
    if actor_map is not None:
        with Path(actor_map).open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                actors[row["recording_id"]] = row["actor_id"]
    samples: list[Sample] = []
    for audio in sorted(root_path.glob("*.wav")):
        emotion = audio.stem.rsplit("_", 1)[-1].lower()
        if emotion not in known:
            raise ValueError(f"cannot parse CREMA-D emotion from {audio.name}")
        labels = {"emotion": emotion}
        if actor_map is not None:
            if audio.stem not in actors:
                raise ValueError(f"actor map has no entry for {audio.stem}")
            labels["actor_id"] = actors[audio.stem]
        samples.append(_sample("crema_d", split, root_path, audio, labels))
    return samples


def assert_group_disjoint(samples: list[Sample], group_label: str, split_a: set[str], split_b: set[str]) -> None:
    groups_a = {str(sample.labels[group_label]) for sample in samples if sample.split in split_a}
    groups_b = {str(sample.labels[group_label]) for sample in samples if sample.split in split_b}
    overlap = groups_a & groups_b
    if overlap:
        raise ValueError(f"group leakage across splits for {group_label}: {sorted(overlap)[:10]}")
