#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EMOTIONS = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
SOURCE_DATASET = "myleslinder/crema-d"
SOURCE_REVISION = "8a11ae8cb89df0d49a85ab0a64410a9f354c11e4"


def actor_from_example(example: dict, audio_key: str) -> tuple[str, str]:
    for key in ("actor_id", "actor", "speaker_id", "speaker"):
        value = example.get(key)
        if value not in (None, ""):
            return str(value), key
    audio = example[audio_key]
    source = str(audio.get("path") or "") if isinstance(audio, dict) else str(audio)
    match = re.search(r"(?:^|[/\\])(\d{4})_", source)
    if match is None:
        match = re.search(r"^(\d{4})_", Path(source).name)
    if match is None:
        raise ValueError(f"cannot recover actor from fields={sorted(example)} audio={source!r}")
    return match.group(1), source


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover CREMA-D actors from preserved source row indices.")
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import Audio, ClassLabel, load_dataset

    selected = []
    for wav in sorted(args.wav_root.glob("cremad_*.wav")):
        match = re.fullmatch(r"cremad_(\d+)_(\w+)", wav.stem)
        if match is None:
            raise ValueError(f"unexpected retained filename: {wav.name}")
        selected.append((wav.stem, int(match.group(1)), match.group(2).lower()))

    dataset = load_dataset(
        SOURCE_DATASET,
        split="train",
        revision=SOURCE_REVISION,
        trust_remote_code=True,
    )

    audio_key = next(key for key, value in dataset.features.items() if isinstance(value, Audio))
    label_key = next(key for key in ("major_emotion", "label", "emotion", "labels", "class") if key in dataset.features)
    label_names = dataset.features[label_key].names if isinstance(dataset.features[label_key], ClassLabel) else None
    dataset = dataset.cast_column(audio_key, Audio(decode=False))

    rows = []
    for recording_id, index, expected_emotion in selected:
        example = dataset[index]
        raw_label = example[label_key]
        source_emotion = label_names[raw_label] if label_names else str(raw_label)
        source_emotion = EMOTIONS.get(str(source_emotion).upper()[:3], str(source_emotion).lower())
        if source_emotion != expected_emotion:
            raise ValueError(
                f"row {index} emotion mismatch: retained={expected_emotion}, source={source_emotion}"
            )
        actor_id, provenance = actor_from_example(example, audio_key)
        rows.append({
            "recording_id": recording_id,
            "actor_id": actor_id,
            "source_row": index,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_actor_provenance": provenance,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"recovered {len(rows)} clips across {len({row['actor_id'] for row in rows})} "
        f"actors from {SOURCE_DATASET}@{SOURCE_REVISION}"
    )


if __name__ == "__main__":
    main()
