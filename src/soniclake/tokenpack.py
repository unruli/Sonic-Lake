from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Iterable


MAGIC = b"SNCLTOK1"


def _pack_values(values: Iterable[int], bits: int) -> bytes:
    accumulator = 0
    buffered = 0
    output = bytearray()
    for value in values:
        if value < 0 or value >= (1 << bits):
            raise ValueError(f"token {value} does not fit in {bits} bits")
        accumulator |= value << buffered
        buffered += bits
        while buffered >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            buffered -= 8
    if buffered:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _unpack_values(payload: bytes, count: int, bits: int) -> list[int]:
    values: list[int] = []
    accumulator = 0
    buffered = 0
    iterator = iter(payload)
    for _ in range(count):
        while buffered < bits:
            try:
                accumulator |= next(iterator) << buffered
            except StopIteration as exc:
                raise ValueError("truncated token payload") from exc
            buffered += 8
        values.append(accumulator & ((1 << bits) - 1))
        accumulator >>= bits
        buffered -= bits
    return values


def write_token_matrix(
    path: str | Path,
    rows: Iterable[Iterable[int]],
    *,
    vocab_size: int,
    sample_rate: int,
    frame_rate_hz: float,
    codec: str,
    model_revision: str,
) -> dict:
    matrix = [list(row) for row in rows]
    if not matrix or not matrix[0]:
        raise ValueError("token matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("token matrix must be rectangular")
    bits = max(1, math.ceil(math.log2(vocab_size)))
    flat = [value for row in matrix for value in row]
    header = {
        "codec": codec,
        "model_revision": model_revision,
        "vocab_size": vocab_size,
        "bits_per_token": bits,
        "codebooks": len(matrix),
        "frames": width,
        "sample_rate": sample_rate,
        "frame_rate_hz": frame_rate_hz,
        "packing": "row-major-lsb-bitpack-v1"
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(_pack_values(flat, bits))
    return header


def read_token_matrix(path: str | Path) -> tuple[dict, list[list[int]]]:
    with Path(path).open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a SonicLake packed-token artifact")
        length_raw = stream.read(4)
        if len(length_raw) != 4:
            raise ValueError("truncated token header length")
        header_length = struct.unpack("<I", length_raw)[0]
        header = json.loads(stream.read(header_length))
        payload = stream.read()
    count = int(header["codebooks"]) * int(header["frames"])
    flat = _unpack_values(payload, count, int(header["bits_per_token"]))
    frames = int(header["frames"])
    matrix = [flat[start:start + frames] for start in range(0, count, frames)]
    return header, matrix

