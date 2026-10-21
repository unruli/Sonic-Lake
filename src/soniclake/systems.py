from __future__ import annotations

import io
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AccessMeasurement:
    sample_id: str
    artifact_path: str
    artifact_bytes: int
    submitted_ns: int
    started_ns: int
    finished_ns: int
    eviction_s: float
    read_s: float
    decode_s: float

    @property
    def queue_s(self) -> float:
        return (self.started_ns - self.submitted_ns) / 1e9

    @property
    def response_s(self) -> float:
        return (self.finished_ns - self.submitted_ns) / 1e9

    def to_dict(self) -> dict:
        row = asdict(self)
        row.update(queue_s=self.queue_s, response_s=self.response_s)
        return row


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def evict_file(path: str | Path) -> bool:
    """Best-effort eviction without root; supported by Linux/HPC cluster."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    return True


def warm_files(paths: Iterable[str | Path]) -> int:
    total = 0
    for path in dict.fromkeys(str(item) for item in paths):
        with open(path, "rb") as handle:
            while block := handle.read(1024 * 1024):
                total += len(block)
    return total


def decode_bytes(payload: bytes, codec: str, ffmpeg: str = "ffmpeg") -> None:
    if codec in {"source", "pcm", "flac"}:
        import soundfile as sf

        sf.read(io.BytesIO(payload), always_2d=False)
        return
    if codec == "opus":
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", "pipe:0", "-f", "null", "-"],
            input=payload,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        return
    raise ValueError(f"concurrent access benchmark does not support codec {codec!r}")


def access_one(
    item: dict,
    *,
    codec: str,
    cache_mode: str,
    submitted_ns: int,
    ffmpeg: str = "ffmpeg",
) -> AccessMeasurement:
    path = str(item["artifact_path"])
    started_ns = time.perf_counter_ns()
    eviction_started = time.perf_counter()
    if cache_mode == "cold":
        evict_file(path)
    eviction_s = time.perf_counter() - eviction_started if cache_mode == "cold" else 0.0
    read_started = time.perf_counter()
    payload = Path(path).read_bytes()
    read_s = time.perf_counter() - read_started
    decode_started = time.perf_counter()
    decode_bytes(payload, codec, ffmpeg)
    decode_s = time.perf_counter() - decode_started
    finished_ns = time.perf_counter_ns()
    return AccessMeasurement(
        sample_id=str(item["sample_id"]),
        artifact_path=path,
        artifact_bytes=len(payload),
        submitted_ns=submitted_ns,
        started_ns=started_ns,
        finished_ns=finished_ns,
        eviction_s=eviction_s,
        read_s=read_s,
        decode_s=decode_s,
    )


def run_closed_loop(
    items: list[dict],
    *,
    codec: str,
    concurrency: int,
    cache_mode: str,
    ffmpeg: str = "ffmpeg",
) -> tuple[list[AccessMeasurement], dict]:
    if not items:
        raise ValueError("items cannot be empty")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if cache_mode not in {"warm", "cold"}:
        raise ValueError("cache_mode must be warm or cold")
    if cache_mode == "warm":
        warm_files(item["artifact_path"] for item in items)

    run_started = time.perf_counter_ns()
    measurements: list[AccessMeasurement] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for item in items:
            submitted_ns = time.perf_counter_ns()
            futures.append(executor.submit(
                access_one,
                item,
                codec=codec,
                cache_mode=cache_mode,
                submitted_ns=submitted_ns,
                ffmpeg=ffmpeg,
            ))
        for future in as_completed(futures):
            measurements.append(future.result())
    wall_s = (time.perf_counter_ns() - run_started) / 1e9
    measurements.sort(key=lambda row: (row.submitted_ns, row.sample_id))
    queue = [row.queue_s for row in measurements]
    evictions = [row.eviction_s for row in measurements]
    reads = [row.read_s for row in measurements]
    decodes = [row.decode_s for row in measurements]
    responses = [row.response_s for row in measurements]
    summary = {
        "method": "closed-loop concurrent real artifact read and decode",
        "codec": codec,
        "cache_mode": cache_mode,
        "concurrency": concurrency,
        "requests": len(measurements),
        "wall_s": wall_s,
        "throughput_queries_per_s": len(measurements) / wall_s,
        "throughput_mb_per_s": sum(row.artifact_bytes for row in measurements) / 1_000_000 / wall_s,
        "queue_median_ms": statistics.median(queue) * 1000,
        "queue_p95_ms": percentile(queue, 0.95) * 1000,
        "eviction_median_ms": statistics.median(evictions) * 1000,
        "eviction_p95_ms": percentile(evictions, 0.95) * 1000,
        "read_median_ms": statistics.median(reads) * 1000,
        "read_p95_ms": percentile(reads, 0.95) * 1000,
        "decode_median_ms": statistics.median(decodes) * 1000,
        "decode_p95_ms": percentile(decodes, 0.95) * 1000,
        "response_median_ms": statistics.median(responses) * 1000,
        "response_p95_ms": percentile(responses, 0.95) * 1000,
        "cache_eviction_supported": hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"),
    }
    return measurements, summary
