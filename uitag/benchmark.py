"""Benchmarking utilities for comparing backend performance."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    backend: str
    total_ms: float
    per_quadrant_ms: list[float]
    detection_count: int
    image_path: str

    @property
    def mean_per_quadrant_ms(self) -> float:
        if not self.per_quadrant_ms:
            return 0.0
        return sum(self.per_quadrant_ms) / len(self.per_quadrant_ms)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "total_ms": round(self.total_ms, 1),
            "mean_per_quadrant_ms": round(self.mean_per_quadrant_ms, 1),
            "per_quadrant_ms": [round(t, 1) for t in self.per_quadrant_ms],
            "detection_count": self.detection_count,
            "image_path": self.image_path,
        }


def run_benchmark(
    backend,
    image_path: str,
    task: str = "<OD>",
    overlap_px: int = 50,
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
) -> list[BenchmarkResult]:
    """Run benchmark against a backend, returning results for each run.

    Args:
        backend: A DetectionBackend instance.
        image_path: Path to the test image.
        task: Florence-2 task token.
        overlap_px: Quadrant overlap.
        warmup_runs: Number of warmup iterations (not measured).
        benchmark_runs: Number of measured iterations.

    Returns:
        List of BenchmarkResult, one per measured run.
    """
    from PIL import Image as PILImage
    from uitag.quadrants import split_object_aware
    from uitag.vision import run_vision_detect

    # Prepare quadrants once
    img = PILImage.open(image_path)
    vision_dets, _ = run_vision_detect(image_path)
    quads, _ = split_object_aware(img, vision_dets, overlap_px=overlap_px)
    quad_inputs = [(q.image, q.offset_x, q.offset_y) for q in quads]

    # Warmup
    backend.warmup()
    for _ in range(warmup_runs):
        backend.detect_quadrants(quad_inputs, task=task)

    # Benchmark
    results = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        dets = backend.detect_quadrants(quad_inputs, task=task)
        total_ms = (time.perf_counter() - t0) * 1000

        results.append(
            BenchmarkResult(
                backend=backend.info().name,
                total_ms=total_ms,
                per_quadrant_ms=[total_ms / len(quad_inputs)] * len(quad_inputs),
                detection_count=len(dets),
                image_path=image_path,
            )
        )

    return results


def save_benchmark_report(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """Save benchmark results as JSON."""
    data = [r.to_dict() for r in results]
    Path(output_path).write_text(json.dumps(data, indent=2) + "\n")
