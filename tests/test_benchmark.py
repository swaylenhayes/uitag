"""Benchmark tests for detection backends."""

import pytest
from uitag.benchmark import BenchmarkResult


def test_benchmark_result_fields():
    r = BenchmarkResult(
        backend="mlx",
        total_ms=3200.0,
        per_quadrant_ms=[800.0, 810.0, 790.0, 800.0],
        detection_count=42,
        image_path="test.png",
    )
    assert r.backend == "mlx"
    assert r.mean_per_quadrant_ms == pytest.approx(800.0, abs=5.0)
    assert r.total_ms == 3200.0


def test_benchmark_result_mean_calculation():
    r = BenchmarkResult(
        backend="test",
        total_ms=100.0,
        per_quadrant_ms=[10.0, 20.0, 30.0, 40.0],
        detection_count=10,
        image_path="test.png",
    )
    assert r.mean_per_quadrant_ms == 25.0
