#!/usr/bin/env python3
"""Quick A/B benchmark: MLX vs CoreML backends on real screenshot.

Usage: uv run python tools/quick_benchmark.py [image_path]
"""

import sys
import time

from PIL import Image

from uitag.quadrants import split_object_aware
from uitag.vision import run_vision_detect


def run_backend_bench(backend, quad_inputs, label, warmup=1, runs=2):
    """Run a backend with warmup + measured runs, print results."""
    print(f"\n{'=' * 50}")
    print(f"  {label} ({backend.info().device})")
    print(f"{'=' * 50}")

    # Warmup
    print(f"  Warming up ({warmup} run)...")
    backend.warmup()
    for _ in range(warmup):
        backend.detect_quadrants(quad_inputs)

    # Measured runs
    all_times = []
    all_counts = []
    for i in range(runs):
        t0 = time.perf_counter()
        dets = backend.detect_quadrants(quad_inputs)
        elapsed = (time.perf_counter() - t0) * 1000
        all_times.append(elapsed)
        all_counts.append(len(dets))

        per_q = backend.last_timing.get("per_quadrant_ms", [])
        print(
            f"  Run {i + 1}: {elapsed:.0f}ms total, {len(dets)} dets, "
            f"per-quad: {[f'{t:.0f}' for t in per_q]}"
        )

    avg = sum(all_times) / len(all_times)
    avg_count = sum(all_counts) / len(all_counts)
    print(f"  Average: {avg:.0f}ms, {avg_count:.0f} detections")
    return avg


def main():
    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "tests/fixtures/test-screen-shot-v2-1920x1080.png"
    )
    print(f"Image: {image_path}")

    # Prepare quadrants (shared between backends)
    print("Preparing quadrants...")
    img = Image.open(image_path)
    print(f"  Size: {img.size}")
    vision_dets, _ = run_vision_detect(image_path)
    print(f"  Vision detections: {len(vision_dets)}")
    quads, _ = split_object_aware(img, vision_dets, overlap_px=50)
    quad_inputs = [(q.image, q.offset_x, q.offset_y) for q in quads]
    print(f"  Quadrants: {len(quad_inputs)}")

    # MLX benchmark
    from uitag.backends.mlx_backend import MLXBackend

    mlx_backend = MLXBackend()
    mlx_avg = run_backend_bench(mlx_backend, quad_inputs, "MLX (GPU)")

    # CoreML benchmark (if available)
    import os

    coreml_model = "models/davit_encoder.mlpackage"
    if os.path.exists(coreml_model):
        from uitag.backends.coreml_backend import CoreMLBackend

        coreml_backend = CoreMLBackend(model_path=coreml_model)
        if coreml_backend.info().available:
            coreml_avg = run_backend_bench(
                coreml_backend, quad_inputs, "CoreML (ANE+GPU)"
            )

            # Comparison
            speedup = mlx_avg / coreml_avg if coreml_avg > 0 else 0
            print(f"\n{'=' * 50}")
            print("  COMPARISON")
            print(f"{'=' * 50}")
            print(f"  MLX:    {mlx_avg:.0f}ms")
            print(f"  CoreML: {coreml_avg:.0f}ms")
            print(f"  Speedup: {speedup:.2f}x")
            if speedup > 1:
                print(f"  CoreML is {speedup:.1f}x faster")
            else:
                print(f"  MLX is {1 / speedup:.1f}x faster")
        else:
            print("\nCoreML backend not available (coremltools missing?)")
    else:
        print(f"\nCoreML model not found at {coreml_model}")
        print("Run: uv run python tools/convert_davit_coreml.py")

    print()


if __name__ == "__main__":
    main()
