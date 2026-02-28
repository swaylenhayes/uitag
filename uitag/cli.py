#!/usr/bin/env python3
"""uitag — CLI entry point.

Usage:
    uitag <image-path> [--output-dir <dir>]
    python detect.py <image-path> [--output-dir <dir>]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from uitag.run import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="uitag Detection Pipeline")
    parser.add_argument("image", help="Path to screenshot")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--task", default="<OD>", help="Florence-2 task token")
    parser.add_argument(
        "--overlap", type=int, default=50, help="Quadrant overlap pixels"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU dedup threshold")
    parser.add_argument(
        "--fast", action="store_true", help="Use fast OCR (less accurate, ~2-3x faster)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "coreml", "mlx"],
        default="auto",
        help="Detection backend: auto (default, uses MLX), coreml (ANE offload), mlx",
    )
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from uitag.backends.selector import BackendPreference, select_backend

    preference = BackendPreference(args.backend)
    backend = select_backend(preference=preference)
    print(f"Running pipeline on: {Path(image_path).name}")
    print(f"Backend: {backend.info().name} ({backend.info().device})")
    t0 = time.perf_counter()

    result, annotated, manifest = run_pipeline(
        image_path,
        florence_task=args.task,
        overlap_px=args.overlap,
        iou_threshold=args.iou,
        recognition_level="fast" if args.fast else "accurate",
        backend=backend,
    )

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"Total: {total_ms:.0f}ms")
    print(f"Detections: {len(result.detections)}")
    print(f"Timing: {json.dumps(result.timing_ms)}")

    # Save outputs
    stem = Path(image_path).stem
    annotated_path = out_dir / f"{stem}-som.png"
    manifest_path = out_dir / f"{stem}-manifest.json"

    annotated.save(annotated_path)
    with open(manifest_path, "w") as f:
        f.write(manifest)

    print("\nSaved:")
    print(f"  Annotated: {annotated_path}")
    print(f"  Manifest:  {manifest_path}")

    # Print element summary
    print("\nElements:")
    for d in result.detections:
        print(
            f"  [{d.som_id}] {d.label} at ({d.x},{d.y}) {d.width}x{d.height} [{d.source}]"
        )


if __name__ == "__main__":
    main()
