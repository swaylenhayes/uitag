#!/usr/bin/env python3
"""Use SomFlow as a Python library.

Prerequisites:
    pip install somflow
    macOS with Apple Silicon (Apple Vision + MLX)

Usage:
    python examples/use_as_library.py screenshot.png
"""

import sys
from pathlib import Path

from somflow import run_pipeline, Detection


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/use_as_library.py <screenshot.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    # Run the full detection pipeline
    result, annotated_image, manifest_json = run_pipeline(image_path)

    # Access detections directly
    print(f"Found {len(result.detections)} UI elements")
    print(f"Image: {result.image_width}x{result.image_height}")

    # Filter by source
    vision_text = [d for d in result.detections if d.source == "vision_text"]
    florence = [d for d in result.detections if d.source == "florence2"]
    print(f"  Vision text: {len(vision_text)}")
    print(f"  Florence-2:  {len(florence)}")

    # Access individual detections
    for det in result.detections[:5]:
        print(f"  [{det.som_id}] {det.label} at ({det.x},{det.y}) "
              f"{det.width}x{det.height} [{det.source}]")

    # Save the annotated image
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    annotated_image.save(out_dir / "annotated.png")
    print(f"\nSaved annotated image to {out_dir / 'annotated.png'}")

    # Save the manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(manifest_json)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
