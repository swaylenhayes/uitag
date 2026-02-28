#!/usr/bin/env python3
"""Write a custom detection backend for SomFlow.

This example shows how to implement the DetectionBackend protocol
to plug in your own inference engine (ONNX, TensorRT, a remote API, etc.)
without modifying any SomFlow internals.

Prerequisites:
    pip install somflow
    macOS with Apple Silicon (Apple Vision still runs Stage 1)

Usage:
    python examples/custom_backend.py screenshot.png
"""

import sys
from pathlib import Path

from PIL import Image

from somflow import run_pipeline, Detection
from somflow.backends.base import BackendInfo, DetectionBackend


class GridBackend:
    """Example backend that returns a fixed grid of detections.

    This is a minimal implementation showing the DetectionBackend protocol.
    Replace the detect_quadrants logic with your own model inference.
    """

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="grid",
            version="0.1.0",
            device="cpu",
            available=True,
        )

    def warmup(self) -> None:
        pass  # No model to load

    def detect_quadrants(
        self,
        quadrants: list[tuple[Image.Image, int, int]],
        task: str = "<OD>",
        max_tokens: int = 512,
    ) -> list[Detection]:
        """Return a grid of detections for each quadrant.

        In a real backend, you would run your model here.
        The quadrants list contains (image, offset_x, offset_y) tuples.
        You MUST translate coordinates by the offset before returning.
        """
        detections = []
        for img, offset_x, offset_y in quadrants:
            w, h = img.size
            # Create a 2x2 grid of detections per quadrant
            for row in range(2):
                for col in range(2):
                    detections.append(
                        Detection(
                            label=f"grid_{col}_{row}",
                            x=offset_x + col * (w // 2),
                            y=offset_y + row * (h // 2),
                            width=w // 4,
                            height=h // 4,
                            confidence=0.5,
                            source="florence2",  # Must be "florence2" for merge priority
                        )
                    )
        return detections


# Verify it satisfies the protocol
assert isinstance(GridBackend(), DetectionBackend), "GridBackend must implement DetectionBackend"


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/custom_backend.py <screenshot.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    # Use our custom backend instead of the default MLX backend
    backend = GridBackend()
    result, annotated_image, manifest_json = run_pipeline(
        image_path, backend=backend
    )

    print(f"Detections: {len(result.detections)} (grid + Apple Vision)")
    print(f"Backend: {result.timing_ms.get('florence_backend', 'unknown')}")

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    annotated_image.save(out_dir / "custom-backend.png")
    print(f"Saved to {out_dir / 'custom-backend.png'}")


if __name__ == "__main__":
    main()
