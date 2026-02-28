"""Backend protocol for Florence-2 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from uitag.types import Detection


@dataclass
class BackendInfo:
    """Metadata about a detection backend."""

    name: str  # "mlx" or "coreml"
    version: str  # Backend library version
    device: str  # "gpu", "ane", "cpu"
    available: bool  # Whether this backend can run on this system


@runtime_checkable
class DetectionBackend(Protocol):
    """Protocol for Florence-2 detection backends.

    Each backend must be able to:
    - Report its availability and metadata
    - Run detection on a list of quadrant images (batched)
    - Optionally warm up (load model, compile, etc.)
    """

    def info(self) -> BackendInfo:
        """Return backend metadata."""
        ...

    def warmup(self) -> None:
        """Pre-load model and warm up inference path. Idempotent."""
        ...

    def detect_quadrants(
        self,
        quadrants: list[tuple[Image.Image, int, int]],
        task: str = "<OD>",
        max_tokens: int = 512,
    ) -> list[Detection]:
        """Run detection on multiple quadrant images.

        Args:
            quadrants: List of (image, offset_x, offset_y) tuples.
            task: Florence-2 task token.
            max_tokens: Maximum generation tokens.

        Returns:
            All detections with coordinates translated to full-image space.
        """
        ...
