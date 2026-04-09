"""Shared data types for the detection pipeline."""

from dataclasses import dataclass, field


@dataclass
class Detection:
    """A single detected UI element."""

    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    source: (
        str  # "vision_text", "vision_text_block", "vision_rect", "florence2", "yolo"
    )
    som_id: int | None = None
    element_type: str | None = None


@dataclass
class PipelineResult:
    """Output of the full detection pipeline."""

    detections: list[Detection]
    image_width: int
    image_height: int
    timing_ms: dict = field(default_factory=dict)
