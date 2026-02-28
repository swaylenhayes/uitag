"""uitag — SoM detection pipeline using Apple Vision + Florence-2 on MLX."""

from uitag.run import run_pipeline
from uitag.types import Detection, PipelineResult

__all__ = ["Detection", "PipelineResult", "run_pipeline"]
