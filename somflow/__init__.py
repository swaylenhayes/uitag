"""SomFlow — SoM detection pipeline using Apple Vision + Florence-2 on MLX."""

from somflow.run import run_pipeline
from somflow.types import Detection, PipelineResult

__all__ = ["Detection", "PipelineResult", "run_pipeline"]
