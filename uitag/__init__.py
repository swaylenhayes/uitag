"""uitag — UI element detection using Apple Vision + fine-tuned YOLO."""

from uitag.run import run_pipeline
from uitag.types import Detection, PipelineResult

__version__ = "0.6.0"
__all__ = ["Detection", "PipelineResult", "run_pipeline"]
