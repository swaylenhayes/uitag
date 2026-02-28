"""Backend auto-selection with fallback."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from uitag.backends.base import DetectionBackend

logger = logging.getLogger(__name__)

DEFAULT_COREML_MODEL = (
    Path(__file__).parent.parent.parent / "models" / "davit_encoder.mlpackage"
)


class BackendPreference(Enum):
    """User preference for which backend to use."""

    AUTO = "auto"  # Default to MLX; CoreML when GPU contended
    COREML = "coreml"  # CoreML preferred (falls back to MLX if unavailable)
    MLX = "mlx"  # MLX only (always available)


def _coreml_available(model_path: Path = DEFAULT_COREML_MODEL) -> bool:
    """Check if CoreML backend is available."""
    if not model_path.exists():
        return False
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


def select_backend(
    preference: BackendPreference = BackendPreference.AUTO,
    coreml_model_path: Path | str | None = None,
) -> DetectionBackend:
    """Select the best available detection backend.

    AUTO defaults to MLX (faster on idle GPU). Use COREML explicitly
    when GPU is contended by other workloads (e.g. embedding/ingestion).

    Args:
        preference: Which backend to prefer.
        coreml_model_path: Path to CoreML model (if not default location).

    Returns:
        A DetectionBackend instance ready for use.
    """
    model_path = Path(coreml_model_path) if coreml_model_path else DEFAULT_COREML_MODEL

    if preference == BackendPreference.COREML:
        if _coreml_available(model_path):
            from uitag.backends.coreml_backend import CoreMLBackend

            logger.info("Using CoreML backend (ANE)")
            return CoreMLBackend(model_path=str(model_path))
        else:
            logger.warning("CoreML requested but unavailable, falling back to MLX")

    if preference == BackendPreference.AUTO:
        # AUTO defaults to MLX — faster on idle GPU (1.25x).
        # CoreML value is as GPU-offload when contended.
        logger.info("Auto-selecting MLX backend (default)")

    from uitag.backends.mlx_backend import MLXBackend

    logger.info("Using MLX backend (GPU)")
    return MLXBackend()
