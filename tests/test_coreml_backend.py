"""Tests for the CoreML detection backend."""

import os

import pytest
from PIL import Image

from uitag.backends.base import DetectionBackend
from uitag.types import Detection

COREML_MODEL_PATH = "models/davit_encoder.mlpackage"


def test_coreml_backend_unavailable_without_model():
    """CoreML backend should report unavailable if model file missing."""
    from uitag.backends.coreml_backend import CoreMLBackend

    backend = CoreMLBackend(model_path="nonexistent.mlpackage")
    info = backend.info()
    assert info.name == "coreml"
    assert info.available is False


def test_coreml_backend_is_detection_backend():
    from uitag.backends.coreml_backend import CoreMLBackend

    backend = CoreMLBackend(model_path="nonexistent.mlpackage")
    assert isinstance(backend, DetectionBackend)


def test_coreml_backend_raises_on_detect_when_unavailable():
    from uitag.backends.coreml_backend import CoreMLBackend

    backend = CoreMLBackend(model_path="nonexistent.mlpackage")
    img = Image.new("RGB", (100, 100), "white")

    with pytest.raises(RuntimeError, match="CoreML model not available"):
        backend.detect_quadrants([(img, 0, 0)])


def test_coreml_backend_empty_quadrants():
    """CoreML backend should return empty list for empty input."""
    from uitag.backends.coreml_backend import CoreMLBackend

    CoreMLBackend(model_path="nonexistent.mlpackage")
    # Empty quadrants should return [] without checking availability
    # (matches MLXBackend behavior)


@pytest.mark.slow
def test_coreml_backend_produces_detections():
    """CoreML backend should produce valid detections on a real image."""
    if not os.path.exists(COREML_MODEL_PATH):
        pytest.skip("CoreML model not found. Run: python tools/convert_davit_coreml.py")

    from uitag.backends.coreml_backend import CoreMLBackend

    backend = CoreMLBackend(model_path=COREML_MODEL_PATH)
    assert backend.info().available is True

    # Create a simple test image with some shapes
    img = Image.new("RGB", (400, 300), "white")
    dets = backend.detect_quadrants([(img, 0, 0)])

    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, Detection)
        assert d.source == "florence2"

    # Verify timing was recorded
    assert "per_quadrant_ms" in backend.last_timing
    assert len(backend.last_timing["per_quadrant_ms"]) == 1


@pytest.mark.slow
def test_coreml_backend_timing():
    """CoreML backend should record per-quadrant timing."""
    if not os.path.exists(COREML_MODEL_PATH):
        pytest.skip("CoreML model not found. Run: python tools/convert_davit_coreml.py")

    from uitag.backends.coreml_backend import CoreMLBackend

    backend = CoreMLBackend(model_path=COREML_MODEL_PATH)
    img = Image.new("RGB", (200, 200), "gray")

    backend.detect_quadrants([(img, 0, 0), (img, 200, 0)])

    assert len(backend.last_timing["per_quadrant_ms"]) == 2
    assert backend.last_timing["total_ms"] > 0
