"""End-to-end integration tests for the detection pipeline."""

import json

import pytest
from PIL import Image

from uitag.run import run_pipeline


@pytest.mark.slow
def test_pipeline_returns_all_outputs(screenshot_path):
    """Full pipeline should return result, annotated image, and manifest."""
    result, annotated, manifest = run_pipeline(screenshot_path)

    # Check result
    assert result.image_width > 0
    assert result.image_height > 0
    assert len(result.detections) > 0
    assert "vision_ms" in result.timing_ms

    # Check annotated image
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (result.image_width, result.image_height)

    # Check manifest
    data = json.loads(manifest)
    assert data["element_count"] == len(result.detections)
    assert data["image_width"] == result.image_width


@pytest.mark.slow
def test_pipeline_on_simple_image(simple_image_path):
    """Pipeline should work on a simple test image."""
    result, annotated, manifest = run_pipeline(simple_image_path)
    assert len(result.detections) >= 0
    assert isinstance(annotated, Image.Image)


@pytest.mark.slow
def test_pipeline_accepts_backend_parameter(screenshot_path):
    """Pipeline should accept an optional backend parameter."""
    from uitag.backends.mlx_backend import MLXBackend

    backend = MLXBackend()
    result, annotated, manifest = run_pipeline(
        screenshot_path, backend=backend, no_florence=False
    )
    assert len(result.detections) > 0
    assert "florence_total_ms" in result.timing_ms
