"""Tests for text manifest generation."""

import json
from uitag.types import Detection, PipelineResult
from uitag.manifest import generate_manifest


def test_manifest_structure():
    result = PipelineResult(
        detections=[
            Detection("Submit", 100, 200, 80, 30, 0.95, "vision_text", som_id=1),
            Detection("rectangle", 0, 0, 400, 50, 0.8, "vision_rect", som_id=2),
        ],
        image_width=1920,
        image_height=1080,
        timing_ms={"vision_time_ms": 4.2, "florence_time_ms": 530.0},
    )

    manifest = generate_manifest(result)
    data = json.loads(manifest)

    assert data["image_width"] == 1920
    assert data["image_height"] == 1080
    assert len(data["elements"]) == 2
    assert data["elements"][0]["som_id"] == 1
    assert data["elements"][0]["label"] == "Submit"
    assert data["elements"][0]["bbox"] == {
        "x": 100,
        "y": 200,
        "width": 80,
        "height": 30,
    }
    assert "timing_ms" in data


def test_manifest_empty():
    result = PipelineResult(detections=[], image_width=1920, image_height=1080)
    manifest = generate_manifest(result)
    data = json.loads(manifest)
    assert data["elements"] == []
    assert data["element_count"] == 0


def test_manifest_includes_backend_info():
    """Manifest should include backend name when present in timing."""
    result = PipelineResult(
        detections=[],
        image_width=1920,
        image_height=1080,
        timing_ms={"florence_backend": "mlx", "florence_total_ms": 650.0},
    )
    manifest = generate_manifest(result)
    data = json.loads(manifest)
    assert data["timing_ms"]["florence_backend"] == "mlx"
