"""Tests for SoM annotation renderer."""

from PIL import Image
from uitag.types import Detection
from uitag.annotate import render_som


def test_render_som_returns_image():
    img = Image.new("RGB", (400, 300), color="white")
    dets = [
        Detection("Submit", 50, 50, 80, 30, 0.9, "vision_text", som_id=1),
        Detection("Cancel", 200, 50, 80, 30, 0.8, "vision_text", som_id=2),
    ]
    result = render_som(img, dets)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_render_som_preserves_dimensions():
    img = Image.new("RGB", (1920, 1080), color="black")
    dets = [Detection("A", 100, 100, 50, 50, 0.9, "vision_text", som_id=1)]
    result = render_som(img, dets)
    assert result.size == (1920, 1080)


def test_render_som_empty_detections():
    img = Image.new("RGB", (400, 300), color="white")
    result = render_som(img, [])
    assert isinstance(result, Image.Image)
