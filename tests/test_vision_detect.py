# SPDX-License-Identifier: MIT
"""Tests for Apple Vision detection (Swift subprocess)."""

import pytest

from uitag.vision import run_vision_detect


@pytest.mark.slow
def test_vision_detect_returns_detections(screenshot_path):
    """Vision detection returns a non-empty list of Detection objects."""
    detections, timing = run_vision_detect(screenshot_path)

    assert len(detections) > 0, "Expected at least one detection"

    for d in detections:
        assert isinstance(d.label, str)
        assert isinstance(d.x, int)
        assert isinstance(d.y, int)
        assert isinstance(d.width, int) and d.width > 0
        assert isinstance(d.height, int) and d.height > 0
        assert 0.0 <= d.confidence <= 1.0
        assert d.source in ("vision_text", "vision_rect")

    # Timing dict should have expected keys
    assert "vision_time_ms" in timing
    assert timing["vision_time_ms"] > 0


@pytest.mark.slow
def test_vision_detect_finds_text(screenshot_path):
    """At least one vision_text detection should be present."""
    detections, _ = run_vision_detect(screenshot_path)

    text_detections = [d for d in detections if d.source == "vision_text"]
    assert len(text_detections) > 0, "Expected at least one text detection"

    # Text detections should have non-empty labels
    for d in text_detections:
        assert len(d.label) > 0, "Text detection label should not be empty"


@pytest.mark.slow
def test_vision_detect_on_simple_image(simple_image_path):
    """Basic sanity check on a simple image."""
    detections, timing = run_vision_detect(simple_image_path)

    # Should succeed without error; may have zero or more detections
    assert isinstance(detections, list)
    assert "image_width" in timing
    assert "image_height" in timing
    assert timing["image_width"] > 0
    assert timing["image_height"] > 0
