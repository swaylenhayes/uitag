"""Tests for Florence-2 detection wrapper."""

import pytest
from uitag.florence import parse_location_tokens


def test_parse_location_tokens_single():
    raw = "<s>button<loc_100><loc_200><loc_300><loc_400>"
    results = parse_location_tokens(raw, image_width=1920, image_height=1080)
    assert len(results) == 1
    assert results[0]["label"] == "button"
    assert results[0]["x"] == int(100 * 1920 / 999)
    assert results[0]["y"] == int(200 * 1080 / 999)


def test_parse_location_tokens_multiple():
    raw = "<s>button<loc_0><loc_0><loc_500><loc_500>text field<loc_500><loc_500><loc_999><loc_999>"
    results = parse_location_tokens(raw, image_width=1000, image_height=1000)
    assert len(results) == 2
    assert results[0]["label"] == "button"
    assert results[1]["label"] == "text field"


def test_parse_location_tokens_empty():
    raw = "<s><s><s><s>"
    results = parse_location_tokens(raw, image_width=1920, image_height=1080)
    assert len(results) == 0


@pytest.mark.slow
def test_detect_elements_on_simple_image(simple_image_path):
    from uitag.florence import detect_elements
    from uitag.types import Detection

    detections = detect_elements(simple_image_path)
    assert isinstance(detections, list)
    for d in detections:
        assert isinstance(d, Detection)
        assert d.source == "florence2"
