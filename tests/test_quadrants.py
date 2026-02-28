"""Tests for image tiling (fixed quadrants and object-aware)."""

from PIL import Image
from uitag.quadrants import (
    split_quadrants,
    split_object_aware,
    _find_best_split,
    QuadrantInfo,
    SplitInfo,
)
from uitag.types import Detection


# --- Fixed quadrant tests (existing) ---


def test_split_returns_four_quadrants():
    img = Image.new("RGB", (1920, 1080), color="white")
    quads = split_quadrants(img)
    assert len(quads) == 4


def test_quadrant_dimensions():
    img = Image.new("RGB", (1920, 1080), color="white")
    quads = split_quadrants(img)
    for q in quads:
        assert isinstance(q, QuadrantInfo)
        assert q.image.size[0] == 960
        assert q.image.size[1] == 540


def test_quadrant_offsets():
    img = Image.new("RGB", (1920, 1080), color="white")
    quads = split_quadrants(img)
    offsets = [(q.offset_x, q.offset_y) for q in quads]
    assert (0, 0) in offsets
    assert (960, 0) in offsets
    assert (0, 540) in offsets
    assert (960, 540) in offsets


def test_quadrant_overlap():
    img = Image.new("RGB", (1920, 1080), color="white")
    quads = split_quadrants(img, overlap_px=50)
    for q in quads:
        assert q.image.size[0] >= 960
        assert q.image.size[1] >= 540


# --- _find_best_split tests ---


def _det(x: int, y: int, w: int, h: int) -> Detection:
    """Helper to create a Detection with only bbox fields."""
    return Detection(
        label="t", x=x, y=y, width=w, height=h, confidence=1.0, source="vision_text"
    )


def test_find_split_clean_gap():
    """When there's a clean gap at the midpoint, return midpoint as clean."""
    # Two elements, one on each side, gap around x=960
    dets = [_det(100, 0, 200, 50), _det(1200, 0, 200, 50)]
    pos, is_clean = _find_best_split(dets, "x", 1920)
    assert is_clean
    assert pos == 960  # Midpoint is clear


def test_find_split_shifts_to_avoid_element():
    """When an element straddles the midpoint, the split shifts to a gap."""
    # Element sits right across the midpoint: x=900..1020
    dets = [_det(900, 0, 120, 50)]
    pos, is_clean = _find_best_split(dets, "x", 1920)
    assert is_clean
    # Must be outside [900, 1020]
    assert pos <= 900 or pos >= 1020


def test_find_split_no_gap_falls_back():
    """When detections span the entire search range, fall back dirty."""
    # One huge detection covering the full width
    dets = [_det(0, 0, 1920, 50)]
    pos, is_clean = _find_best_split(dets, "x", 1920, search_range=200)
    assert not is_clean
    assert pos == 960  # Falls back to midpoint


def test_find_split_y_axis():
    """Split search works on the y axis too."""
    dets = [_det(0, 100, 50, 200), _det(0, 700, 50, 200)]
    pos, is_clean = _find_best_split(dets, "y", 1080)
    assert is_clean
    assert pos == 540  # Midpoint is clear


# --- split_object_aware tests ---


def test_object_aware_returns_four_tiles():
    img = Image.new("RGB", (1920, 1080), color="white")
    quads, info = split_object_aware(img, [])
    assert len(quads) == 4
    assert isinstance(info, SplitInfo)


def test_object_aware_avoids_element():
    """Split line should shift to avoid an element at the midpoint."""
    img = Image.new("RGB", (1920, 1080), color="white")
    # Wide element straddling the vertical midpoint (x=900..1040)
    dets = [_det(900, 200, 140, 50)]
    quads, info = split_object_aware(img, dets, overlap_px=0)

    # The split point should be outside [900, 1040]
    assert info.x_clean
    assert info.split_x <= 900 or info.split_x >= 1040


def test_object_aware_no_detections_uses_midpoint():
    """With no detections, split at the image midpoint (same as fixed)."""
    img = Image.new("RGB", (1920, 1080), color="white")
    quads, info = split_object_aware(img, [], overlap_px=0)

    assert info.split_x == 960
    assert info.split_y == 540
    assert info.x_clean
    assert info.y_clean

    # TL tile should be exactly the top-left quadrant
    tl = quads[0]
    assert tl.offset_x == 0
    assert tl.offset_y == 0
    assert tl.image.size[0] == 960
    assert tl.image.size[1] == 540


def test_object_aware_split_info_diagnostics():
    """SplitInfo should report shift from midpoint."""
    img = Image.new("RGB", (1920, 1080), color="white")
    dets = [_det(950, 200, 40, 50)]  # Straddles x midpoint
    _, info = split_object_aware(img, dets, overlap_px=0)
    assert info.midpoint_x == 960
    assert info.split_x != 960  # Must have shifted
