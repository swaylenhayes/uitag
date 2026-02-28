"""Tests for detection merging and deduplication."""

from uitag.types import Detection
from uitag.merge import merge_detections, compute_iou


def test_compute_iou_identical():
    a = Detection("a", 0, 0, 100, 100, 1.0, "vision_text")
    b = Detection("b", 0, 0, 100, 100, 1.0, "florence2")
    assert compute_iou(a, b) == 1.0


def test_compute_iou_no_overlap():
    a = Detection("a", 0, 0, 50, 50, 1.0, "vision_text")
    b = Detection("b", 200, 200, 50, 50, 1.0, "florence2")
    assert compute_iou(a, b) == 0.0


def test_compute_iou_partial():
    a = Detection("a", 0, 0, 100, 100, 1.0, "vision_text")
    b = Detection("b", 50, 50, 100, 100, 1.0, "florence2")
    iou = compute_iou(a, b)
    assert 0.0 < iou < 1.0


def test_merge_removes_duplicates():
    dets = [
        Detection("Submit", 100, 100, 80, 30, 0.95, "vision_text"),
        Detection("button", 98, 98, 84, 34, 0.5, "florence2"),
        Detection("Cancel", 300, 100, 80, 30, 0.9, "vision_text"),
    ]
    merged = merge_detections(dets, iou_threshold=0.3)
    assert len(merged) == 2


def test_merge_assigns_som_ids():
    dets = [
        Detection("A", 0, 0, 50, 50, 0.9, "vision_text"),
        Detection("B", 200, 200, 50, 50, 0.8, "florence2"),
    ]
    merged = merge_detections(dets)
    ids = [d.som_id for d in merged]
    assert ids == [1, 2]


def test_merge_prefers_vision_text_over_florence():
    dets = [
        Detection("Submit", 100, 100, 80, 30, 0.95, "vision_text"),
        Detection("button", 100, 100, 80, 30, 0.5, "florence2"),
    ]
    merged = merge_detections(dets, iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0].label == "Submit"
    assert merged[0].source == "vision_text"
