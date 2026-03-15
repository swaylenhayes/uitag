"""Tests for Florence-2 detection filtering."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from uitag.filter import filter_florence2
from uitag.types import Detection


def _make_det(label, x=100, y=100, w=80, h=30, conf=0.5, source="florence2", som_id=1):
    return Detection(label, x, y, w, h, conf, source, som_id=som_id)


# --- Coverage filter ---


def test_coverage_filter_strips_large_bbox():
    """Florence2 detection covering >5% of image area is stripped."""
    # 1920x1080 image = 2,073,600 px. 5% = 103,680 px.
    # Detection 400x300 = 120,000 px > 5%.
    dets = [_make_det("mobile phone", x=0, y=0, w=400, h=300)]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 0
    assert stats["florence2_coverage_filtered"] == 1


def test_coverage_filter_keeps_small_bbox():
    """Florence2 detection covering <5% of image area survives coverage filter."""
    # 50x40 = 2,000 px. 2000/2073600 = 0.1% — well below 5%.
    dets = [_make_det("button", x=100, y=200, w=50, h=40)]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 1
    assert stats["florence2_coverage_filtered"] == 0


def test_coverage_filter_boundary_at_threshold():
    """Detection at exactly 5% threshold survives (> not >=)."""
    # Image 1000x1000 = 1,000,000 px. 5% = 50,000. Detection exactly 50,000.
    # 250x200 = 50,000. Ratio = 0.05. Filter uses >, so this should survive.
    # Label "button" is not on the COCO blocklist so only coverage is tested.
    dets = [_make_det("button", x=0, y=0, w=250, h=200)]
    filtered, stats = filter_florence2(dets, image_width=1000, image_height=1000)
    assert len(filtered) == 1  # exactly at threshold, not above it
    assert stats["florence2_coverage_filtered"] == 0


# --- COCO blocklist ---


def test_blocklist_strips_known_coco_label():
    """Small florence2 detection with COCO label is stripped by blocklist."""
    # Small bbox (under coverage threshold) but COCO label.
    dets = [_make_det("human face", x=100, y=200, w=50, h=40)]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 0
    assert stats["florence2_blocklist_filtered"] == 1
    assert stats["florence2_coverage_filtered"] == 0


def test_blocklist_passes_unknown_label():
    """Small florence2 detection with non-COCO label survives both filters."""
    dets = [_make_det("toggle switch", x=100, y=200, w=50, h=40)]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 1
    assert filtered[0].label == "toggle switch"
    assert stats["florence2_kept"] == 1
    assert stats["florence2_labels_kept"] == ["toggle switch"]


def test_blocklist_case_insensitive():
    """Blocklist matching is case-insensitive."""
    dets = [_make_det("Mobile Phone", x=100, y=200, w=50, h=40)]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 0
    assert stats["florence2_blocklist_filtered"] == 1


# --- Non-florence2 passthrough ---


def test_non_florence2_detections_pass_through():
    """Vision detections are never filtered."""
    dets = [
        _make_det("Submit", source="vision_text", w=400, h=300),  # Large bbox
        _make_det("mobile phone", source="vision_rect", w=400, h=300),  # COCO label
    ]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 2
    assert stats["florence2_total"] == 0


# --- Stats accuracy ---


def test_stats_dict_accurate():
    """Stats dict correctly counts each filter layer."""
    dets = [
        _make_det("poster", x=0, y=0, w=500, h=500),  # Large + COCO → coverage
        _make_det("human face", x=100, y=200, w=50, h=40),  # Small + COCO → blocklist
        _make_det("toggle", x=300, y=400, w=30, h=20),  # Small + unknown → kept
        _make_det("Submit", source="vision_text"),  # Non-florence2 → passthrough
    ]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert stats["florence2_total"] == 3
    assert stats["florence2_coverage_filtered"] == 1
    assert stats["florence2_blocklist_filtered"] == 1
    assert stats["florence2_kept"] == 1
    assert stats["florence2_labels_kept"] == ["toggle"]
    assert len(filtered) == 2  # vision_text + toggle


# --- Zero florence2 input ---


def test_zero_florence2_is_noop():
    """No florence2 detections → stats show all zeros, all detections kept."""
    dets = [
        _make_det("File", source="vision_text"),
        _make_det("rect", source="vision_rect"),
    ]
    filtered, stats = filter_florence2(dets, image_width=1920, image_height=1080)
    assert len(filtered) == 2
    assert stats["florence2_total"] == 0
    assert stats["florence2_kept"] == 0
    assert stats["florence2_labels_kept"] == []


# --- Integration: run_pipeline with no_florence ---


def test_no_florence_skips_florence_stages(tmp_path):
    """run_pipeline with no_florence=True skips tiling and florence."""
    from PIL import Image

    # Create a minimal test image
    img = Image.new("RGB", (200, 100), "white")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    # Mock Apple Vision to return a simple detection
    mock_dets = [_make_det("File", source="vision_text", som_id=None)]
    with patch("uitag.run.run_vision_detect") as mock_vision:
        mock_vision.return_value = (mock_dets, {})
        from uitag.run import run_pipeline

        result, annotated, manifest = run_pipeline(str(img_path), no_florence=True)

    # Florence-related keys absent, downstream stages still ran
    assert result.timing_ms.get("florence_skipped") is True
    assert "florence_total_ms" not in result.timing_ms
    assert "tiling_ms" not in result.timing_ms
    assert "florence2_filter_ms" not in result.timing_ms
    assert "correct_ms" in result.timing_ms  # downstream stages ran
    assert "group_ms" in result.timing_ms

    # Manifest should contain the vision detection
    manifest_data = json.loads(manifest)
    assert len(manifest_data["elements"]) == 1


def test_filter_stats_in_timing(tmp_path):
    """run_pipeline includes filter stats in timing_ms when florence runs."""
    from PIL import Image

    img = Image.new("RGB", (200, 100), "white")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    mock_vision_dets = [_make_det("File", source="vision_text", som_id=None)]
    mock_florence_dets = [_make_det("mobile phone", x=0, y=0, w=180, h=90)]

    mock_backend = MagicMock(spec=["detect_quadrants", "info"])
    mock_backend.detect_quadrants.return_value = mock_florence_dets
    # Use SimpleNamespace — MagicMock(name=) treats name as internal param,
    # and MagicMock attributes aren't JSON-serializable (manifest stage crashes).
    # spec= prevents auto-creation of last_timing, which would also be non-serializable.
    mock_backend.info.return_value = SimpleNamespace(name="mock", device="cpu")

    with patch("uitag.run.run_vision_detect") as mock_vision:
        mock_vision.return_value = (mock_vision_dets, {})
        from uitag.run import run_pipeline

        result, annotated, manifest = run_pipeline(str(img_path), backend=mock_backend)

    assert "florence2_filter_ms" in result.timing_ms
    assert result.timing_ms["florence2_total"] == 1
    assert result.timing_ms["florence2_filtered"] == 1
    assert result.timing_ms["florence2_kept"] == 0
