"""Tests for VLM classification module."""

import json
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from uitag.types import Detection
from uitag.vocab import load_vocab
from uitag.classify import (
    classify_detections,
    _parse_element_type,
    _crop_detection,
    _check_server,
)


@pytest.fixture
def vocab():
    return load_vocab("leith-17")


@pytest.fixture
def sample_detections():
    """Mix of sources — only vision_rect and yolo should be classified."""
    return [
        Detection("File", 10, 5, 40, 20, 0.95, "vision_text", som_id=1),
        Detection("", 100, 50, 24, 24, 0.87, "vision_rect", som_id=2),
        Detection("", 200, 100, 30, 30, 0.72, "yolo", som_id=3),
        Detection("Open File", 10, 30, 100, 20, 0.90, "vision_text_block", som_id=4),
        Detection("", 300, 200, 16, 16, 0.65, "florence2", som_id=5),
    ]


class TestParseElementType:
    def test_clean_json(self, vocab):
        assert _parse_element_type('{"element_type": "icon"}', vocab) == "icon"

    def test_json_with_surrounding_text(self, vocab):
        raw = 'Here is my answer: {"element_type": "button"} hope that helps!'
        assert _parse_element_type(raw, vocab) == "button"

    def test_unparseable_returns_none(self, vocab):
        assert _parse_element_type("I think it's a button", vocab) is None

    def test_type_not_in_vocab_returns_none(self, vocab):
        assert _parse_element_type('{"element_type": "banana"}', vocab) is None

    def test_empty_response_returns_none(self, vocab):
        assert _parse_element_type("", vocab) is None


class TestCropDetection:
    def test_crop_with_padding(self):
        """Crop should expand by padding_pct and clamp to image bounds."""
        img = Image.new("RGB", (400, 300))
        det = Detection("", 100, 100, 50, 50, 0.9, "vision_rect", som_id=1)
        crop = _crop_detection(img, det, padding_pct=25)
        # Padding: 25% of 50 = 12px each side
        # Expected: (88, 88) to (162, 162) = 74x74
        assert crop.size[0] > 0
        assert crop.size[1] > 0

    def test_crop_clamps_to_image_bounds(self):
        """Crop near image edge should not exceed bounds."""
        img = Image.new("RGB", (100, 100))
        det = Detection("", 0, 0, 20, 20, 0.9, "vision_rect", som_id=1)
        crop = _crop_detection(img, det, padding_pct=50)
        assert crop.size[0] <= 100
        assert crop.size[1] <= 100


class TestClassifyDetections:
    def _mock_vlm_response(self, element_type):
        """Create a mock requests.Response for a VLM classification."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"element_type": element_type})}}
            ]
        }
        return resp

    @patch("uitag.classify._check_server", return_value=True)
    @patch("uitag.classify.requests.post")
    def test_only_rect_and_yolo_classified(
        self, mock_post, mock_check, sample_detections, vocab
    ):
        mock_post.return_value = self._mock_vlm_response("icon")
        img = Image.new("RGB", (400, 300))
        classified, stats = classify_detections(
            sample_detections, img, vocab, vlm_url="http://fake:8000/v1"
        )
        # vision_rect (som_id=2) and yolo (som_id=3) get classified
        typed = [d for d in classified if d.element_type is not None]
        assert len(typed) == 2
        assert classified[0].element_type is None  # vision_text
        assert classified[1].element_type == "icon"  # vision_rect
        assert classified[2].element_type == "icon"  # yolo
        assert classified[3].element_type is None  # vision_text_block

    @patch("uitag.classify._check_server", return_value=False)
    def test_server_unreachable_skips_classification(
        self, mock_check, sample_detections, vocab
    ):
        img = Image.new("RGB", (400, 300))
        classified, stats = classify_detections(
            sample_detections, img, vocab, vlm_url="http://fake:8000/v1"
        )
        assert all(d.element_type is None for d in classified)
        assert stats["skipped_reason"] == "server_unreachable"

    @patch("uitag.classify._check_server", return_value=True)
    @patch("uitag.classify.requests.post", side_effect=Exception("connection reset"))
    def test_request_failure_leaves_none(
        self, mock_post, mock_check, sample_detections, vocab
    ):
        img = Image.new("RGB", (400, 300))
        classified, stats = classify_detections(
            sample_detections, img, vocab, vlm_url="http://fake:8000/v1"
        )
        # Should not crash — failed elements stay untyped
        rect_det = classified[1]  # vision_rect
        assert rect_det.element_type is None
        assert stats["errors"] > 0


class TestCheckServer:
    @patch("uitag.classify.requests.get")
    def test_server_reachable(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert _check_server("http://localhost:8000/v1") is True

    @patch("uitag.classify.requests.get", side_effect=Exception("refused"))
    def test_server_unreachable(self, mock_get):
        assert _check_server("http://localhost:8000/v1") is False
