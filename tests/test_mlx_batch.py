"""Tests for MLX optimized quadrant inference."""

from unittest.mock import patch, MagicMock

from PIL import Image

from uitag.backends.mlx_backend import MLXBackend


def test_batch_detect_produces_same_results_as_sequential():
    """Optimized detection should produce equivalent results."""
    backend = MLXBackend()

    mock_output = MagicMock()
    mock_output.text = "button<loc_100><loc_200><loc_300><loc_400>"

    with (
        patch("uitag.backends.mlx_backend.generate") as mock_gen,
        patch("uitag.backends.mlx_backend._load_model") as mock_load,
    ):
        mock_gen.return_value = mock_output
        mock_load.return_value = (MagicMock(), MagicMock())

        imgs = [
            Image.new("RGB", (100, 100), color)
            for color in ["red", "green", "blue", "yellow"]
        ]
        quads = [
            (imgs[0], 0, 0),
            (imgs[1], 100, 0),
            (imgs[2], 0, 100),
            (imgs[3], 100, 100),
        ]
        dets = backend.detect_quadrants(quads)

    # 4 quadrants, each should produce 1 detection from the mocked output
    assert mock_gen.call_count == 4
    assert len(dets) == 4


def test_batch_detect_translates_coordinates():
    """Optimized detection should translate coordinates by quadrant offset."""
    backend = MLXBackend()

    # Mock output: button at (10, 20) in 100x100 space
    # loc values: x1=100, y1=200, x2=300, y2=400 in 999 space
    mock_output = MagicMock()
    mock_output.text = "icon<loc_100><loc_200><loc_300><loc_400>"

    with (
        patch("uitag.backends.mlx_backend.generate") as mock_gen,
        patch("uitag.backends.mlx_backend._load_model") as mock_load,
    ):
        mock_gen.return_value = mock_output
        mock_load.return_value = (MagicMock(), MagicMock())

        img = Image.new("RGB", (100, 100), "white")
        quads = [(img, 500, 300)]  # offset_x=500, offset_y=300
        dets = backend.detect_quadrants(quads)

    assert len(dets) == 1
    # The detection should have offset added to coordinates
    assert dets[0].x >= 500  # At least the offset
    assert dets[0].y >= 300


def test_batch_detect_records_per_quadrant_timing():
    """Backend should track per-quadrant inference times."""
    backend = MLXBackend()

    mock_output = MagicMock()
    mock_output.text = ""

    with (
        patch("uitag.backends.mlx_backend.generate") as mock_gen,
        patch("uitag.backends.mlx_backend._load_model") as mock_load,
    ):
        mock_gen.return_value = mock_output
        mock_load.return_value = (MagicMock(), MagicMock())

        img = Image.new("RGB", (100, 100), "white")
        quads = [(img, 0, 0), (img, 100, 0)]
        backend.detect_quadrants(quads)

    assert hasattr(backend, "last_timing")
    assert "per_quadrant_ms" in backend.last_timing
    assert len(backend.last_timing["per_quadrant_ms"]) == 2
    assert "total_ms" in backend.last_timing


def test_batch_detect_empty_input():
    """Backend should handle empty quadrant list."""
    backend = MLXBackend()
    dets = backend.detect_quadrants([])
    assert dets == []
