"""Tests for the MLX detection backend."""

from unittest.mock import patch, MagicMock

from PIL import Image

from uitag.backends.base import DetectionBackend
from uitag.backends.mlx_backend import MLXBackend


def test_mlx_backend_is_detection_backend():
    backend = MLXBackend()
    assert isinstance(backend, DetectionBackend)


def test_mlx_backend_info():
    backend = MLXBackend()
    info = backend.info()
    assert info.name == "mlx"
    assert info.available is True


def test_mlx_backend_detect_quadrants_delegates():
    """Backend should call inference for each quadrant."""
    backend = MLXBackend()

    mock_output = MagicMock()
    mock_output.text = "button<loc_100><loc_200><loc_300><loc_400>"

    # Create a small test image
    img = Image.new("RGB", (100, 100), "white")

    with (
        patch("uitag.backends.mlx_backend.generate") as mock_gen,
        patch("uitag.backends.mlx_backend._load_model") as mock_load,
    ):
        mock_gen.return_value = mock_output
        mock_load.return_value = (MagicMock(), MagicMock())

        quads = [(img, 0, 0), (img, 100, 0)]
        dets = backend.detect_quadrants(quads)

        assert mock_gen.call_count == 2
        assert len(dets) == 2  # 1 detection per quadrant
