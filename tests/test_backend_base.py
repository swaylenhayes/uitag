"""Tests for the backend protocol."""

from uitag.backends.base import DetectionBackend, BackendInfo


def test_backend_info_fields():
    info = BackendInfo(name="test", version="1.0", device="gpu", available=True)
    assert info.name == "test"
    assert info.available is True


def test_backend_protocol_is_importable():
    """DetectionBackend should be a runtime-checkable protocol."""
    assert hasattr(DetectionBackend, "detect_quadrants")
    assert hasattr(DetectionBackend, "info")
    assert hasattr(DetectionBackend, "warmup")
