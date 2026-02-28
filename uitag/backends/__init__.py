"""Detection backends -- MLX and CoreML."""

from uitag.backends.base import BackendInfo, DetectionBackend
from uitag.backends.selector import BackendPreference, select_backend

__all__ = ["BackendInfo", "BackendPreference", "DetectionBackend", "select_backend"]
