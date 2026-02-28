"""Tests for backend auto-selection and fallback."""

from unittest.mock import patch

from uitag.backends.selector import BackendPreference, select_backend


def test_select_backend_auto_defaults_to_mlx():
    """AUTO preference should default to MLX (faster on idle GPU)."""
    backend = select_backend()
    assert backend.info().name == "mlx"


def test_select_backend_explicit_mlx():
    """Explicit MLX preference should return MLX backend."""
    backend = select_backend(preference=BackendPreference.MLX)
    assert backend.info().name == "mlx"


def test_select_backend_coreml_falls_back_to_mlx():
    """If CoreML is preferred but unavailable, fall back to MLX."""
    with patch("uitag.backends.selector._coreml_available", return_value=False):
        backend = select_backend(preference=BackendPreference.COREML)
        assert backend.info().name == "mlx"


def test_select_backend_coreml_when_available():
    """CoreML preference should use CoreML if model exists."""
    with patch("uitag.backends.selector._coreml_available", return_value=True):
        backend = select_backend(preference=BackendPreference.COREML)
        assert backend.info().name == "coreml"


def test_backend_preference_from_string():
    """BackendPreference should construct from CLI string values."""
    assert BackendPreference("auto") == BackendPreference.AUTO
    assert BackendPreference("coreml") == BackendPreference.COREML
    assert BackendPreference("mlx") == BackendPreference.MLX
