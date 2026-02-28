"""Tests for the CoreML-to-MLX encoder bridge."""

import numpy as np
import pytest


def test_coreml_to_mlx_nchw_conversion():
    """Bridge should convert NCHW numpy arrays to MLX sequence format."""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not available")

    from uitag.backends.encoder_bridge import coreml_to_mlx_embeddings

    # Simulate CoreML vision tower output: [1, 1024, 24, 24]
    fake_output = {"output": np.random.randn(1, 1024, 24, 24).astype(np.float32)}
    result = coreml_to_mlx_embeddings(fake_output)

    assert result.shape == (1, 576, 1024)  # H*W=576, C=1024
    assert result.dtype == mx.float32


def test_coreml_to_mlx_sequence_passthrough():
    """Bridge should pass through already-sequenced arrays."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from uitag.backends.encoder_bridge import coreml_to_mlx_embeddings

    # Already in sequence format: [1, 576, 1024]
    fake_output = {"embeddings": np.random.randn(1, 576, 1024).astype(np.float32)}
    result = coreml_to_mlx_embeddings(fake_output, output_key="embeddings")

    assert result.shape == (1, 576, 1024)


def test_coreml_to_mlx_auto_key():
    """Bridge should auto-detect the output key when not specified."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from uitag.backends.encoder_bridge import coreml_to_mlx_embeddings

    fake_output = {"layer_norm_52": np.random.randn(1, 1024, 24, 24).astype(np.float16)}
    result = coreml_to_mlx_embeddings(fake_output)

    assert result.shape == (1, 576, 1024)
