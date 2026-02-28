"""Bridge between CoreML vision encoder output and MLX decoder input.

Handles format conversion between CoreML numpy output (NCHW) and MLX
array format (sequence) for the Florence-2 pipeline.

CoreML vision tower outputs: [1, 1024, 24, 24] (NCHW, float16)
MLX _encode_image expects: [1, 576, 1024] (batch, seq, features)
"""

from __future__ import annotations

import numpy as np


def coreml_to_mlx_embeddings(
    coreml_output: dict,
    output_key: str | None = None,
):
    """Convert CoreML encoder output to MLX array for the Florence-2 pipeline.

    Handles the NCHW → (batch, seq, features) reshape needed for
    model._encode_image(embeddings, extract_features=False).

    Args:
        coreml_output: Dictionary from CoreML model.predict().
        output_key: Key in the output dict. If None, uses the first key.

    Returns:
        MLX array of shape [1, 576, 1024] suitable for _encode_image().
    """
    import mlx.core as mx

    if output_key is None:
        output_key = next(iter(coreml_output))

    np_embeddings = coreml_output[output_key]

    # CoreML outputs float16 or float32 numpy arrays
    if np_embeddings.dtype == np.float64:
        np_embeddings = np_embeddings.astype(np.float32)

    # Handle NCHW format [1, C, H, W] → [1, H*W, C]
    if np_embeddings.ndim == 4:
        batch, channels, h, w = np_embeddings.shape
        # Reshape: [1, 1024, 24, 24] → [1, 1024, 576] → transpose → [1, 576, 1024]
        np_embeddings = np_embeddings.reshape(batch, channels, h * w)
        np_embeddings = np_embeddings.transpose(0, 2, 1)

    return mx.array(np_embeddings)
