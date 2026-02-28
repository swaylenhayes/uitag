"""CoreML-based Florence-2 detection backend.

Uses CoreML for the DaViT vision encoder (runs on ANE) and MLX for
the decoder (runs on GPU). Follows Apple FastVLM architecture pattern.

The vision tower is converted to CoreML via tools/convert_davit_coreml.py.
Output: [1, 1024, 24, 24] NCHW features, bridged to [1, 576, 1024] for MLX.
The position embeddings, projector, and decoder remain on MLX.

Integration strategy: temporarily patch model.vision_tower with pre-computed
CoreML features, then run the standard mlx_vlm generate pipeline unchanged.
This reuses all tokenization, decoding, and stopping logic.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from PIL import Image

from uitag.backends.base import BackendInfo
from uitag.florence import parse_location_tokens
from uitag.types import Detection

# Lazy-loaded at first use.
_generate = None


def _ensure_generate():
    global _generate
    if _generate is None:
        from mlx_vlm import generate as gen

        _generate = gen


class CoreMLBackend:
    """Florence-2 detection using CoreML encoder + MLX decoder.

    The DaViT vision encoder runs on the Apple Neural Engine via CoreML,
    while the projector, position embeddings, and transformer decoder
    remain on GPU via MLX.

    Args:
        model_path: Path to the converted .mlpackage file.
    """

    def __init__(self, model_path: str = "models/davit_encoder.mlpackage"):
        self._model_path = Path(model_path)
        self._coreml_model = None
        self._mlx_model = None
        self._processor = None
        self.last_timing: dict = {}

    def info(self) -> BackendInfo:
        available = self._model_path.exists()

        try:
            import coremltools as ct

            version = ct.__version__
        except ImportError:
            version = "not installed"
            available = False

        return BackendInfo(
            name="coreml",
            version=version,
            device="ane",
            available=available,
        )

    def warmup(self) -> None:
        """Load CoreML encoder and MLX model."""
        if not self.info().available:
            return

        import coremltools as ct

        if self._coreml_model is None:
            self._coreml_model = ct.models.MLModel(
                str(self._model_path),
                compute_units=ct.ComputeUnit.ALL,
            )

        from uitag.florence import _load_model

        model, processor = _load_model()
        self._mlx_model = model
        self._processor = processor

    def detect_quadrants(
        self,
        quadrants: list[tuple[Image.Image, int, int]],
        task: str = "<OD>",
        max_tokens: int = 512,
    ) -> list[Detection]:
        """Run detection using CoreML encoder + MLX decoder.

        For each quadrant:
        1. Preprocess image to [1, 3, 768, 768] for CoreML
        2. Run CoreML encoder on ANE → [1, 1024, 24, 24]
        3. Bridge to MLX format → [1, 576, 1024]
        4. Patch model.vision_tower to return pre-computed features
        5. Run standard mlx_vlm generate (handles projection + decoding)
        6. Parse output tokens into detections
        """
        if not self.info().available:
            raise RuntimeError(
                f"CoreML model not available at {self._model_path}. "
                "Run: python tools/convert_davit_coreml.py"
            )

        if not quadrants:
            return []

        self.warmup()
        _ensure_generate()

        import numpy as np

        from uitag.backends.encoder_bridge import coreml_to_mlx_embeddings

        all_dets: list[Detection] = []
        per_quad_ms: list[float] = []

        # Pre-save quadrant images to temp files (processor needs file paths)
        tmp_paths: list[tuple[str, int, int, int, int]] = []
        for image, offset_x, offset_y in quadrants:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.save(tmp_path)
            w, h = image.size
            tmp_paths.append((tmp_path, offset_x, offset_y, w, h))

        # Save original vision tower for restoration after each quadrant
        original_vt = self._mlx_model.vision_tower

        try:
            for i, (tmp_path, offset_x, offset_y, img_w, img_h) in enumerate(tmp_paths):
                t0 = time.perf_counter()

                # Step 1: Preprocess for CoreML (768x768, CHW, normalized)
                pil_img = quadrants[i][0]
                resized = pil_img.resize((768, 768), Image.LANCZOS)
                pixel_values = np.array(resized, dtype=np.float32)
                pixel_values = pixel_values.transpose(2, 0, 1)  # HWC → CHW
                pixel_values = pixel_values[np.newaxis]  # [1, 3, 768, 768]
                pixel_values = pixel_values / 255.0

                # Step 2: Run CoreML encoder (ANE)
                coreml_out = self._coreml_model.predict({"pixel_values": pixel_values})

                # Step 3: Bridge NCHW [1,1024,24,24] → seq [1,576,1024]
                vt_features = coreml_to_mlx_embeddings(coreml_out)

                # Step 4: Patch vision tower to return CoreML features.
                # The generate pipeline calls model.vision_tower(pixel_values)
                # inside _encode_image. Our patch returns pre-computed features,
                # then _encode_image applies position embeddings + projection.
                self._mlx_model.vision_tower = lambda _pv, _f=vt_features: _f

                # Step 5: Run standard generate pipeline
                output = _generate(
                    self._mlx_model,
                    self._processor,
                    task,
                    image=tmp_path,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    verbose=False,
                )

                per_quad_ms.append(round((time.perf_counter() - t0) * 1000, 1))

                # Step 6: Parse detections
                raw_text = output.text if hasattr(output, "text") else str(output)
                parsed = parse_location_tokens(raw_text, img_w, img_h)

                for d in parsed:
                    all_dets.append(
                        Detection(
                            label=d["label"],
                            x=d["x"] + offset_x,
                            y=d["y"] + offset_y,
                            width=d["width"],
                            height=d["height"],
                            confidence=0.5,
                            source="florence2",
                        )
                    )

        finally:
            # Always restore original vision tower
            self._mlx_model.vision_tower = original_vt
            # Clean up temp files
            for tmp_path, *_ in tmp_paths:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        self.last_timing = {
            "per_quadrant_ms": per_quad_ms,
            "total_ms": round(sum(per_quad_ms), 1),
        }

        return all_dets
