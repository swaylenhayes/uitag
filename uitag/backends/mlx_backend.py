"""MLX-based Florence-2 detection backend."""

from __future__ import annotations

import os
import tempfile
import time

from PIL import Image

from uitag.backends.base import BackendInfo
from uitag.florence import _load_model, parse_location_tokens
from uitag.types import Detection

# Lazy-loaded at first use; module-level so tests can patch it.
generate = None


def _ensure_generate():
    global generate
    if generate is None:
        from mlx_vlm import generate as _gen

        generate = _gen


class MLXBackend:
    """Florence-2 detection using mlx_vlm.

    This is the default backend. Optimized to pre-save all quadrant
    images and reuse the loaded model across calls.
    """

    def __init__(self):
        self.last_timing: dict = {}

    def info(self) -> BackendInfo:
        try:
            import mlx_vlm

            version = getattr(mlx_vlm, "__version__", "unknown")
            available = True
        except ImportError:
            version = "not installed"
            available = False

        return BackendInfo(
            name="mlx",
            version=version,
            device="gpu",
            available=available,
        )

    def warmup(self) -> None:
        """Pre-load the Florence-2 model."""
        _load_model()

    def detect_quadrants(
        self,
        quadrants: list[tuple[Image.Image, int, int]],
        task: str = "<OD>",
        max_tokens: int = 512,
    ) -> list[Detection]:
        """Run Florence-2 on each quadrant with optimized I/O.

        Pre-saves all quadrant images to temp files, loads model once,
        then runs inference sequentially with minimal overhead.
        """
        if not quadrants:
            return []

        _ensure_generate()
        model, processor = _load_model()
        all_dets: list[Detection] = []
        per_quad_ms: list[float] = []

        # Pre-save all quadrants to temp files
        tmp_paths: list[tuple[str, int, int, int, int]] = []
        for image, offset_x, offset_y in quadrants:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.save(tmp_path)
            w, h = image.size
            tmp_paths.append((tmp_path, offset_x, offset_y, w, h))

        try:
            for tmp_path, offset_x, offset_y, img_w, img_h in tmp_paths:
                t0 = time.perf_counter()

                output = generate(
                    model,
                    processor,
                    task,
                    image=tmp_path,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    verbose=False,
                )

                per_quad_ms.append(round((time.perf_counter() - t0) * 1000, 1))

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
            for tmp_path, *_ in tmp_paths:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        self.last_timing = {
            "per_quadrant_ms": per_quad_ms,
            "total_ms": round(sum(per_quad_ms), 1),
        }

        return all_dets
