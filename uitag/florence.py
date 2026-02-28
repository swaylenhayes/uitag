"""Florence-2 detection wrapper using mlx_vlm."""

from __future__ import annotations

import os
import re
import tempfile
from typing import TYPE_CHECKING

from uitag.types import Detection

if TYPE_CHECKING:
    from PIL import Image

# ---------------------------------------------------------------------------
# Singleton model cache
# ---------------------------------------------------------------------------

_model = None
_processor = None

MODEL_ID = "mlx-community/Florence-2-base-ft-4bit"

# Pattern: label text followed by exactly four <loc_N> tokens
_LOC_PATTERN = re.compile(
    r"([^<]+)"  # label (non-< characters)
    r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
)


def _load_model():
    """Lazy-load model as singleton (load once, reuse across calls)."""
    global _model, _processor
    if _model is None:
        from mlx_vlm import load

        _model, _processor = load(MODEL_ID)
    return _model, _processor


# ---------------------------------------------------------------------------
# Location token parser
# ---------------------------------------------------------------------------


def parse_location_tokens(
    raw_text: str,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Parse Florence-2 raw output into bounding box dicts.

    Florence-2 outputs tokens like:
        <s>button<loc_100><loc_200><loc_300><loc_400>

    Location values are 0-999 (normalised). This function converts them
    to pixel coordinates and returns a list of dicts with keys:
    label, x, y, width, height.

    Degenerate output (repeated ``<s>`` tokens with no detections) returns
    an empty list.
    """
    # Strip leading <s> tokens
    text = re.sub(r"^(<s>\s*)+", "", raw_text)

    results: list[dict] = []
    for match in _LOC_PATTERN.finditer(text):
        label = match.group(1).strip()
        x1 = int(int(match.group(2)) * image_width / 999)
        y1 = int(int(match.group(3)) * image_height / 999)
        x2 = int(int(match.group(4)) * image_width / 999)
        y2 = int(int(match.group(5)) * image_height / 999)

        results.append(
            {
                "label": label,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Detection entry points
# ---------------------------------------------------------------------------


def detect_elements(
    image_path: str,
    task: str = "<OD>",
    max_tokens: int = 512,
) -> list[Detection]:
    """Run Florence-2 on a single image and return Detection objects.

    Parameters
    ----------
    image_path:
        Path to the image file on disk (mlx_vlm requires a file path).
    task:
        Florence-2 task token, e.g. ``"<OD>"`` for object detection.
    max_tokens:
        Maximum tokens to generate.

    Returns
    -------
    list[Detection]
        Detected elements with ``source="florence2"`` and
        ``confidence=0.5`` (Florence-2 does not emit per-box scores).
    """
    from mlx_vlm import generate
    from PIL import Image as PILImage

    model, processor = _load_model()

    # Get image dimensions for coordinate conversion
    with PILImage.open(image_path) as img:
        image_width, image_height = img.size

    output = generate(
        model,
        processor,
        task,
        image=image_path,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )

    # Handle both string output and objects with .text attribute
    raw_text = output.text if hasattr(output, "text") else str(output)

    parsed = parse_location_tokens(raw_text, image_width, image_height)

    return [
        Detection(
            label=d["label"],
            x=d["x"],
            y=d["y"],
            width=d["width"],
            height=d["height"],
            confidence=0.5,
            source="florence2",
        )
        for d in parsed
    ]


def detect_on_quadrant(
    quadrant_image: Image,
    offset_x: int,
    offset_y: int,
    task: str = "<OD>",
) -> list[Detection]:
    """Run detection on a PIL Image quadrant, translating coords back.

    Saves the PIL Image to a temporary file (mlx_vlm requires a path),
    runs ``detect_elements``, then shifts each detection's x/y by the
    given offsets so coordinates are in the full-image frame.

    Parameters
    ----------
    quadrant_image:
        A PIL Image representing a cropped quadrant.
    offset_x:
        Horizontal offset of this quadrant in the full image (pixels).
    offset_y:
        Vertical offset of this quadrant in the full image (pixels).
    task:
        Florence-2 task token.

    Returns
    -------
    list[Detection]
        Detections with coordinates in the full-image coordinate space.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        quadrant_image.save(tmp_path)

        detections = detect_elements(tmp_path, task=task)

        for d in detections:
            d.x += offset_x
            d.y += offset_y

        return detections
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
