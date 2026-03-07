"""Multi-crop ensemble OCR rescan for low-confidence text detections.

Apple Vision's OCR of special characters is sensitive to crop boundaries.
Shifting the crop by just 5px can flip the result from correct to garbled.
This module tries multiple padding values and picks the best result using
a "most special characters" heuristic — raw, symbol-rich readings are more
likely correct than sanitized ones.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from PIL import Image

from uitag.types import Detection

# Padding values to try during ensemble rescan. 10px is empirically the
# best single value, but results oscillate unpredictably — the ensemble
# compensates by sampling multiple boundaries.
_ENSEMBLE_PADDINGS = (5, 10, 15, 20, 25)

_SPECIAL_CHARS = set(r"\[]{}();_+*?^$|")


def find_low_confidence(
    detections: list[Detection],
    threshold: float = 0.8,
) -> list[Detection]:
    """Return vision_text detections below the confidence threshold."""
    return [
        d for d in detections if d.source == "vision_text" and d.confidence < threshold
    ]


def _crop(
    image: Image.Image,
    det: Detection,
    padding: int = 10,
) -> Image.Image:
    """Crop a detection region with padding."""
    img_w, img_h = image.size
    x1 = max(0, det.x - padding)
    y1 = max(0, det.y - padding)
    x2 = min(img_w, det.x + det.width + padding)
    y2 = min(img_h, det.y + det.height + padding)
    return image.crop((x1, y1, x2, y2))


def _special_char_count(text: str) -> int:
    """Count special/punctuation characters — higher means more raw reading."""
    return sum(1 for c in text if c in _SPECIAL_CHARS)


def _ocr_crop(crop: Image.Image) -> tuple[str, float]:
    """Run Vision OCR on a single crop image. Returns (label, confidence)."""
    from uitag.vision import run_vision_detect

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        crop.save(tmp_path)

    try:
        dets, _ = run_vision_detect(
            tmp_path, recognition_level="accurate", use_lang_correction=False
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    text_dets = [d for d in dets if d.source == "vision_text"]
    if not text_dets:
        return "", 0.0

    best = max(text_dets, key=lambda d: d.confidence)
    return best.label, best.confidence


def _rescan_single(
    image: Image.Image,
    det: Detection,
    scale: int = 1,
) -> tuple[str, float]:
    """Re-OCR a single detection using multi-crop ensemble.

    Tries multiple padding values and selects the result with the most
    special characters (ties broken by confidence). This compensates for
    Apple Vision's chaotic sensitivity to crop boundaries on symbol text.

    Returns (label, confidence) from the best ensemble result.
    """
    candidates: list[tuple[str, float, int]] = []

    for padding in _ENSEMBLE_PADDINGS:
        crop = _crop(image, det, padding=padding)
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        label, conf = _ocr_crop(crop)
        if label:
            candidates.append((label, conf, _special_char_count(label)))

    if not candidates:
        return det.label, det.confidence

    # Pick the reading with the most special characters, then highest confidence
    candidates.sort(key=lambda c: (c[2], c[1]), reverse=True)
    return candidates[0][0], candidates[0][1]


def rescan_low_confidence(
    detections: list[Detection],
    image: Image.Image,
    threshold: float = 0.8,
    scale: int = 2,
    som_ids: list[int] | None = None,
    return_stats: bool = False,
) -> list[Detection] | tuple[list[Detection], dict]:
    """Re-scan low-confidence vision_text detections at higher resolution.

    Args:
        detections: All merged detections.
        image: Original full-size image.
        threshold: Confidence below this triggers rescan.
        scale: Upscale factor for crops (default 2x).
        som_ids: If provided, only rescan these specific som_ids.
        return_stats: If True, return (detections, stats_dict).

    Returns:
        Updated detection list (or tuple with stats if return_stats=True).
    """
    result = []
    rescanned_count = 0
    improved_count = 0

    for det in detections:
        # Determine if this detection should be rescanned
        should_rescan = det.source == "vision_text" and det.confidence < threshold
        if som_ids is not None and det.som_id not in som_ids:
            should_rescan = False

        if not should_rescan:
            result.append(det)
            continue

        rescanned_count += 1
        new_label, new_conf = _rescan_single(image, det, scale=scale)

        updated = copy.copy(det)
        # Only replace if confidence improved AND special chars weren't lost.
        # High-confidence crops can produce sanitized text that drops backslashes,
        # swaps characters (l→I, w→W), or introduces Cyrillic substitutions.
        orig_sc = _special_char_count(det.label)
        new_sc = _special_char_count(new_label)
        if new_conf > det.confidence and new_sc >= orig_sc:
            updated.label = new_label
            updated.confidence = new_conf
            improved_count += 1

        result.append(updated)

    if return_stats:
        stats = {
            "total": len(detections),
            "rescanned": rescanned_count,
            "improved": improved_count,
        }
        return result, stats

    return result
