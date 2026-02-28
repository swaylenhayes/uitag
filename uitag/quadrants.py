"""Image tiling for Florence-2 detection.

Provides both a simple fixed 4-quadrant split and an object-aware split
that uses Apple Vision detections to find cut lines that avoid bisecting
UI elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

from uitag.types import Detection


@dataclass
class QuadrantInfo:
    """A tile with its position in the original image."""

    image: Image.Image
    offset_x: int
    offset_y: int
    index: int


def split_quadrants(img: Image.Image, overlap_px: int = 0) -> list[QuadrantInfo]:
    """Split an image into 4 quadrants with optional overlap."""
    w, h = img.size
    mid_x = w // 2
    mid_y = h // 2

    regions = [
        (0, 0, min(mid_x + overlap_px, w), min(mid_y + overlap_px, h)),
        (max(mid_x - overlap_px, 0), 0, w, min(mid_y + overlap_px, h)),
        (0, max(mid_y - overlap_px, 0), min(mid_x + overlap_px, w), h),
        (max(mid_x - overlap_px, 0), max(mid_y - overlap_px, 0), w, h),
    ]

    offsets = [
        (0, 0),
        (max(mid_x - overlap_px, 0), 0),
        (0, max(mid_y - overlap_px, 0)),
        (max(mid_x - overlap_px, 0), max(mid_y - overlap_px, 0)),
    ]

    quads = []
    for i, (box, (ox, oy)) in enumerate(zip(regions, offsets)):
        tile = img.crop(box)
        quads.append(QuadrantInfo(image=tile, offset_x=ox, offset_y=oy, index=i))

    return quads


# ---------------------------------------------------------------------------
# Object-aware tiling
# ---------------------------------------------------------------------------


def _find_best_split(
    detections: list[Detection],
    axis: str,
    size: int,
    search_range: int = 200,
) -> tuple[int, bool]:
    """Find the best split position along *axis* that avoids cutting elements.

    Searches outward from the midpoint within *search_range* pixels.

    Args:
        detections: Bounding boxes to avoid splitting.
        axis: ``"x"`` for a vertical cut, ``"y"`` for a horizontal cut.
        size: Image dimension along *axis* (width or height).
        search_range: Max pixels to search from midpoint in each direction.

    Returns:
        ``(position, is_clean)`` — the chosen split coordinate and whether
        it avoids all bounding boxes.
    """
    mid = size // 2

    def _crosses(pos: int) -> bool:
        """True if a cut at *pos* would bisect any detection."""
        for d in detections:
            if axis == "x":
                lo, hi = d.x, d.x + d.width
            else:
                lo, hi = d.y, d.y + d.height
            if lo < pos < hi:
                return True
        return False

    # Search outward from midpoint: mid, mid+1, mid-1, mid+2, mid-2, ...
    for offset in range(0, search_range + 1):
        for candidate in (mid + offset, mid - offset):
            if candidate < 1 or candidate >= size - 1:
                continue
            if not _crosses(candidate):
                return candidate, True

    # No clean gap found — fall back to midpoint
    return mid, False


@dataclass
class SplitInfo:
    """Diagnostic info about where tiles were split."""

    split_x: int
    split_y: int
    x_clean: bool
    y_clean: bool
    midpoint_x: int
    midpoint_y: int


def split_object_aware(
    img: Image.Image,
    detections: list[Detection],
    overlap_px: int = 50,
    search_range: int = 200,
) -> tuple[list[QuadrantInfo], SplitInfo]:
    """Split image into tiles using detection bounding boxes to guide cut lines.

    Finds vertical and horizontal cut positions near the midpoint that avoid
    bisecting any detected element.  Falls back to the midpoint (with overlap
    padding) if no clean gap is found within *search_range*.

    Args:
        img: Source image.
        detections: Apple Vision detections used to guide split placement.
        overlap_px: Pixels of overlap padding added around each cut line.
        search_range: How far from the midpoint to search for clean gaps.

    Returns:
        Tuple of (tiles, split_info) where tiles is a list of 4
        ``QuadrantInfo`` (TL, TR, BL, BR) and split_info has diagnostics.
    """
    w, h = img.size

    split_x, x_clean = _find_best_split(detections, "x", w, search_range)
    split_y, y_clean = _find_best_split(detections, "y", h, search_range)
    info = SplitInfo(
        split_x=split_x,
        split_y=split_y,
        x_clean=x_clean,
        y_clean=y_clean,
        midpoint_x=w // 2,
        midpoint_y=h // 2,
    )

    # When the cut is clean, we can use a smaller overlap (just for context).
    # When dirty, keep the full overlap so Florence-2 sees more of the split element.
    ovl_x = overlap_px // 2 if x_clean else overlap_px
    ovl_y = overlap_px // 2 if y_clean else overlap_px

    regions = [
        # TL
        (0, 0, min(split_x + ovl_x, w), min(split_y + ovl_y, h)),
        # TR
        (max(split_x - ovl_x, 0), 0, w, min(split_y + ovl_y, h)),
        # BL
        (0, max(split_y - ovl_y, 0), min(split_x + ovl_x, w), h),
        # BR
        (max(split_x - ovl_x, 0), max(split_y - ovl_y, 0), w, h),
    ]

    offsets = [
        (0, 0),
        (max(split_x - ovl_x, 0), 0),
        (0, max(split_y - ovl_y, 0)),
        (max(split_x - ovl_x, 0), max(split_y - ovl_y, 0)),
    ]

    quads = []
    for i, (box, (ox, oy)) in enumerate(zip(regions, offsets)):
        tile = img.crop(box)
        quads.append(QuadrantInfo(image=tile, offset_x=ox, offset_y=oy, index=i))

    return quads, info
