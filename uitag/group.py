"""Text block grouping — merges adjacent vision_text lines into paragraphs."""

from __future__ import annotations

from uitag.types import Detection


def group_text_blocks(
    detections: list[Detection],
    max_y_gap_factor: float = 1.0,
    x_align_tolerance: int = 20,
    containment_threshold: float = 0.85,
) -> tuple[list[Detection], int]:
    """Group adjacent vision_text lines into text blocks.

    Adjacent text lines are grouped when:
    - Vertical gap < max_y_gap_factor * last line's height
    - Left edge within x_align_tolerance of group's first line

    Groups of 2+ lines become a single detection with source
    ``"vision_text_block"`` and space-joined labels. Vision rectangles
    mostly contained within text blocks are absorbed.

    Returns:
        (updated_detections, groups_formed)
    """
    text_dets = [d for d in detections if d.source == "vision_text"]
    other_dets = [d for d in detections if d.source != "vision_text"]

    if len(text_dets) < 2:
        result = text_dets + other_dets
        result.sort(key=lambda d: (d.y, d.x))
        for i, det in enumerate(result):
            det.som_id = i + 1
        return result, 0

    # Sort text detections by vertical position
    text_dets.sort(key=lambda d: (d.y, d.x))

    # Build groups of adjacent, left-aligned lines
    groups: list[list[Detection]] = [[text_dets[0]]]

    for det in text_dets[1:]:
        last = groups[-1][-1]
        first = groups[-1][0]
        y_gap = det.y - (last.y + last.height)
        x_diff = abs(det.x - first.x)

        if y_gap < max_y_gap_factor * last.height and x_diff <= x_align_tolerance:
            groups[-1].append(det)
        else:
            groups.append([det])

    # Merge groups into text blocks
    merged_text: list[Detection] = []
    groups_formed = 0

    for group in groups:
        if len(group) == 1:
            merged_text.append(group[0])
            continue

        groups_formed += 1
        min_x = min(d.x for d in group)
        min_y = min(d.y for d in group)
        max_x2 = max(d.x + d.width for d in group)
        max_y2 = max(d.y + d.height for d in group)
        min_conf = min(d.confidence for d in group)
        label = " ".join(d.label for d in group)

        merged_text.append(
            Detection(
                label=label,
                x=min_x,
                y=min_y,
                width=max_x2 - min_x,
                height=max_y2 - min_y,
                confidence=min_conf,
                source="vision_text_block",
            )
        )

    # Absorb contained vision_rects
    text_blocks = [d for d in merged_text if d.source == "vision_text_block"]
    filtered_other: list[Detection] = []

    for det in other_dets:
        if det.source == "vision_rect" and text_blocks:
            if _is_contained_in_any(det, text_blocks, containment_threshold):
                continue
        filtered_other.append(det)

    # Combine, re-sort, re-assign SoM IDs
    result = merged_text + filtered_other
    result.sort(key=lambda d: (d.y, d.x))
    for i, det in enumerate(result):
        det.som_id = i + 1

    return result, groups_formed


def _is_contained_in_any(
    rect: Detection,
    blocks: list[Detection],
    threshold: float,
) -> bool:
    """Check if rect is mostly contained within any text block."""
    for block in blocks:
        ix1 = max(rect.x, block.x)
        iy1 = max(rect.y, block.y)
        ix2 = min(rect.x + rect.width, block.x + block.width)
        iy2 = min(rect.y + rect.height, block.y + block.height)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        rect_area = rect.width * rect.height
        if rect_area == 0:
            return True
        if inter_area / rect_area >= threshold:
            return True

    return False
