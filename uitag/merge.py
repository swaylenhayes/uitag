"""Detection merging and deduplication."""

from uitag.types import Detection

SOURCE_PRIORITY = {
    "vision_text": 3,
    "vision_rect": 2,
    "florence2": 1,
}


def compute_iou(a: Detection, b: Detection) -> float:
    """Compute Intersection over Union between two detections."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = a.width * a.height
    b_area = b.width * b.height
    union_area = a_area + b_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def merge_detections(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Merge and deduplicate detections from multiple sources.

    When two detections overlap above IoU threshold, keep the higher-priority source.
    Sort by position (top-to-bottom, left-to-right) and assign SoM IDs.
    """
    if not detections:
        return []

    sorted_dets = sorted(
        detections,
        key=lambda d: (SOURCE_PRIORITY.get(d.source, 0), d.confidence),
        reverse=True,
    )

    kept: list[Detection] = []
    for det in sorted_dets:
        is_duplicate = False
        for existing in kept:
            if compute_iou(det, existing) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    kept.sort(key=lambda d: (d.y, d.x))
    for i, det in enumerate(kept):
        det.som_id = i + 1

    return kept
