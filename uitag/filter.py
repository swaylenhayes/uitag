"""Florence-2 detection filtering.

Pipeline stage 4a — runs after merge (stage 4), before rescan (stage 4b).
Strips Florence-2 noise using two layers:
  1. Coverage filter: detections covering >5% of image area
  2. COCO blocklist: known object-class labels from COCO training
"""

from uitag.types import Detection

# Populated in Task 2
COCO_BLOCKLIST: frozenset[str] = frozenset()


def filter_florence2(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    coverage_threshold: float = 0.05,
) -> tuple[list[Detection], dict]:
    """Filter Florence-2 noise from merged detections.

    Non-florence2 detections pass through untouched.

    Returns:
        (filtered_detections, filter_stats) where filter_stats contains
        florence2_total, florence2_coverage_filtered, florence2_blocklist_filtered,
        florence2_kept, and florence2_labels_kept.
    """
    image_area = image_width * image_height
    stats = {
        "florence2_total": 0,
        "florence2_coverage_filtered": 0,
        "florence2_blocklist_filtered": 0,
        "florence2_kept": 0,
        "florence2_labels_kept": [],
    }

    if image_area == 0:
        return detections, stats

    kept: list[Detection] = []
    for det in detections:
        if det.source != "florence2":
            kept.append(det)
            continue

        stats["florence2_total"] += 1

        # Layer 1: Coverage filter
        det_area = det.width * det.height
        if det_area / image_area > coverage_threshold:
            stats["florence2_coverage_filtered"] += 1
            continue

        # Layer 2: COCO blocklist
        if det.label.lower() in COCO_BLOCKLIST:
            stats["florence2_blocklist_filtered"] += 1
            continue

        # Survived both filters
        kept.append(det)
        stats["florence2_kept"] += 1
        stats["florence2_labels_kept"].append(det.label)

    return kept, stats
