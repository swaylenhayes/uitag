"""Florence-2 detection filtering.

Pipeline stage 4a — runs after merge (stage 4), before rescan (stage 4b).
Strips Florence-2 noise using two layers:
  1. Coverage filter: detections covering >5% of image area
  2. COCO blocklist: known object-class labels from COCO training
"""

from uitag.types import Detection

COCO_BLOCKLIST: frozenset[str] = frozenset(
    {
        # --- Observed in 261-image baseline (63 labels) ---
        "mobile phone",
        "poster",
        "tablet computer",
        "human face",
        "computer monitor",
        "person",
        "envelope",
        "window",
        "human hand",
        "footwear",
        "television",
        "swimming pool",
        "man",
        "book",
        "picture frame",
        "clothing",
        "woman",
        "laptop",
        "car",
        "building",
        "human nose",
        "land vehicle",
        "house",
        "animal",
        "dog",
        "cat",
        "tree",
        "flower",
        "food",
        "furniture",
        "chair",
        "table",
        "desk",
        "shelf",
        "bird",
        "horse",
        "drink",
        "human head",
        "christmas tree",
        "pumpkin",
        "stairs",
        "porch",
        "cake",
        "trousers",
        "rose",
        "sunflower",
        "wheel",
        "door",
        "home appliance",
        "watch",
        "pizza",
        "headphones",
        "bed",
        "studio couch",
        "digital clock",
        "goggles",
        "clock",
        "toy",
        "penguin",
        "marine mammal",
        "strawberry",
        "balloon",
        "boy",
        "ruler",
        "box",
        # --- Prophylactic COCO classes (not yet observed) ---
        "airplane",
        "backpack",
        "baseball bat",
        "bicycle",
        "boat",
        "bottle",
        "bowl",
        "bus",
        "cup",
        "glasses",
        "handbag",
        "hat",
        "kite",
        "motorcycle",
        "plate",
        "skateboard",
        "snowboard",
        "sports equipment",
        "suitcase",
        "surfboard",
        "tennis racket",
        "tie",
        "train",
        "truck",
        "umbrella",
        "vase",
        "calendar",
        "computer keyboard",
        "office supplies",
        "scoreboard",
        "vehicle",
        "whiteboard",
        "ball",
    }
)


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
