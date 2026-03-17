"""End-to-end detection pipeline orchestrator."""

import time

from PIL import Image

from uitag.types import PipelineResult
from uitag.vision import run_vision_detect
from uitag.quadrants import split_object_aware
from uitag.merge import merge_detections
from uitag.correct import correct_detections
from uitag.group import group_text_blocks
from uitag.filter import filter_florence2
from uitag.annotate import render_som
from uitag.manifest import generate_manifest


def run_pipeline(
    image_path: str,
    florence_task: str = "<OD>",
    overlap_px: int = 50,
    iou_threshold: float = 0.5,
    recognition_level: str = "accurate",
    backend=None,
    rescan: bool = False,
    rescan_threshold: float = 0.8,
    rescan_ids: list[int] | None = None,
    no_florence: bool = True,
) -> tuple[PipelineResult, Image.Image, str]:
    """Run the full detection pipeline on a screenshot.

    Pipeline stages:
    1. Apple Vision (text + rectangles)
    2. Quadrant split (skipped if no_florence)
    3. Florence-2 on each quadrant via backend (skipped if no_florence)
    4. Merge + deduplicate
    4a. Florence-2 filter (coverage + COCO blocklist)
    4b. Rescan (optional)
    4c. OCR correction
    4d. Text block grouping
    5. Annotate SoM
    6. Generate manifest

    Args:
        backend: Optional DetectionBackend. If None, uses MLXBackend.
        no_florence: Skip Florence-2 entirely (stages 2, 3, 4a). Vision-only mode.
            Auto-set to False when a backend is explicitly provided.

    Returns:
        (PipelineResult, annotated_image, manifest_json)
    """
    # If caller explicitly provides a backend, they want Florence-2 to run
    if backend is not None and no_florence:
        no_florence = False

    timing = {}
    img = Image.open(image_path)
    w, h = img.size

    # Stage 1: Apple Vision
    t0 = time.perf_counter()
    vision_dets, vision_timing = run_vision_detect(
        image_path, recognition_level=recognition_level
    )
    timing["vision_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timing.update(vision_timing)

    if no_florence:
        florence_dets = []
        timing["florence_skipped"] = True
    else:
        # Stage 2: Object-aware tiling
        t0 = time.perf_counter()
        quads, split_info = split_object_aware(img, vision_dets, overlap_px=overlap_px)
        timing["tiling_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timing["split_x"] = split_info.split_x
        timing["split_y"] = split_info.split_y
        timing["split_x_clean"] = split_info.x_clean
        timing["split_y_clean"] = split_info.y_clean

        # Stage 3: Florence-2 via backend
        if backend is None:
            from uitag.backends.mlx_backend import MLXBackend

            backend = MLXBackend()

        quad_inputs = [(q.image, q.offset_x, q.offset_y) for q in quads]

        t0 = time.perf_counter()
        florence_dets = backend.detect_quadrants(quad_inputs, task=florence_task)
        timing["florence_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timing["florence_backend"] = backend.info().name

        # Capture per-quadrant timing if backend provides it
        if hasattr(backend, "last_timing"):
            timing["florence_per_quadrant_ms"] = backend.last_timing.get(
                "per_quadrant_ms", []
            )

    # Stage 4: Merge + deduplicate
    all_dets = vision_dets + florence_dets
    t0 = time.perf_counter()
    merged = merge_detections(all_dets, iou_threshold=iou_threshold)
    timing["merge_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 4a: Florence-2 filter
    if not no_florence:
        t0 = time.perf_counter()
        merged, filter_stats = filter_florence2(merged, w, h)
        timing["florence2_filter_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timing["florence2_total"] = filter_stats["florence2_total"]
        timing["florence2_filtered"] = (
            filter_stats["florence2_coverage_filtered"]
            + filter_stats["florence2_blocklist_filtered"]
        )
        timing["florence2_kept"] = filter_stats["florence2_kept"]
        timing["florence2_labels_kept"] = filter_stats["florence2_labels_kept"]

    # Stage 4b: Rescan (optional)
    if rescan:
        from uitag.rescan import rescan_low_confidence

        t0 = time.perf_counter()
        merged, rescan_stats = rescan_low_confidence(
            merged,
            img,
            threshold=rescan_threshold,
            som_ids=rescan_ids,
            return_stats=True,
        )
        timing["rescan_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timing["rescan_count"] = rescan_stats["rescanned"]
        timing["rescan_improved"] = rescan_stats["improved"]

    # Stage 4c: Deterministic label corrections
    t0 = time.perf_counter()
    merged, correction_count = correct_detections(merged)
    timing["correct_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timing["corrections"] = correction_count

    # Stage 4d: Text block grouping
    t0 = time.perf_counter()
    merged, groups_formed = group_text_blocks(merged)
    timing["group_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timing["groups_formed"] = groups_formed

    # Build result
    result = PipelineResult(
        detections=merged,
        image_width=w,
        image_height=h,
        timing_ms=timing,
    )

    # Stage 5: Annotate
    t0 = time.perf_counter()
    annotated = render_som(img, merged)
    timing["annotate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 6: Manifest
    t0 = time.perf_counter()
    manifest = generate_manifest(result)
    timing["manifest_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return result, annotated, manifest
