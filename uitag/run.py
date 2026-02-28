"""End-to-end detection pipeline orchestrator."""

import time

from PIL import Image

from uitag.types import PipelineResult
from uitag.vision import run_vision_detect
from uitag.quadrants import split_object_aware
from uitag.merge import merge_detections
from uitag.annotate import render_som
from uitag.manifest import generate_manifest


def run_pipeline(
    image_path: str,
    florence_task: str = "<OD>",
    overlap_px: int = 50,
    iou_threshold: float = 0.5,
    recognition_level: str = "accurate",
    backend=None,
) -> tuple[PipelineResult, Image.Image, str]:
    """Run the full detection pipeline on a screenshot.

    Pipeline stages:
    1. Apple Vision (text + rectangles)
    2. Quadrant split
    3. Florence-2 on each quadrant (via backend)
    4. Merge + deduplicate
    5. Annotate SoM
    6. Generate manifest

    Args:
        backend: Optional DetectionBackend. If None, uses MLXBackend.

    Returns:
        (PipelineResult, annotated_image, manifest_json)
    """
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

    # Stage 2: Object-aware tiling
    quads, split_info = split_object_aware(img, vision_dets, overlap_px=overlap_px)
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
    merged = merge_detections(all_dets, iou_threshold=iou_threshold)

    # Build result
    result = PipelineResult(
        detections=merged,
        image_width=w,
        image_height=h,
        timing_ms=timing,
    )

    # Stage 5: Annotate
    annotated = render_som(img, merged)

    # Stage 6: Manifest
    manifest = generate_manifest(result)

    return result, annotated, manifest
