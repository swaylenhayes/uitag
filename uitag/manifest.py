"""Text manifest generation for detected UI elements."""

import json

from uitag.types import PipelineResult


def generate_manifest(result: PipelineResult) -> str:
    """Generate a JSON manifest of all detected elements."""
    elements = []
    for det in result.detections:
        elem = {
            "som_id": det.som_id,
            "label": det.label,
            "bbox": {
                "x": det.x,
                "y": det.y,
                "width": det.width,
                "height": det.height,
            },
            "confidence": det.confidence,
            "source": det.source,
        }
        if det.element_type is not None:
            elem["element_type"] = det.element_type
        elements.append(elem)

    manifest = {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "element_count": len(elements),
        "elements": elements,
        "timing_ms": result.timing_ms,
    }

    return json.dumps(manifest, indent=2)
