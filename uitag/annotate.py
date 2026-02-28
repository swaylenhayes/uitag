"""Set-of-Mark annotation renderer."""

from PIL import Image, ImageDraw, ImageFont

from uitag.types import Detection

SOM_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 100, 255),
    (255, 165, 0),
    (180, 0, 255),
    (0, 200, 200),
    (255, 255, 0),
    (255, 0, 150),
]


def render_som(
    image: Image.Image,
    detections: list[Detection],
    marker_size: int = 20,
) -> Image.Image:
    """Render Set-of-Mark numbered annotations on an image.

    Each detection gets a colored bounding box and a numbered circle marker.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", marker_size - 4
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for det in detections:
        if det.som_id is None:
            continue

        color = SOM_COLORS[(det.som_id - 1) % len(SOM_COLORS)]

        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        cx, cy = x1, y1
        r = marker_size // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        label = str(det.som_id)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2 - 1), label, fill="white", font=font)

    return annotated
