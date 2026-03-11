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
    (200, 170, 0),
    (255, 0, 150),
]


def _text_color(bg: tuple[int, int, int]) -> str:
    """Return 'black' or 'white' for best contrast against the background color."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return "black" if luminance > 130 else "white"


def render_som(
    image: Image.Image,
    detections: list[Detection],
    marker_size: int = 20,
) -> Image.Image:
    """Render Set-of-Mark numbered annotations on an image.

    Each detection gets a colored bounding box and a numbered circle marker
    positioned outside the top-left corner of the bounding box.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", marker_size - 4
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    r = marker_size // 2

    for det in detections:
        if det.som_id is None:
            continue

        color = SOM_COLORS[(det.som_id - 1) % len(SOM_COLORS)]

        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height
        if x1 > x2 or y1 > y2:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Position marker fully outside the bounding box (above-left of corner)
        cx = max(r, x1 - r - 1)
        cy = max(r, y1 - r - 1)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        label = str(det.som_id)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        text_color = _text_color(color)
        draw.text((cx - tw // 2, cy - th // 2 - 1), label, fill=text_color, font=font)

    return annotated
