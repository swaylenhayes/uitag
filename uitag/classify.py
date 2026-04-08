"""VLM classification of UI elements (Stage 4e)."""

import base64
import io
import json
import re
import sys
import time

import requests
from PIL import Image

from uitag.types import Detection
from uitag.vocab import Vocab

CLASSIFIABLE_SOURCES = {"vision_rect", "yolo"}


def classify_detections(
    detections: list[Detection],
    img: Image.Image,
    vocab: Vocab,
    vlm_url: str = "http://localhost:8000/v1",
    vlm_model: str | None = None,
    verbose: bool = False,
) -> tuple[list[Detection], dict]:
    """Classify non-text detections via VLM.

    Args:
        detections: All pipeline detections (only vision_rect + yolo are classified).
        img: Original screenshot image.
        vocab: Vocabulary defining types and prompt.
        vlm_url: Base URL for OpenAI-compatible VLM server.
        vlm_model: Model name override (None = auto-detect).
        verbose: Print per-element timing.

    Returns:
        (detections_with_element_type, stats_dict)
    """
    stats: dict = {
        "total": len(detections),
        "classifiable": 0,
        "classified": 0,
        "errors": 0,
        "skipped_reason": None,
    }

    # Check server connectivity
    if not _check_server(vlm_url):
        print(
            f"VLM server not reachable at {vlm_url} — skipping classification",
            file=sys.stderr,
        )
        stats["skipped_reason"] = "server_unreachable"
        return detections, stats

    # Resolve model name
    if vlm_model is None:
        vlm_model = _detect_model(vlm_url)

    # Filter to classifiable detections
    targets = [
        (i, d) for i, d in enumerate(detections) if d.source in CLASSIFIABLE_SOURCES
    ]
    stats["classifiable"] = len(targets)

    if not targets:
        return detections, stats

    prompt = vocab.build_prompt()
    endpoint = f"{vlm_url}/chat/completions"

    print(
        f"Classifying {len(targets)} elements via VLM "
        f"({vocab.name}, {len(vocab.types)} types)...",
        file=sys.stderr,
    )
    t_start = time.perf_counter()

    for count, (idx, det) in enumerate(targets, 1):
        try:
            crop = _crop_detection(img, det, padding_pct=vocab.padding_pct)
            crop_b64 = _image_to_base64(crop)

            payload = {
                "model": vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{crop_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 50,
            }

            resp = requests.post(endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            element_type = _parse_element_type(content, vocab)
            detections[idx].element_type = element_type
            stats["classified"] += 1

        except Exception:
            detections[idx].element_type = vocab.fallback_type
            stats["errors"] += 1

        # Progress
        if count % 10 == 0 or count == len(targets):
            elapsed = time.perf_counter() - t_start
            print(f"  [{count}/{len(targets)}] {elapsed:.1f}s elapsed", file=sys.stderr)

    return detections, stats


def _check_server(vlm_url: str) -> bool:
    """Check if the VLM server is reachable."""
    try:
        resp = requests.get(f"{vlm_url}/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _detect_model(vlm_url: str) -> str:
    """Auto-detect the model name from the server."""
    try:
        resp = requests.get(f"{vlm_url}/models", timeout=5)
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"


def _crop_detection(
    img: Image.Image, det: Detection, padding_pct: int = 25
) -> Image.Image:
    """Crop a detection from the image with padding."""
    pad_x = int(det.width * padding_pct / 100)
    pad_y = int(det.height * padding_pct / 100)
    x1 = max(0, det.x - pad_x)
    y1 = max(0, det.y - pad_y)
    x2 = min(img.width, det.x + det.width + pad_x)
    y2 = min(img.height, det.y + det.height + pad_y)
    return img.crop((x1, y1, x2, y2))


def _image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_element_type(raw: str, vocab: Vocab) -> str:
    """Extract element_type from VLM response text.

    Parsing strategy:
    1. Try json.loads() on full response
    2. Regex for {"element_type": "..."} anywhere in text
    3. Fall back to vocab.fallback_type
    """
    # Strategy 1: direct JSON parse
    try:
        data = json.loads(raw)
        et = data.get("element_type", "")
        if et in vocab.types:
            return et
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Strategy 2: regex extraction
    match = re.search(r'"element_type"\s*:\s*"([^"]+)"', raw)
    if match:
        et = match.group(1)
        if et in vocab.types:
            return et

    # Strategy 3: fallback
    return vocab.fallback_type
