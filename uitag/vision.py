# SPDX-License-Identifier: MIT
"""Apple Vision text + rectangle detection via Swift subprocess."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from uitag.types import Detection

# Resolve tool paths. The Swift source ships inside the package (uitag/tools/)
# so pip installs work. The compiled binary may exist in the repo root (tools/)
# for dev installs — check both locations.
_PKG_TOOLS = Path(__file__).resolve().parent / "tools"
_REPO_TOOLS = Path(__file__).resolve().parent.parent / "tools"

_SWIFT_BINARY = (
    _PKG_TOOLS / "vision-detect"
    if (_PKG_TOOLS / "vision-detect").exists()
    else _REPO_TOOLS / "vision-detect"
)
_SWIFT_SOURCE = _PKG_TOOLS / "vision-detect.swift"


def run_vision_detect(
    image_path: str | Path,
    recognition_level: str = "accurate",
) -> tuple[list[Detection], dict]:
    """Run Apple Vision detection on an image.

    Args:
        image_path: Absolute or relative path to the input image.
        recognition_level: ``"accurate"`` (default) or ``"fast"``.

    Returns:
        Tuple of (detections, timing_info) where timing_info contains
        keys like ``vision_time_ms``, ``text_count``, ``rect_count``.

    Raises:
        FileNotFoundError: If the Swift tool or the image does not exist.
        RuntimeError: If the Swift subprocess fails.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Build command: prefer compiled binary, fall back to interpreter
    if _SWIFT_BINARY.exists():
        cmd = [str(_SWIFT_BINARY), str(image_path)]
    elif _SWIFT_SOURCE.exists():
        cmd = ["swift", str(_SWIFT_SOURCE), str(image_path)]
    else:
        raise FileNotFoundError(
            f"Swift tool not found at {_SWIFT_BINARY} or {_SWIFT_SOURCE}"
        )

    if recognition_level == "fast":
        cmd.append("--fast")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"vision-detect.swift failed (exit {result.returncode}): {result.stderr}"
        )

    # Parse JSON stdout
    payload = json.loads(result.stdout)
    detections: list[Detection] = []
    for d in payload.get("detections", []):
        detections.append(
            Detection(
                label=d["label"],
                x=d["x"],
                y=d["y"],
                width=d["width"],
                height=d["height"],
                confidence=d["confidence"],
                source=d["source"],
            )
        )

    # Parse stderr timing lines
    timing: dict = {
        "image_width": payload.get("image_width", 0),
        "image_height": payload.get("image_height", 0),
    }
    for line in result.stderr.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("mise "):
            key, _, value = line.partition("=")
            try:
                timing[key] = float(value) if "." in value else int(value)
            except ValueError:
                timing[key] = value

    return detections, timing
