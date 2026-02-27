# SomFlow

[![Tests](https://github.com/swaylenhayes/somflow/actions/workflows/test.yml/badge.svg)](https://github.com/swaylenhayes/somflow/actions/workflows/test.yml)

A Set-of-Mark (SoM) detection pipeline for macOS that transforms screenshots into structured, annotated element maps. Built for Apple Silicon using Apple Vision Framework and Florence-2 on MLX.

![SomFlow output — 151 UI elements detected on a 1920x1080 screenshot](https://raw.githubusercontent.com/swaylenhayes/somflow/main/docs/examples/vscode-som.png)

*151 numbered elements detected in ~0.8s — text labels (Apple Vision), rectangles, icons, and buttons (Florence-2). [Full manifest JSON →](docs/examples/vscode-manifest.json)*

## Why This Exists

Vision language models under 10B parameters cannot reliably detect individual UI elements on complex professional screenshots. They collapse to a single full-screen bounding box. This was validated empirically across Florence-2, PTA-1, and other MIT-licensed detection models during a [benchmark of 14+ models](docs/research.md#model-selection).

SomFlow solves this by combining two complementary detection systems and preprocessing the image before any VLM ever sees it:

- **Apple Vision Framework** detects text labels and rectangular UI elements natively on the ANE — effectively free
- **Florence-2** detects non-text elements (icons, buttons, images) via open-vocabulary detection on tiled image quadrants

The result is a numbered element map (SoM annotation) and a structured JSON manifest that downstream agents can consume directly.

## Pipeline Architecture

```
Screenshot (1920x1080)
    |
    v
[1] Apple Vision (Swift binary)
    |  VNRecognizeTextRequest + VNDetectRectanglesRequest
    |  ~189ms (fast) / ~980ms (accurate)
    v
[2] Object-Aware Tiling
    |  Split into 4 quadrants, cut lines avoid bounding boxes
    v
[3] Florence-2 (mlx_vlm, per quadrant)
    |  <OD> detection on each tile, ~160ms/quadrant
    v
[4] Merge + Deduplicate
    |  IoU-based overlap removal, source priority ranking
    v
[5] SoM Annotation
    |  Numbered markers + colored bounding boxes
    v
[6] JSON Manifest
    |  Element list with coordinates, labels, sources, timing
    v
Output: annotated.png + manifest.json
```

End-to-end on a 1920x1080 VS Code screenshot (~151 UI elements detected):
- **~0.8s** with fast OCR (Florence-2 ~650ms + Vision ~189ms)
- **~1.6s** with accurate OCR (Florence-2 ~650ms + Vision ~980ms)

## Quick Start

```bash
# Clone
git clone https://github.com/swaylenhayes/somflow.git
cd somflow

# Install
uv pip install -e ".[dev]"

# Run on a screenshot
somflow screenshot.png --output-dir out/

# Or directly
python detect.py screenshot.png -o out/
```

### Output

Two files are produced:
- `screenshot-som.png` — the original image with numbered SoM annotations overlaid
- `screenshot-manifest.json` — structured element data:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "element_count": 151,
  "elements": [
    {
      "som_id": 1,
      "label": "File",
      "bbox": {"x": 48, "y": 0, "width": 26, "height": 16},
      "confidence": 1.0,
      "source": "vision_text"
    }
  ],
  "timing_ms": {
    "vision_ms": 189.3,
    "florence_total_ms": 648.2,
    "florence_backend": "mlx"
  }
}
```

### CLI Options

```
somflow <image> [options]

Options:
  -o, --output-dir DIR    Output directory (default: current)
  --task TASK             Florence-2 task token (default: <OD>)
  --overlap N             Quadrant overlap in pixels (default: 50)
  --iou FLOAT             IoU dedup threshold (default: 0.5)
  --fast                  Use fast OCR (5x faster, noisier text)
  --backend BACKEND       Detection backend: auto (default), coreml, mlx
```

## Requirements

- **macOS** (Apple Vision Framework is macOS-only)
- **Apple Silicon** (MLX requires Metal)
- **Python 3.10+**
- Florence-2 model: `mlx-community/Florence-2-base-ft-4bit` (~159MB, downloaded automatically on first run)

## Backend System

SomFlow supports pluggable detection backends via the `DetectionBackend` protocol:

- **MLX** (default) — Florence-2 inference on GPU via Metal. ~160ms per quadrant on M2 Max.
- **CoreML** — DaViT vision encoder on Apple Neural Engine, decoder on GPU. Useful when GPU is contended by other workloads. Requires a converted model (`python tools/convert_davit_coreml.py`).

```bash
# Use default MLX backend
somflow screenshot.png

# Use CoreML backend (ANE offload)
somflow screenshot.png --backend coreml
```

## Research Background

SomFlow emerged from a structured research effort evaluating detection approaches for a UI agent operating on macOS:

**Model survey (14+ models):** Evaluated detection models across HuggingFace, academic sources, and commercial options. AGPL-licensed models (Screen2AX, OmniParser, YOLO variants) were excluded — the target product ships under MIT. Florence-2, PTA-1, and Florence-2-large were shortlisted.

**Benchmark findings:**
- Florence-2-base-ft-4bit: 133ms warm inference, 159MB RAM, effective 4-bit quantization
- Florence-2-large-ft-4bit: **eliminated** — degenerate output (repeating `<s>` tokens) at 4-bit quantization
- PTA-1 (UI-specialized Florence-2 fine-tune): viable but 3x RAM (quantization achieved only 14-bit vs 4-bit target)

**Critical discovery:** All sub-10B detection models produce single full-screen bounding boxes on complex screenshots but work correctly on tiled inputs. This is a model capacity limitation, not a tuning problem — 7 configurations of frequency/repetition penalties were tested with no improvement. Tiling is architecturally required.

**Object-aware tiling:** Naive quadrant splits bisect UI elements at cut boundaries. SomFlow searches outward from the midpoint to find cut lines that avoid intersecting any detected bounding box, falling back to the midpoint with extra overlap padding when no clean gap exists.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Apple Vision + Florence-2 hybrid | Complementary strengths: Vision handles text/rectangles (free, ANE), Florence-2 handles open-vocabulary objects |
| 4-quadrant tiling | Simple, effective — keeps element count per tile manageable for sub-10B models |
| Object-aware cut placement | Prevents element fragmentation at tile boundaries |
| Pre-compiled Swift binary | Saves ~230ms JIT startup per Vision invocation |
| IoU dedup with source priority | Vision text > Vision rect > Florence-2 (higher priority sources kept on overlap) |
| No confidence score gating | VLM confidence scores correlate poorly with actual accuracy (~0.55 AUROC = near random) |
| MLX default backend | 1.25x faster than CoreML on idle GPU; CoreML available for GPU-contended workflows |

## Tests

```bash
# Fast tests (no model loading required)
pytest

# All tests including model-dependent ones
pytest --run-slow
```

53 fast tests covering: location token parsing, quadrant splitting, IoU computation, merge deduplication, SoM rendering, manifest generation, Apple Vision integration, backend protocol, backend selection, and encoder bridge conversion.

## License

MIT
