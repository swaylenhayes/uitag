# uitag

[![Tests](https://github.com/swaylenhayes/uitag/actions/workflows/test.yml/badge.svg)](https://github.com/swaylenhayes/uitag/actions/workflows/test.yml)

A Set-of-Mark (SoM) detection pipeline for macOS that transforms screenshots into structured, annotated element maps. Built for Apple Silicon using Apple Vision Framework and Florence-2 on MLX.

![uitag demo — 229 UI elements detected in 2.4s on a VS Code screenshot](docs/assets/demo-composite-upper-right-10s.gif)

*229 elements detected in 2.4s — text labels (Apple Vision), rectangles, icons, and buttons (Florence-2). [Full manifest JSON →](docs/examples/vscode-229-manifest.json)*

## Quick Start

```bash
pip install uitag

# Tag a single screenshot
uitag screenshot.png -o out/

# Batch process a folder of screenshots
uitag batch screenshots/ -o out/
```

Two files per image: `screenshot-uitag.png` (annotated) + `screenshot-uitag-manifest.json` (structured data).

## Commands

```
uitag <image>                   Tag a single screenshot
uitag batch <dir> -o <out>      Batch process all images in a directory
uitag patch <image> -m <manifest> -p <patch>   Re-annotate with corrections
uitag render <image> -m <manifest>             Render from existing manifest
uitag benchmark <image>         Measure per-stage pipeline timing
```

### Common Options

```
-o, --output-dir DIR    Output directory (default: current dir or uitag-output/)
--fast                  Use fast OCR (5x faster, noisier text)
--rescan                Re-scan low-confidence text at higher resolution
--backend BACKEND       Detection backend: auto (default), coreml, mlx
```

### OCR Rescan

Low-confidence text detections are flagged automatically with an interactive rescan prompt. Or use `--rescan` directly:

```bash
uitag screenshot.png --rescan          # rescan all low-confidence text
uitag screenshot.png --rescan 7,27     # rescan specific elements by SOM ID
```

Rescan outputs use a `-rescan` suffix (`{stem}-uitag-rescan.png`) so standard outputs are preserved.

Rescan crops each element at 5 padding values and selects the best reading, with language correction disabled for code/regex accuracy. See [research details](docs/research/ocr-rescan-experiments.md).

### Patch & Render

Re-annotate images from modified manifests without re-running detection:

```bash
# Apply corrections from a patch file
uitag patch screenshot.png -m manifest.json -p corrections.json -o out/

# Render annotations from an existing manifest
uitag render screenshot.png -m manifest.json -o out/
```

Patch file format:
```json
{
  "patches": [
    {"som_id": 7, "label": "corrected text"},
    {"som_id": 12, "hide": true}
  ]
}
```

## Why This Exists

We needed a vision model that could find every button, label, and icon on a macOS screenshot — to make screenshots machine-readable. We surveyed 14 detection models. The best ones (Screen2AX, OmniParser) were AGPL — unusable for MIT distribution. The MIT-licensed options under 10B parameters — Florence-2, PTA-1, and others — all produced the same failure: a single bounding box covering the entire screen.

We tried 7 configurations of frequency and repetition penalties. Prompt engineering. Resolution reduction. Nothing fixed it. This isn't a tuning problem — it's a model capacity limitation.

Then we noticed something: the same models detect reliably on cropped regions.

That's the core insight. uitag doesn't force a small model to see a complex desktop. It tiles the screenshot into quadrants first — with cut lines placed to avoid bisecting UI elements — and runs detection on each tile separately. Apple Vision handles text and rectangles natively on the ANE (fast, free, no model download). Florence-2 catches everything else — icons, buttons, images — at 159MB on Metal.

⚡ **Every model we tested returned 1 bounding box. uitag returns 151 — in 1.7 seconds, fully open-source under MIT.** [Full research methodology →](docs/research.md)


## Pipeline Architecture

| Pipeline Stage | Process Flow Description |
| :--- | :--- |
| ![system architecture uitag](docs/assets/uitag-architecture-diagram.jpg)|* [1] VNRecognizeTextRequest + VNDetectRectanglesRequest<br> * [3] \<OD\> detection on each tile<br> * [4] IoU-based overlap removal, source priority ranking, numbered markers + colored bounding boxes<br> * [5] manifest includes element list with coordinates, labels, sources, timing |


## Output Format
### JSON

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
    "vision_ms": 942.5,
    "tiling_ms": 10.0,
    "florence_total_ms": 676.0,
    "merge_ms": 3.3,
    "annotate_ms": 15.3,
    "manifest_ms": 0.2
  }
}
```
### Annotated Image
![uitag output — 151 tagged UI elements on a VS Code screenshot](docs/examples/hero-after.png)
*151 elements detected in ~1.7s — text labels (Apple Vision), rectangles, icons, and buttons (Florence-2). [Full manifest JSON →](docs/examples/vscode-manifest.json)*

## Tips

- **Use light mode for best OCR accuracy.** Apple Vision produces measurably better results on light mode screenshots, especially for special characters in code, regex patterns, and variable names. In testing, a backslash character (`\`) that was unrecoverable in dark mode across all techniques was correctly read in light mode. [Full research findings →](docs/research/ocr-rescan-experiments.md)
- **Use `--rescan` for code-heavy UIs.** If your screenshot contains regex, variable names, or other non-prose text, `--rescan` re-checks low-confidence elements with language correction disabled — preventing Apple Vision from "correcting" `Local_Trigger` to `Local Trigger` or `[\w_]+` to garbled text.

## Documentation

- [API Reference](docs/api.md) — Functions, types, and manifest schema
- [Performance](docs/performance.md) — Benchmarks and optimization tips
- [Troubleshooting](docs/troubleshooting.md) — Common issues and FAQ
- [Research Background](docs/research.md) — Model selection and benchmark methodology
- [OCR Rescan Research](docs/research/ocr-rescan-experiments.md) — Light/dark mode analysis and crop boundary experiments
- [Contributing](CONTRIBUTING.md) — Setup and PR guidelines

## Requirements

- **macOS** (Apple Vision Framework is macOS-only)
- **Apple Silicon** (MLX requires Metal)
- **Python 3.10+**
- Florence-2 model: `mlx-community/Florence-2-base-ft-4bit` (~159MB, downloaded automatically on first run)

## Backend System

uitag supports pluggable detection backends via the `DetectionBackend` protocol:

- **MLX** (default) — Florence-2 inference on GPU via Metal. ~220ms per quadrant on M2 Max.
- **CoreML** — DaViT vision encoder on Apple Neural Engine, decoder on GPU. Useful when GPU is contended by other workloads. Requires a converted model (`python tools/convert_davit_coreml.py`).


## Development

```bash
git clone https://github.com/swaylenhayes/uitag.git
cd uitag
uv pip install -e ".[dev]"
uv run pytest  # 94 fast tests (11 skipped without --run-slow)
```

105 tests covering: location token parsing, quadrant splitting, IoU computation, merge deduplication, SoM rendering, manifest generation, schema validation, Apple Vision integration, backend protocol, backend selection, encoder bridge conversion, batch processing, benchmark formatting, OCR rescan, and patch/render re-annotation.

<details>
<summary>📋 <strong>Research Background</strong></summary>

uitag emerged from a structured research effort evaluating detection approaches for a UI agent operating on macOS:

**Model survey (14+ models):** Evaluated detection models across HuggingFace, academic sources, and commercial options. AGPL-licensed models (Screen2AX, OmniParser, YOLO variants) were excluded — the target product ships under MIT. Florence-2, PTA-1, and Florence-2-large were shortlisted.

**Benchmark findings:**
- Florence-2-base-ft-4bit: 133ms warm inference, 159MB RAM, effective 4-bit quantization
- Florence-2-large-ft-4bit: **eliminated** — degenerate output (repeating `<s>` tokens) at 4-bit quantization
- PTA-1 (UI-specialized Florence-2 fine-tune): viable but 3x RAM (quantization achieved only 14-bit vs 4-bit target)

**Critical discovery:** All sub-10B detection models produce single full-screen bounding boxes on complex screenshots but work correctly on tiled inputs. This is a model capacity limitation, not a tuning problem — 7 configurations of frequency/repetition penalties were tested with no improvement. Tiling is architecturally required.

**Object-aware tiling:** Naive quadrant splits bisect UI elements at cut boundaries. uitag searches outward from the midpoint to find cut lines that avoid intersecting any detected bounding box, falling back to the midpoint with extra overlap padding when no clean gap exists.

</details>

<details>
<summary>⚖️ <strong>Design Decisions</strong></summary>

| Decision | Rationale |
|----------|-----------|
| Apple Vision + Florence-2 hybrid | Complementary strengths: Vision handles text/rectangles (free, ANE), Florence-2 handles open-vocabulary objects |
| 4-quadrant tiling | Simple, effective — keeps element count per tile manageable for sub-10B models |
| Object-aware cut placement | Prevents element fragmentation at tile boundaries |
| Pre-compiled Swift binary | Saves ~230ms JIT startup per Vision invocation |
| IoU dedup with source priority | Vision text > Vision rect > Florence-2 (higher priority sources kept on overlap) |
| No confidence score gating | VLM confidence scores correlate poorly with actual accuracy (~0.55 AUROC = near random) |
| MLX default backend | 1.25x faster than CoreML on idle GPU; CoreML available for GPU-contended workflows |

</details>

## License

MIT
