# uitag

[![Tests](https://github.com/swaylenhayes/uitag/actions/workflows/test.yml/badge.svg)](https://github.com/swaylenhayes/uitag/actions/workflows/test.yml)

A Set-of-Mark (SoM) detection pipeline for macOS that transforms screenshots into structured element maps. Apple Vision handles text recognition and rectangles; a fine-tuned YOLO model (bundled, 18 MB) covers icons, buttons, and visual controls that Vision misses. Everything runs on-device — no server, no API cost.

![uitag --yolo demo — 287 UI elements detected in ~3s on a VS Code screenshot](docs/assets/demo-yolo-composite.gif)

_287 elements detected in ~3s — text labels (Apple Vision) + icons and buttons (YOLO). [Full manifest →](docs/examples/vscode-287-yolo-manifest.json)_

## Why This Exists

Screenshots should be machine-readable. Every button, label, and icon should have a bounding box, a label, and coordinates — in seconds, on-device, under MIT license.

Apple Vision's text recognition and rectangle detection runs natively on macOS and catches most text-based UI elements. But it misses icons, toolbar buttons, and visual controls that have no text label. On ScreenSpot-Pro (1,581 targets across 26 professional applications), Vision-only detection covers 57.3% of targets. The remaining 42.7% are invisible to Vision — predominantly icons.

The bundled YOLO model was trained on GroundCUA (55K desktop screenshots, 3.56M human-verified annotations, MIT) to detect these missing elements. With both sources merged, detection coverage reaches 90.8% across macOS, Windows, and Linux screenshots.

## Quick Start

```bash
pip install uitag

# Vision-only (default, ~1s, ~150 detections)
uitag screenshot.png -o out/

# Vision + YOLO (~3s, ~300 detections, closes icon gap)
uitag screenshot.png --yolo -o out/

# Vision + YOLO + VLM (~15s, adds element_type labels)
uitag screenshot.png --yolo --vlm -o out/

# Batch process a folder
uitag batch screenshots/ -o out/
```

Two files per image: `screenshot-uitag.png` (annotated) + `screenshot-uitag-manifest.json` (structured data).

## Detection Coverage

Measured on [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) (1,581 annotations, 26 professional applications, macOS + Windows + Linux — [leaderboard](https://gui-agent.github.io/grounding-leaderboard)). Center-hit: does any detection's bounding box contain the center of the ground-truth target?

| Mode                     | Text  | Icon  | Overall | Zero Detection |
| ------------------------ | ----- | ----- | ------- | -------------- |
| Vision + YOLO (`--yolo`) | 92.7% | 87.6% | 90.8%   | 5.8%           |
| Vision-only (default)    | 66.4% | 42.5% | 57.3%   | 32.9%          |

These numbers measure detection _coverage_ — whether the element was found — not grounding accuracy (whether a model can follow an instruction to click a specific element). Detection coverage is the ceiling for any downstream agent built on uitag's SoM annotations.

Per-application results on the hardest cases:

| Application | Vision-only | Vision + YOLO |
| ----------- | ----------- | ------------- |
| SolidWorks  | 27%         | 84%           |
| Inventor    | 40%         | 97%           |
| Illustrator | 35%         | 77%           |
| Photoshop   | 51%         | 94%           |
| Blender     | 56%         | 99%           |
| EViews      | 100%        | 100%          |

Full per-application breakdown and methodology in [Performance](docs/performance.md).

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
--yolo                  Enable YOLO detection (adds ~2-3s, closes icon gap)
--vlm                   Enable VLM element typing (requires a running VLM server)
--vlm-vocab VOCAB       Vocabulary name or JSON path (default: leith-17)
--fast                  Use fast OCR (5x faster, noisier text)
--rescan                Re-scan low-confidence text at higher resolution
-v, --verbose           Show element list and timing JSON
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
    { "som_id": 7, "label": "corrected text" },
    { "som_id": 12, "hide": true }
  ]
}
```

## Output Format

### JSON

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "element_count": 312,
  "elements": [
    {
      "som_id": 1,
      "label": "File",
      "bbox": { "x": 48, "y": 0, "width": 26, "height": 16 },
      "confidence": 1.0,
      "source": "vision_text"
    },
    {
      "som_id": 52,
      "label": "Button",
      "bbox": { "x": 2752, "y": 305, "width": 101, "height": 104 },
      "confidence": 0.87,
      "source": "yolo",
      "element_type": "button"
    }
  ],
  "timing_ms": {
    "vision_ms": 942.5,
    "yolo_ms": 2150.3,
    "merge_ms": 3.3,
    "annotate_ms": 15.3,
    "manifest_ms": 0.2
  }
}
```

### Annotated Image

![uitag output — 287 tagged UI elements on a VS Code screenshot](docs/examples/hero-after-yolo.png)
_287 elements detected in ~3s with `--yolo`. [Full manifest JSON →](docs/examples/vscode-287-yolo-manifest.json)_

## Pipeline Architecture

```
Screenshot
  │
  ├─ [1] Apple Vision (compiled Swift binary)
  │      text recognition + rectangle detection
  │
  ├─ [2] YOLO tiled detection (--yolo, optional)
  │      640x640 tiles, 20% overlap, cross-tile NMS
  │
  ├─ [3] Merge + deduplicate
  │      IoU-based overlap removal, source priority ranking
  │
  ├─ [4] OCR correction
  │      Cyrillic homoglyphs, invisible Unicode, NFC normalization
  │
  ├─ [5] Text block grouping
  │      adjacent lines → paragraphs
  │
  ├─ [6] VLM classification (--vlm, optional)
  │      crop each non-text detection → VLM → element_type label
  │
  ├─ [7] SoM annotation
  │      numbered markers + colored bounding boxes
  │
  └─ [8] JSON manifest
         coordinates, labels, sources, element types, timing
```

## Tips

- _Use light mode for best OCR accuracy._ Apple Vision produces measurably better results on light mode screenshots, especially for special characters in code, regex patterns, and variable names. In testing, a backslash character (`\`) that was unrecoverable in dark mode across all techniques was correctly read in light mode. [Full research findings →](docs/research/ocr-rescan-experiments.md)
- _Use `--rescan` for code-heavy UIs._ If your screenshot contains regex, variable names, or other non-prose text, `--rescan` re-checks low-confidence elements with language correction disabled — preventing Apple Vision from "correcting" `Local_Trigger` to `Local Trigger` or `[\w_]+` to garbled text.
- _Use `--yolo` when you need icon coverage._ Default Vision-only mode is fast (~1s) but misses icons and visual controls. Adding `--yolo` brings coverage from 57% to 91% at the cost of ~2-3 extra seconds.
- _Use `--vlm` when you need element types._ VLM classification labels each non-text detection with a semantic type (button, icon, slider, text_field, etc.). Requires a running VLM server — see [Performance](docs/performance.md) for setup and timing.

## Documentation

- [API Reference](docs/api.md) — Functions, types, and manifest schema
- [Performance](docs/performance.md) — Benchmarks, detection coverage, and per-application breakdown
- [Troubleshooting](docs/troubleshooting.md) — Common issues and FAQ
- [Research Background](docs/research.md) — Model selection and benchmark methodology
- [OCR Rescan Research](docs/research/ocr-rescan-experiments.md) — Light/dark mode analysis and crop boundary experiments
- [Standard Prompt Research](docs/research/standard-prompt-investigation.md) — Toward a standard prompt for UI element classification

## Requirements

- macOS (Apple Vision Framework is macOS-only)
- Python 3.10+
- No model download needed for default mode (Vision-only)
- YOLO detection (`--yolo`): `pip install "uitag[yolo]"`. The model weights (18 MB) are bundled.
- VLM classification (`--vlm`): requires a running OpenAI-compatible VLM server. Validated with [MAI-UI-2B-bf16-v2](https://huggingface.co/mlx-community/MAI-UI-2B-bf16-v2) (5.3 GB).

## Development

### Research Background

Fourteen detection models were evaluated across HuggingFace, academic sources, and commercial options. AGPL-licensed models (Screen2AX, OmniParser) were excluded for MIT compatibility. All sub-10B detection models tested during initial evaluation (early 2026) produced single full-screen bounding boxes on complex screenshots but worked correctly on tiled inputs — a model capacity limitation confirmed across 7 penalty configurations. Tiling is architecturally required for small detection VLMs.

Apple Vision's text recognition and rectangle detection runs natively on macOS with zero model overhead. Against ScreenSpot-Pro (604 macOS screenshots), Vision-only text detection coverage reaches 71.1%.

To close the icon gap, a YOLO11s model was fine-tuned on [GroundCUA](https://huggingface.co/datasets/ServiceNow/GroundCUA) (224K tiled images, 9 element classes, 100 epochs on 2x H100 PCIe). The resulting model (18 MB) detects buttons, menus, inputs, navigation, sidebars, and visual elements that Vision misses. Combined coverage: 90.8% across 26 professional applications on 3 platforms.

VLM classification was validated using [MAI-UI-2B-bf16-v2](https://huggingface.co/mlx-community/MAI-UI-2B-bf16-v2) on 206 icon crops from the ScreenSpot-Pro macOS subset. Strict accuracy: 96.1% with zero-flip reproducibility across 3 runs.

### Design decisions made through research

| Decision                            | Rationale                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| Apple Vision as default             | Text + rectangle detection with zero model overhead                                             |
| YOLO detection opt-in (`--yolo`)    | Bundled 18 MB model closes icon gap (42.5% → 87.6%) at ~2s cost. Trained on GroundCUA (MIT).    |
| VLM classification opt-in (`--vlm`) | Crops non-text detections, sends to VLM server, attaches `element_type`. 96.1% strict accuracy. |
| Pre-compiled Swift binary           | Saves ~230ms JIT startup per Vision invocation                                                  |
| IoU dedup with source priority      | Vision text > Vision rect = YOLO (higher priority sources kept on overlap)                      |
| OCR correction pipeline             | Cyrillic homoglyphs, invisible Unicode, NFC normalization — zero false-positive risk            |
| Text block grouping                 | Adjacent text lines merged into paragraphs, reducing detection count by ~33% on dense UIs       |

### Reproducing results

Use the following commands to re-run/validate results:

```bash
git clone https://github.com/swaylenhayes/uitag.git
cd uitag
uv pip install -e ".[dev,yolo]"
uv run pytest
```

## License

MIT
