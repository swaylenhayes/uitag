# Performance

uitag runs the full detection pipeline in ~5 seconds on an M2 Max laptop with `--yolo`, producing ~300 detections with 90.8% coverage on ScreenSpot-Pro. Everything runs on-device — Apple Vision handles text, and the opt-in YOLO model handles icons and visual controls. The default Vision-only mode completes in ~1 second with ~150 detections. No API calls, no model downloads beyond the YOLO optional dependency.

---

## Methodology

All timing measurements were collected on an Apple M2 Max with 96 GB unified memory, using warm caches and 3-run averages. Individual runs varied by +/- 60ms (~2-3%).

Detection coverage was evaluated on [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) — 1,581 ground-truth targets across 26 professional applications on macOS, Windows, and Linux. The metric is center-hit: does any detection's bounding box contain the center of the ground-truth target? This measures whether uitag _found_ the element, not whether a downstream model could _name_ it from a natural language instruction. Detection coverage is the ceiling for any grounding system built on uitag's SoM annotations.

---

## Pipeline Modes

| Mode | Time | Detections | Coverage | Use Case |
| ---- | ---- | ---------- | -------- | -------- |
| Vision + YOLO + VLM (`--yolo --vlm`) | ~15s | ~300 | 90.8% | Full coverage + element typing |
| Vision + YOLO (`--yolo`) | ~3s | ~300 | 90.8% | Full coverage including icons |
| Vision + VLM (`--vlm`) | ~10s | ~150 | 57.3% | Element typing without YOLO |
| Vision-only (default) | ~1.0s | ~150 | 57.3% | Fast, no model download needed |
| Fast OCR (`--fast`) | ~0.4s | ~140 | — | Interactive use, rapid iteration |

All times are warm (models loaded). First invocation with YOLO adds ~14s for model loading.

Vision-only is the default. YOLO is opt-in via `--yolo` (requires `pip install uitag[yolo]`). VLM is opt-in via `--vlm` (requires a running VLM server). Florence-2 (`--florence`) is a legacy option superseded by YOLO.

---

## ScreenSpot-Pro Detection Coverage

### All Platforms (1,581 targets, 26 apps)

| Mode | Text (n=977) | Icon (n=604) | Overall | Zero Detection |
| ---- | ------------ | ------------ | ------- | -------------- |
| Vision + YOLO (`--yolo`) | 92.7% | 87.6% | 90.8% | 5.8% |
| Vision-only (default) | 66.4% | 42.5% | 57.3% | 32.9% |

### macOS Subset (604 targets, 9 apps)

| Mode | Text (n=398) | Icon (n=206) | Overall |
| ---- | ------------ | ------------ | ------- |
| Vision + YOLO | 93.7% | 87.9% | 91.7% |
| Vision-only | 71.1% | 46.6% | 62.7% |

The YOLO model closes the icon detection gap: 42.5% to 87.6% (+45.1 percentage points) across all platforms. Vision-only struggles most on Windows applications where UI conventions differ from macOS.

### Per-Application Results (Vision + YOLO, selected)

| Application | Overall | Text | Icon | Vision-only |
| ----------- | ------- | ---- | ---- | ----------- |
| Blender | 99% | 100% | 93% | 56% |
| EViews | 100% | 100% | 100% | 100% |
| Inventor | 97% | 97% | 100% | 40% |
| Linux common | 98% | 97% | 100% | 60% |
| Photoshop | 94% | 96% | 92% | 51% |
| Word | 98% | 99% | 93% | 75% |
| AutoCAD | 68% | 63% | 86% | 47% |
| FL Studio | 79% | 92% | 68% | 44% |
| VS Code | 87% | 91% | 82% | 71% |

YOLO lifts icon detection from 40-56% to 86-100% on most desktop UIs. AutoCAD (68%) and FL Studio (79%) remain the weakest — applications with specialized UI patterns that diverge from the training distribution.

---

## Cross-Benchmark Comparison

uitag was evaluated on three independent benchmarks. All measurements are detection coverage — did uitag find the target element?

### ScreenSpot-Pro (all platforms, n=1,581)

| Config | Text | Icon | Overall |
| ------ | ---- | ---- | ------- |
| Vision + YOLO | 92.7% | 87.6% | 90.8% |
| YOLO only | 82.4% | 75.7% | 80.1% |
| Vision only | 66.4% | 42.5% | 57.3% |

### GroundCUA (desktop, n=500 sample from 55K)

| Config | Recall@0.5 | Precision@0.5 | F1 | PageIoU |
| ------ | ---------- | ------------- | -- | ------- |
| YOLO only | 94.0% | 83.6% | 88.5% | 68.7% |
| Vision + YOLO | 89.9% | 49.0% | 63.5% | 47.2% |
| Vision only | 22.1% | — | — | 32.5% |

YOLO-only scores highest on GroundCUA because the model was trained on that distribution. The combined pipeline scores lower because Vision's text detections have different bounding box boundaries than GroundCUA annotations, hurting IoU matching. Per-category YOLO-only recall: Menu 97.9%, Button 97.2%, Sidebar 95.8%, Input 94.2%, Navigation 91.3%.

### UI-Vision (desktop, n=1,181)

| Config | Recall@0.5 |
| ------ | ---------- |
| YOLO only | 83.5% |
| Vision + YOLO | 82.0% |
| Vision only | 17.0% |

UI-Vision "basic" annotations label 1-3 target elements per image. High recall indicates the target was found; low precision is expected since uitag detects all elements on screen.

---

## VLM Classification

Detection tells you _where_ UI elements are. VLM classification (`--vlm`) tells you _what_ they are — button, icon, slider, text_field. Each non-text detection is cropped with padding and sent to an OpenAI-compatible VLM server, which returns a type label from a configurable vocabulary. Text detections are skipped because OCR already provides their labels.

The default vocabulary is `leith-17` (17 UI element types). Alternative vocabularies can be loaded via `--vlm-vocab`. VLM requires a separate server process — uitag does not embed the model.

### Accuracy

MAI-UI-2B-bf16-v2 was evaluated on 206 icon crops from the ScreenSpot-Pro macOS subset (604 images, 9 apps). Strict accuracy: the VLM's type label exactly matches the hand-labeled ground truth.

| Metric | Value |
| ------ | ----- |
| Strict accuracy | 96.1% (198/206) |
| Reproducibility | Zero flips across 3 runs (618 classifications) |
| Model memory | 5.3 GB |

The 8 misclassified crops (3.9%) are consistently wrong across runs — characterizable errors, not random noise.

### Timing

Measured on 4 images (synth control panels, CJK UI, game controller overlay) on an M2 Max with 96 GB unified memory. MAI-UI-2B-bf16-v2 served via vllm-mlx on localhost. All times are warm (server loaded, YOLO model cached).

| Config | Detections (avg) | Typed | Time (avg) |
| ------ | ---------------- | ----- | ---------- |
| Apple Vision | 75 | — | 0.5s |
| Apple Vision + YOLO | 82 | — | 2.2s |
| Apple Vision + VLM | 75 | 35/35 | 10.0s |
| Apple Vision + YOLO + VLM | 82 | 42/42 | 13.7s |

VLM adds ~0.27s per element, sequential. On a screenshot with 40 classifiable elements, that is roughly 11 seconds of VLM inference on top of the detection pipeline. First invocation with YOLO adds ~14s for model loading; VLM server startup is separate and depends on the serving framework.

Detection coverage is unchanged by VLM — the same elements are found with or without `--vlm`. VLM only adds type labels to existing detections.

---

## Stage Timing

### Vision + YOLO Pipeline

Measured on a 3840x2160 screenshot (32 tiles). M2 Max, warm cache.

| Stage | Time |
| ----- | ---- |
| Apple Vision (text + rectangles) | ~1.0s |
| YOLO tiled inference (32 tiles) | ~2.2s |
| Merge + dedup | <5ms |
| OCR correction + text grouping | <5ms |
| Annotate + manifest | <20ms |
| Total | ~3.5-5s |

Tile count scales with image resolution: 1920x1080 produces ~12 tiles, 3840x2160 produces ~32. The YOLO model runs each tile independently with cross-tile NMS at the end. Total varies with image resolution and element density.

### Vision-Only Pipeline

Measured on a 1920x1080 VS Code screenshot (~151 UI elements). M2 Max, warm cache.

| Stage | Accurate | Fast (`--fast`) | Notes |
| ----- | -------- | --------------- | ----- |
| Apple Vision | 977ms | 213ms | Text + rectangles via compiled Swift binary |
| OCR correction | <1ms | <1ms | Cyrillic homoglyphs, invisible Unicode, NFC |
| Text block grouping | <1ms | <1ms | Adjacent lines merged into paragraphs |
| Merge + dedup | <1ms | <1ms | IoU-based overlap removal |
| Annotate | <1ms | <1ms | SoM numbered overlay rendering |
| Manifest | <1ms | <1ms | JSON output generation |
| Total | ~1.0s | ~0.3s | |

Apple Vision dominates wall-clock time in both modes. All post-processing stages together consume less than 5ms.

---

## OCR Mode Comparison

Apple Vision offers two recognition levels. The `--fast` flag selects fast mode.

| | Accurate (default) | Fast (`--fast`) |
| --- | --- | --- |
| Vision time | ~977ms | ~213ms |
| Text quality | High fidelity, better with small/dense text | Noisier, may miss or misread small labels |
| Text count | 129 | 119 |
| Rectangle detection | Identical | Identical |
| Total pipeline (Vision-only) | ~1.0s | ~0.3s |

The 4.6x speedup comes from Apple Vision's fast recognition level, which uses a lighter text model. Rectangle detection is identical in both modes. The 10-element delta (129 vs 119) comes from small or low-contrast labels that the fast model misses.

---

## Caveats

- Timing scales with image resolution and element density. A 3840x2160 screenshot with YOLO produces ~32 tiles vs ~12 at 1920x1080, roughly doubling YOLO inference time.
- Apple Vision's text detection quality varies across screenshots of the same content. This is an upstream behavior, not a pipeline artifact.
- ScreenSpot-Pro coverage numbers reflect the dataset's application mix (26 apps, 3 platforms). Applications with non-standard UI frameworks score lower.
- Detection coverage is not grounding accuracy. A 90.8% center-hit rate means 90.8% of targets have _some_ overlapping detection. Whether a VLM can select the _correct_ detection from a natural language instruction is a separate evaluation.

---

## Reproducing

```bash
# Vision-only (default)
uitag screenshot.png -o out/

# With YOLO
uitag screenshot.png --yolo -o out/

# With VLM (requires running VLM server on port 8000)
uitag screenshot.png --vlm -o out/

# Full stack
uitag screenshot.png --yolo --vlm -o out/

# Fast mode
uitag screenshot.png --fast -o out/

# Full pipeline benchmark (per-stage timing)
uitag benchmark screenshot.png --runs 3
```

Timing data appears in the CLI output with `-v` and in the `timing_ms` field of the JSON manifest.
