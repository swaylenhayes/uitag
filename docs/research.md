# Research Notes

Technical context behind SomFlow's architecture decisions.

## Model Selection

### Survey Scope

14+ detection models evaluated for macOS-native UI element detection. Selection criteria:

1. **License**: MIT only (AGPL excluded — target product ships commercially)
2. **Runtime**: Must run on Apple Silicon via MLX or CoreML
3. **Task**: UI element detection on professional macOS screenshots (VS Code, Finder, System Preferences)
4. **Size**: Under 10B parameters (co-hosting with VLM in 96GB unified memory)

### Models Evaluated

| Model | License | Status | Notes |
|-------|---------|--------|-------|
| Florence-2-base-ft-4bit | MIT | **Selected** | 159MB, 133ms warm, effective 4-bit quant |
| Florence-2-large-ft-4bit | MIT | Eliminated | Degenerate at 4-bit (repeating `<s>` tokens) |
| PTA-1 (UI fine-tune of Florence-2) | MIT | Viable alt | 458MB, 130ms warm, but quantization only reached 14-bit |
| Screen2AX | AGPL | Reference only | macOS-specific YOLO, best quality, but AGPL |
| OmniParser | AGPL | Excluded | Strong results but license incompatible |
| YOLO variants (Ultralytics) | Commercial | Contingency | Requires paid license for distribution |
| Various HF models | Mixed | Excluded | See notes below |

Additional models excluded during survey: PaddleOCR-VL (incompatible), GLM-OCR (incompatible), Kimi-VL (incompatible), InternVL3-8B (incompatible), gemma-3n (upstream bug).

### Why Florence-2-base Over Florence-2-large

Florence-2-large-ft-4bit produces degenerate output at 4-bit quantization — repeating `<s>` tokens instead of detection results. This pattern matches other aggressively quantized models (Qwen3-VL-4B in long-context mode, early LFM variants). The base model's architecture quantizes cleanly to 4-bit with no quality degradation.

### Why Not PTA-1

PTA-1 is a UI-specialized fine-tune of Florence-2-base from Microsoft. It works well but:
- `mlx_vlm.convert` quantization achieved only 14.15-bit effective (vs 4-bit target)
- Result: 458MB vs Florence-2-base's 159MB — 3x memory for marginal quality improvement
- The UI-specific fine-tuning doesn't justify the cost when tiling already makes base-model detection reliable

## The Tiling Requirement

### Problem

Sub-10B vision models (Florence-2, PTA-1, and others tested) produce a single full-screen bounding box when given complex macOS screenshots. They work correctly on simpler images or cropped regions.

### Attempted Fixes (None Worked)

| Approach | Result |
|----------|--------|
| `frequency_penalty` tuning (7 configs) | No improvement — structural limitation |
| `repetition_penalty` tuning | Actively harmful for structured output (penalizes repeated structural tokens) |
| Prompt engineering | Minimal impact on detection behavior |
| Resolution reduction | Quality degrades before behavior changes |

### Solution: Tiling

Split the image into 4 quadrants. Each quadrant has ~1/4 the complexity. Models detect reliably on tiles. Merge results with IoU deduplication.

**Object-aware tiling** improves on naive splitting: search outward from the image midpoint for cut lines that avoid intersecting any detected bounding box. This prevents elements at tile boundaries from being fragmented across two tiles.

## Apple Vision Integration

Apple's Vision framework runs on the ANE (Apple Neural Engine) and provides:
- `VNRecognizeTextRequest`: OCR with bounding boxes (~189ms fast, ~980ms accurate)
- `VNDetectRectanglesRequest`: Rectangle detection (near-instant)

These are free, require no model download, and handle the majority of UI text elements. Florence-2 supplements with non-text elements that Vision misses (icons, images, unlabeled buttons).

The integration uses a pre-compiled Swift binary (`swiftc -O`) rather than the Swift interpreter, saving ~230ms JIT startup per invocation.

### Fast vs Accurate OCR

| Mode | Time | Quality | Example |
|------|------|---------|---------|
| Accurate (default) | ~980ms | High fidelity | "Settings" |
| Fast | ~189ms | Noisy on rendered text | "88ttlngs" |

Default is accurate. Use `--fast` when text label quality is less important than throughput.

## Performance Profile

Measured on M2 Max (96GB) with 1920x1080 VS Code screenshot:

| Stage | Time | Notes |
|-------|------|-------|
| Apple Vision (accurate) | 980ms | ANE, no GPU contention |
| Apple Vision (fast) | 189ms | 5x faster, quality tradeoff |
| Object-aware split | <1ms | CPU only, trivial |
| Florence-2 (4 quadrants) | ~600ms | ~148ms/quadrant, sequential (optimized MLX) |
| Merge + dedup | <1ms | CPU only |
| SoM annotation | <10ms | Pillow drawing |
| Manifest generation | <1ms | JSON serialization |
| **Total (accurate)** | **~1600ms** | |
| **Total (fast)** | **~800ms** | |

**Pre-optimization baseline** (v0.1.0): Florence-2 ran at ~800ms/quadrant (~3200ms total). MLX optimizations (pre-saved temp files, warm inference) achieved a 4.9x speedup.

### Bottleneck

Florence-2 inference at ~148ms per quadrant is now well-optimized. CoreML backend is available (`--backend coreml`) and offloads to the ANE, which is beneficial under GPU contention but slightly slower (~186ms/quadrant) on an idle GPU.

### Co-hosting Validation

Florence-2 (159MB) co-hosts successfully alongside an 8B VLM (7.4GB) in 96GB unified memory:
- No memory contention
- VLM server throughput unaffected
- Florence-2 latency increase: +5.7% (within noise)

## References

- [Set-of-Mark (SoM) Prompting](https://arxiv.org/abs/2310.11441) — Yang et al., 2023
- [Florence-2: Advancing a Unified Representation](https://arxiv.org/abs/2311.06242) — Xiao et al., 2023
- [MLX: Machine Learning on Apple Silicon](https://github.com/ml-explore/mlx)
- [mlx-vlm: Vision Language Models on MLX](https://github.com/Blaizzy/mlx-vlm)
