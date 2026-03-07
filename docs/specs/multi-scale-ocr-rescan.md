# Feature Spec: Multi-Scale OCR Rescan

> **Status: Pivoting.** Original hypothesis (upscale + re-OCR) disproved for the
> motivating case. But experiments revealed a much simpler fix: disabling Apple
> Vision's language correction on low-confidence elements produces dramatically
> better results for code/regex/variable text — without any upscaling needed.

---

## Summary

When UITag detects text elements with low OCR confidence, it can re-scan those
regions to improve accuracy. The original plan was crop + upscale + re-OCR.
Experiments showed upscaling doesn't help — but disabling language correction
on the re-scan pass does, dramatically.

---

## Background

During real-world testing (Keyboard Maestro macro configuration), UITag's
single-pass OCR produced unreliable results on special characters:

| som_id | OCR output (current) | Actual value | Confidence |
|--------|---------------------|--------------|------------|
| 7 | `([\\w_l+);` | `;([\w_]+);` | 0.30 |
| 27 | `: (Ow_]+);` | `;([\w_]+);` | 0.50 |
| 4 | `xmi-wrap-word` | `xml-wrap-word` | 0.85 |
| 39 | `TAG=[%Variable%Local_Tag%]` | `TAG=[%Variable%Local_Tag%]` | 0.50 |

Additionally, language correction introduces spurious changes to variables:
- `Local_Trigger` → `Local Trigger` (underscore removed)
- `%TriggerValue%` → `%Trigger Value%` (space inserted)

---

## Experiment Log

### Phase 1: Upscale + Re-OCR (2026-03-07) — Disproved

**Hypothesis:** Cropping and upscaling low-confidence elements will help
Apple Vision recognize special characters better.

**Method:** Crop each low-confidence element with 20-150px padding, upscale
1x-3x, re-OCR with Apple Vision in accurate mode.

**Results for SOM [7] (regex: `;([\w_]+);`):**

| Strategy | Crop Size | Vision Output | Conf | Verdict |
|----------|-----------|---------------|------|---------|
| Tight 1x (original) | 160×72 | `;[\w_]+);` | 0.30 | Baseline |
| Tight 2x | 320×144 | *no text detected* | — | Worse |
| Wide 2x (80px pad) | 560×384 | `;[\w_]+);` | 0.30 | Same |
| Very wide 2x | 840×664 | `¡ [w ]+);` | 0.30 | Worse (hallucinated `¡`) |
| Wide 3x | 840×576 | `¡ [w ]+);` | 0.30 | Worse |
| Very wide 1x | 420×332 | `; ([\w_l+);` | 0.50 | Closest, still wrong |

SOM [27] and [39]: similar — no improvement or degradation at any scale.

**Conclusion:** Upscaling does not fix the problem. Vision reads adjacent
English text at 1.00 confidence while reading the regex at 0.30 in the same
crop. This is not a resolution problem.

### Phase 2: `customWords` (2026-03-07) — No Effect

**Hypothesis:** Adding the expected regex patterns as `customWords` on
`VNRecognizeTextRequest` might bias Vision toward correct recognition.

**Method:** Set `customWords = [";([\w_]+);", "[\w_]+", "\w"]` and re-ran
on the same crops.

**Result:** Identical output to baseline. `customWords` operates at the
word-selection stage, which happens after character-level recognition. Since
the character recognizer can't produce the correct candidates, the custom
vocabulary has nothing to match against.

**Conclusion:** `customWords` cannot fix character-level recognition failures.

### Phase 3: `topCandidates(10)` (2026-03-07) — Revealing

**Hypothesis:** The correct reading might be a lower-ranked candidate.

**Method:** Retrieved all 10 candidates from Vision for each observation.

**Results for SOM [7] (lang correction ON):**

```
[1] conf=0.30  ';[\w_]+);'
[2] conf=0.30  ';([w_]+);'
[3] conf=0.30  ';[w_]+);'
[4] conf=0.30  ':[w_]+);'
...
[10] conf=0.30  '[w]+);'
```

The correct answer (`;([\w_]+);`) does NOT appear in any of the 10 candidates.
All candidates are at the same low confidence, suggesting Vision is uncertain
about the entire reading.

### Phase 4: Disabling Language Correction (2026-03-07) — BREAKTHROUGH

**Hypothesis:** Language correction's internal language model may be actively
corrupting the character-level output for non-natural-language text.

**Method:** Set `recognitionLevel = .accurate` (unchanged) with
`usesLanguageCorrection = false` (the key change). Tested on both crops
and the full screenshot.

**Results — Crops (tight 1x, accurate, NO lang correction):**

| Element | Lang Correction ON | Lang Correction OFF | Ground Truth |
|---------|-------------------|---------------------|--------------|
| **SOM [7]** | `;[\w_]+);` @ 0.30 | **`;([\w_]+);`** @ **1.00** | `;([\w_]+);` |
| **SOM [27]** | `:(Ow_]+);` @ 0.30 | `:([w_]+);` @ 1.00 | `;([\w_]+);` |
| **SOM [39]** | `TAG=[%Variable%Local_Tag%]` @ 0.50 | `TAG=[%Variable%Local_Tag%]` @ **1.00** | `TAG=[%Variable%Local_Tag%]` |

**SOM [7] is PERFECT.** The language correction was literally corrupting
a correct character-level reading.

**Results — Full screenshot (accurate, NO lang correction):**

| Text | Lang ON | Lang OFF | Verdict |
|------|---------|----------|---------|
| `xmi-wrap-word` @ 1.00 | `xml-wrap-word` @ 1.00 | **Fixed** (l vs i) |
| `Local Trigger` @ 1.00 | `Local_Trigger` @ 1.00 | **Fixed** (underscore preserved) |
| `%Trigger Value%` @ 1.00 | `%TriggerValue%` @ 1.00 | **Fixed** (no spurious space) |
| SOM 7: `([\w_l+);` @ 0.30 | `:([\w_1+);` @ 1.00 | **Much better** (still not perfect on full img) |
| SOM 27: `: (Ow_]+);` @ 0.50 | `:([w_]+);` @ 1.00 | **Much better** |
| SOM 39: same @ 0.50 | same @ **1.00** | **Confidence jump** |
| English text | All 1.00 | All 1.00 | **No degradation** |

Element count: identical (31 text elements both ways).

**Key insight:** On the full screenshot, SOM [7] gets `:([\w_1+);` (close
but not perfect). As a tight crop + no lang correction, it gets the perfect
`;([\w_]+);`. The combination of **crop + accurate + no lang correction**
is the winning strategy.

**Root cause confirmed:** Apple Vision's character-level recognizer (the
`.accurate` neural network) CAN see `[\w_]+` correctly. The language
correction post-processing then "fixes" it — replacing `\w` with characters
that look more like English words, inserting spaces in variable names,
and lowering confidence because the "corrected" result doesn't match any
known word either.

### Phase 5: Crop Boundary Sensitivity (2026-03-08) — CRITICAL FINDING

**Hypothesis:** SOM [27] produces imperfect results despite identical text
to SOM [7] because of differences in the surrounding pixel context.

**Method:** Systematically varied symmetric padding (0-60px in 5px steps)
on both SOMs, running Apple Vision with `accurate` + `usesLanguageCorrection
= false` on each crop.

**Results — SOM [7] (x=242, y=253, w=119, h=30):**

| Padding | Result | Quality |
|---------|--------|---------|
| 0px | `;([w ]+);` | Missing `\`, `_` → space |
| 5px | `:([w_ ]+);` | `;` → `:`, missing `\` |
| **10px** | **`;([\w_]+);`** | **PERFECT** |
| 15px | `;([w ]+);` | Missing `\`, `_` → space |
| 20px | `:([w_]+);` | `;` → `:`, missing `\` |
| 25px | `;([w ]+);` | Missing `\` |
| 30px | `;([w_]+);` | Missing `\` |
| 40px | garbled @ 0.30 | Wrong text entirely |

**Results — SOM [27] (x=119, y=1052, w=119, h=28):**

| Padding | Result | Quality |
|---------|--------|---------|
| 0px | `;(Dw_]+);` | `[` → `D` |
| 5px | `E(0w_]+);` | `;` → `E`, `[` → `0` |
| **10px** | **`;([w_]+);`** | **Best — only `\` missing** |
| **15px** | **`;([w_]+);`** | **Best — only `\` missing** |
| 20px | `:(Dw ]+);` | `;` → `:`, `[` → `D`, `_` → space |
| 25px | `;([w_]+);` | Best — only `\` missing |
| 30px | `;([w ]+);` | `_` → space |
| 50px | `¡(Dw_]+);` | `;` → `¡` hallucination |

**Key findings:**

1. **Apple Vision's OCR of special characters is chaotically sensitive to
   crop boundaries.** Shifting by 5px flips results from perfect to garbled.
2. **20px padding (our default) was a local minimum** for both SOMs — the
   worst symmetric padding value.
3. **10px padding is the sweet spot** — best single result for both SOMs.
4. **SOM [27]'s `\` before `w` is unrecoverable** at any padding. SOM [7]
   at 10px is the only combination that preserves it, likely due to
   sub-pixel antialiasing differences at different screen positions.

### Phase 5b: Multi-Crop Ensemble Strategy (2026-03-08)

**Hypothesis:** Since no single padding is universally optimal, trying
multiple paddings and selecting the best result should be more robust.

**Method:** For each detection, crop at 5 padding values (5, 10, 15, 20,
25px). Select the result with the most special/punctuation characters
(raw readings with more symbols are more likely correct), ties broken
by confidence.

**Results:**

```
=== SOM [7] ===
  pad= 5  special=7  ':([w_ ]+);'
  pad=10  special=9  ';([\w_]+);'  ◀ WINNER (PERFECT)
  pad=15  special=7  ';([w ]+);'
  pad=20  special=7  ':([w_]+);'
  pad=25  special=7  ';([w ]+);'

=== SOM [27] ===
  pad= 5  special=6  'E(0w_]+);'
  pad=10  special=8  ';([w_]+);'  ◀ WINNER (best possible)
  pad=15  special=8  ';([w_]+);'
  pad=20  special=5  ':(Dw ]+);'
  pad=25  special=8  ';([w_]+);'
```

The "most special characters" heuristic correctly selects the best reading
for both SOMs. Implemented in `_rescan_single()` as the default strategy.

**Root cause of SOM [27] `\` loss:** The `\` character (a thin 1-pixel
diagonal stroke in monospaced font) falls at the recognition threshold
of Vision's neural network. Sub-pixel antialiasing at y=1052 renders
it slightly differently than at y=253, pushing it below the threshold.
This is a hard limit of Apple Vision — not something we can fix with
crop strategies.

### Phase 6: Pre-Processing Gradient (2026-03-08) — No Sweet Spot

**Hypothesis:** A moderate level of sharpening or contrast enhancement
might recover the `\` without destroying other characters.

**Method:** Swept sharpening and contrast parameters across full gradients
on SOM [27]'s 10px crop:
- UnsharpMask: percent 50-600% (step 50), radius 0.5-5.0
- Contrast: 1.0-5.0x
- Sharpness: 1.0-20.0x
- Combined: Contrast × Sharpness grid

**Key results:**

| Processing | Result | Notes |
|-----------|--------|-------|
| Baseline (no processing) | `;([w_]+);` sc=8 | Best overall reading |
| Sharpness 8x | `:(+[м\])!` sc=6 ★ | Backslash recovered but garbled |
| Sharpness 10x | `:(+[м\])!` sc=6 ★ | Same garbled pattern |
| USM pct=500 r=2 | `!(+[m\])!` sc=5 ★ | Backslash but completely garbled |
| Edge Enhance More | `:(+[м\])!` sc=6 ★ | Same garbled pattern |
| All other combinations | sc=3-8 | No backslash, various degradation |

**Conclusion:** The `\` IS present in the pixel data — extreme sharpening
amplifies it above Vision's threshold. But every processing level that
recovers `\` also destroys the surrounding character recognition. There
is no sweet spot: the sharpness required for `\` exceeds the tolerance
of all other characters.

### Phase 7: Surrounding Context Influence (2026-03-08) — VALIDATED

**Hypothesis:** The nature of surrounding UI elements (text-heavy vs
UI-chrome-heavy) biases Vision's character recognition.

**Background:** SOM [7] sits in a UI-rich area (green/red +/- buttons
above, dropdown selector to the right, checkbox below, generous negative
space to the left). SOM [27] sits in a text-heavy area (label text
above and below, full-width text field, close to left edge of screenshot).

**Method:** Tested SOM [27]'s text pixels in various synthetic contexts,
blanked real context, and context-swapped between SOMs.

**Key results:**

| Context | Result | sc | Verdict |
|---------|--------|---:|---------|
| Real context, 10px crop | `;([w_]+);` | 8 | Baseline |
| UI rectangles, no text | `;([w_]+);` | 8 | Matches baseline at wider size |
| Mimic SOM[7] layout | `;([w_]+);` | 8 | Matches baseline at wider size |
| Real context, text blanked | `:(Ow ]+);` | 5 | Worse — blank regions hurt |
| Real context, text→UI rects | `:(Ow ]+);` | 5 | Replacing text with UI didn't help |
| **SOM[7] text in SOM[27]'s context** | **`L:(0w_]+);`** | 5 | **GARBLED — proves context influence** |
| SOM[27] text in SOM[7]'s context | `;(Dw_]+);` | 6 | Leading `;` recovered |
| SOM[7] context + SOM[27] text (any sharp) | `;([w_]+);` | 8 | **Ceiling — `\` still missing** |
| SOM[27] on large canvas, left-flush | `;([w_]+);` | 8 | Position within canvas matters |
| SOM[27] on large canvas, centered | `:(Ow ]+);` | 5 | Same text, worse when centered |

**Findings:**

1. **Context DOES influence OCR** — SOM [7]'s normally-perfect text
   becomes garbled when placed in SOM [27]'s surrounding context. Proved
   by the context-swap experiment.
2. **UI-like context is more favorable** than text-like context for
   special character recognition. Supports the hypothesis that Vision's
   neural network applies stronger "prose assumptions" when surrounded
   by text content.
3. **Context affects `;`, `[`, `_` but NOT `\`** — even in SOM [7]'s
   perfect context, SOM [27]'s text caps at `;([w_]+);`. The `\` is a
   pixel-level limit independent of context.
4. **Position within the crop matters** — left-flush vs centered on the
   same canvas produces different results.

### Phase 8: Light Mode vs Dark Mode (2026-03-08) — MAJOR FINDING

**Hypothesis:** Apple Vision may perform differently on light vs dark mode
screenshots due to training data bias.

**Background:** Operator confirmed screenshot is Retina 2x (144 DPI via
`sips`). Operator also confirmed KM font size cannot be increased — the
text size preference doesn't affect input fields, and macOS system text
size only affects Finder/Apple apps. Light mode screenshot captured by
operator for comparison.

**Method:** Same Keyboard Maestro macro, same screen, light mode. Ran
Vision on full screenshot and crop padding sweep (0-30px), with and
without language correction.

**Results — Full screenshot:**

| Element | Dark Mode (lang OFF) | Light Mode (lang OFF) |
|---------|---------------------|----------------------|
| **SOM [7]** | `R([w_]+);` @ 1.00 | **`;([\w_]+);`** @ 1.00 ★ **PERFECT** |
| **SOM [27]** | `¡(dW]+);` @ 0.50 | `;(Lw_]+);` @ 1.00 — much better |
| English text | All 1.00 | All 1.00 |

Light mode SOM [7] is **perfect on the full screenshot** — no cropping
needed at all.

**Results — SOM [7] padding sweep:**

| Padding | Dark Mode | Light Mode |
|---------|-----------|------------|
| 0px | `;([w ]+);` | `;([\w_]+);` ★ |
| 5px | `:([w_ ]+);` | `;([\w_]+);` ★ |
| **10px** | **`;([\w_]+);`** ★ | `;([\w_]+);` ★ |
| 15px | `;([w ]+);` | `;([w_]+);` |
| 20px | `:([w_]+);` | `;([w_]+);` |

Dark mode: backslash at exactly 1 padding value. Light mode: **3 padding
values**, far more robust.

**Results — SOM [27] padding sweep:**

| Padding | Dark Mode | Light Mode |
|---------|-----------|------------|
| 5px | `E(0w_]+);` | `;([w_]+);` |
| 10px | `;([w_]+);` | `;(0w_]+);` |
| 15px | `;([w_]+);` | `;([w_]+);` |
| 20px | `:(Dw ]+);` | `;(Dw_]+);` |
| **25px** | `;([w_]+);` | **`;([\w_]+);`** ★ **PERFECT** |

Dark mode SOM [27]: backslash **unrecoverable at any padding**. Light
mode: **backslash recovered at pad=25**.

**Multi-crop ensemble results:**

| Element | Dark Mode Ensemble | Light Mode Ensemble |
|---------|-------------------|---------------------|
| SOM [7] | `;([\w_]+);` ★ PERFECT | `;([\w_]+);` ★ PERFECT |
| SOM [27] | `;([w_]+);` (best, no `\`) | **`;([\w_]+);` ★ PERFECT** |

**Both SOMs achieve perfect readings in light mode with the ensemble.**

**Root cause:** Apple Vision's neural network almost certainly has a
training data bias toward light-background content (documents, web pages,
standard macOS UI). Dark text on light backgrounds provides higher
character contrast and more consistent sub-pixel rendering for the
neural network's convolutional feature extractors. The thin `\` stroke
that was below the detection threshold in dark mode (light-on-dark) is
above it in light mode (dark-on-light).

### Root Cause Summary (Updated)

Three factors at play, in order of significance:

1. **Color mode / contrast polarity (affects all characters including `\`):**
   Light mode (dark text on light background) produces dramatically better
   OCR for special characters. The `\` that was unrecoverable in dark mode
   is correctly read in light mode. This is the strongest factor — likely
   due to Apple Vision's training data distribution favoring light backgrounds.

2. **Sub-pixel rendering (position-dependent, dark mode amplifier):**
   macOS Core Text renders the same glyph with different sub-pixel patterns
   at different screen positions. In dark mode, this variance pushes thin
   strokes like `\` below the detection threshold at some positions. In
   light mode, the higher base contrast means sub-pixel variance matters
   less.

3. **Surrounding visual context (affects `;`, `[`, `_`):** Vision's text
   recognizer uses surrounding pixel context as part of its neural
   processing. Text-heavy surroundings bias toward "prose mode" which
   degrades special character recognition. UI-element-heavy surroundings
   maintain or improve accuracy. This is a secondary factor — it affects
   character disambiguation but cannot overcome pixel-level ambiguity.

---

## What We Built (Current State)

### Rescan Module (`uitag/rescan.py`)

Multi-crop ensemble re-OCR pipeline:
- Crops each low-confidence detection at 5 padding values (5, 10, 15, 20, 25px)
- Runs Apple Vision with `accurate` + `usesLanguageCorrection = false`
- Selects the reading with the most special characters (raw reading heuristic)
- No upscaling (1x crops are sufficient)

### Pipeline Integration

Stage 4b between merge and annotate, activated via `--rescan` flag.
Uses original image (before SoM annotation) — confirmed no color overlay
contamination.

### CLI

`--rescan` flag and automatic low-confidence callout. 7 unit tests passing.

### Swift Binary

`--no-lang-correction` flag added to `vision-detect.swift`. Binary
recompiled with `swiftc -O`.

---

## Lane Boundaries

### In UITag's lane (this spec)

- **Multi-crop ensemble re-OCR with different Vision config** — detection
  accuracy enhancement using the same OCR engine with different settings.
  No external dependencies, no domain knowledge, no maintenance burden.
- **Dual-pass strategy:** First pass with lang correction ON (best for
  English text), second pass with lang correction OFF (best for code/symbols)
  on low-confidence elements only.
- **MLX-native OCR models** as an alternative/complementary engine — still
  in the OCR lane, but changes system architecture (see below).

### In the inter-agent debate lane (NOT this spec)

- **Tesseract with custom dictionaries** — this is a domain-specific
  vocabulary approach. Training Tesseract on code patterns, regex syntax,
  or app-specific text requires maintaining curated training data. This
  overlaps with domain dictionaries and regex validation topics. It's not an
  OCR engine swap — it's a knowledge-maintenance commitment.
- **LLM post-correction** — out of scope for this spec.
- **Curated pattern libraries** — maintenance and boundary question.

### Why Tesseract is in the debate lane, not here

Tesseract with a default config is just another OCR engine. But the value
proposition for our use case is Tesseract with *custom training data* for
UI text and code patterns. That custom training data is exactly the
"curated domain dictionaries" topic from the OCR correction strategy doc.
It requires:
- Collecting training data (screenshots with known-correct labels)
- Maintaining the model as new UI patterns emerge
- Deciding who owns that maintenance (open source, premium, or external tooling)

That's the same boundary question as domain dictionaries. Mixing it into
the OCR rescan lane would create coherency issues with the broader
correction strategy.

---

## Performance & Architecture Notes

### Current Apple Vision Performance

Apple Vision runs on the Apple Neural Engine (ANE) via CoreML. It's
effectively free in terms of GPU contention — the ANE is a dedicated
accelerator. On the Keyboard Maestro screenshot (1644×2058):

- Accurate + lang correction: ~977ms for 31 text elements
- A single crop re-OCR: ~50-100ms estimated
- Multi-crop ensemble (5 paddings): ~250-500ms per element on ANE

### If We Introduce MLX-Native OCR Models

Models like PaddleOCR (PP-OCRv5) or Surya run on the GPU via Metal/MLX.
This means they **contend with Florence-2** for GPU time. Architectural
implications:

- Florence-2 already uses ~222ms/quadrant on the GPU
- An MLX OCR model running during rescan would add GPU latency
- Unlike Apple Vision (ANE), this wouldn't be "free" — it would extend
  the pipeline by the model's inference time
- Model download size: PaddleOCR ~15-50MB, Surya ~200MB+
- Additional dependency: PaddlePaddle or Surya packages

**Bottom line:** MLX OCR models add real latency and GPU contention.
They're worth investigating if the lang-correction fix is insufficient,
but they change the performance profile of the tool. Apple Vision's
ANE advantage (zero GPU cost) is a key selling point of uitag's speed.

### The Multi-Crop Ensemble Approach

- Uses the same Apple Vision engine (no new dependencies)
- Runs on the same ANE (no GPU contention)
- 5× more Vision calls per element, but ANE is fast (~250-500ms total)
- No model download, no library maintenance
- No domain dictionaries to curate
- Correctly selects best reading via special-character heuristic

---

## Open Items

| Item | Status | Notes |
|------|--------|-------|
| `--no-lang-correction` in Swift binary | **Done** | Implemented and compiled |
| Multi-crop ensemble in `_rescan_single()` | **Done** | 5 paddings, special-char heuristic |
| Pre-processing gradient sweep | **Done** | No sweet spot in dark mode |
| Surrounding context investigation | **Done** | Context swap proves influence; UI-like > text-heavy |
| Light mode vs dark mode | **Done** | Light mode dramatically better — both SOMs perfect with ensemble |
| DPI confirmation | **Done** | Confirmed Retina 144 DPI; KM font size not adjustable |
| Root cause hierarchy | **Documented** | Contrast polarity > sub-pixel rendering > surrounding context |
| Test on broader set of screenshots | TBD | Need IDE, settings UIs, more apps in both modes |
| Dark mode mitigation research | **Potential** | Could we pre-process dark crops to simulate light mode contrast? |
| Consider dual-pass on full screenshot | Under debate | Run entire screenshot twice? Or just crops? |
| MLX OCR fallback investigation | Parked | Only if ensemble approach is insufficient |
| Ensemble performance profiling | TBD | Measure actual ANE latency for 5-crop ensemble |
| Light mode recommendation in docs | **Potential** | Note that light mode screenshots produce better OCR accuracy |

---

*Created: 2026-03-06*
*Updated: 2026-03-08 — Added light/dark mode analysis, updated root cause hierarchy, full experiment log (Phases 1-8)*
