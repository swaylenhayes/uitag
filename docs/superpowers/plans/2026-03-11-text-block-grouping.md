---
title: 2026-03-11-text-block-grouping
type: note
permalink: uitag/docs/superpowers/plans/2026-03-11-text-block-grouping
---

# Text Block Grouping (Stage 4d) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group adjacent `vision_text` lines into paragraph-level text blocks, absorb contained word-level rectangles, and reduce manifest noise.

**Architecture:** New pipeline stage 4d (`uitag/group.py`) runs after OCR correction (4c) and before annotation (5). Groups vertically adjacent, left-aligned text lines into single `vision_text_block` detections with space-joined labels. Removes `vision_rect` detections >=85% contained within text blocks. Always-on, no flags.

**Tech Stack:** Python 3.11, pytest, uitag Detection dataclass

**Spec:** `docs/superpowers/specs/2026-03-11-text-block-grouping-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `uitag/group.py` | `group_text_blocks()` — grouping + rect absorption + SoM reassignment |
| Create | `tests/test_group.py` | Unit tests for grouping logic |
| Modify | `uitag/schema.json:60` | Add `"vision_text_block"` to source enum |
| Modify | `uitag/merge.py:5-9` | Add `"vision_text_block": 3` to `SOURCE_PRIORITY` |
| Modify | `uitag/types.py:16` | Update source field comment |
| Modify | `uitag/run.py` (after line 109) | Add Stage 4d call + top-level import + docstring update |
| Modify | `examples/use_as_library.py:36` | Update source filter to include text blocks |

---

## Chunk 1: Downstream Scaffolding + Core Grouping

### Task 1: Update schema and source references

**Files:**
- Modify: `uitag/schema.json:60`
- Modify: `uitag/merge.py:5-9`
- Modify: `uitag/types.py:16`
- Modify: `examples/use_as_library.py:36`

- [ ] **Step 1: Add `"vision_text_block"` to schema.json source enum**

In `uitag/schema.json`, line 60, change:
```json
"enum": ["vision_text", "vision_rect", "florence2"]
```
to:
```json
"enum": ["vision_text", "vision_text_block", "vision_rect", "florence2"]
```

- [ ] **Step 2: Add `"vision_text_block"` to SOURCE_PRIORITY in merge.py**

In `uitag/merge.py`, lines 5-9, change:
```python
SOURCE_PRIORITY = {
    "vision_text": 3,
    "vision_rect": 2,
    "florence2": 1,
}
```
to:
```python
SOURCE_PRIORITY = {
    "vision_text": 3,
    "vision_text_block": 3,
    "vision_rect": 2,
    "florence2": 1,
}
```

- [ ] **Step 3: Update source comment in types.py**

In `uitag/types.py`, line 16, change:
```python
source: str  # "vision_text", "vision_rect", "florence2"
```
to:
```python
source: str  # "vision_text", "vision_text_block", "vision_rect", "florence2"
```

- [ ] **Step 4: Update example library filter**

In `examples/use_as_library.py`, line 36, change:
```python
vision_text = [d for d in result.detections if d.source == "vision_text"]
```
to:
```python
vision_text = [d for d in result.detections if d.source in ("vision_text", "vision_text_block")]
```

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `pytest -x -q`
Expected: 118 passed, 11 skipped (current baseline)

- [ ] **Step 6: Commit**

```bash
git add uitag/schema.json uitag/merge.py uitag/types.py examples/use_as_library.py
git commit -m "feat: add vision_text_block source to schema and priority map"
```

---

### Task 2: Core text grouping function (TDD)

**Files:**
- Create: `tests/test_group.py`
- Create: `uitag/group.py`

- [ ] **Step 1: Write failing tests for text line grouping**

Create `tests/test_group.py`:

```python
"""Tests for text block grouping."""

from uitag.types import Detection


def _det(label, x, y, w, h, conf=1.0, source="vision_text", som_id=None):
    return Detection(label, x, y, w, h, conf, source, som_id=som_id)


# --- group_text_blocks: grouping behavior ---


def test_groups_adjacent_aligned_text_lines():
    """Adjacent left-aligned text lines merge into one text block."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Uses an LLM to analyze the image and", x=46, y=189, w=361, h=19),
        _det("generate a descriptive prompt. This", x=46, y=215, w=338, h=17),
        _det("prompt can be refined to help create new", x=46, y=237, w=390, h=18),
        _det("images with a similar look and feel.", x=39, y=260, w=338, h=21),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    assert len(result) == 1
    block = result[0]
    assert block.source == "vision_text_block"
    assert block.label == (
        "Uses an LLM to analyze the image and "
        "generate a descriptive prompt. This "
        "prompt can be refined to help create new "
        "images with a similar look and feel."
    )
    # Union bbox
    assert block.x == 39
    assert block.y == 189
    assert block.x + block.width == 46 + 390  # max right edge
    assert block.y + block.height == 260 + 21  # max bottom edge


def test_does_not_group_large_vertical_gap():
    """Lines with gap > 1.0x line height stay separate."""
    from uitag.group import group_text_blocks

    dets = [
        _det("CLIP Score", x=44, y=439, w=174, h=24),  # header
        _det("CLIP Score is used to evaluate the", x=46, y=497, w=324, h=16),  # body
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 0
    assert len(result) == 2
    assert result[0].source == "vision_text"
    assert result[1].source == "vision_text"


def test_does_not_group_different_x_alignment():
    """Lines at different x positions are not grouped."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Interrogate", x=161, y=336, w=136, h=26),
        _det("V", x=429, y=339, w=21, h=15),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 0
    assert len(result) == 2


def test_single_line_passes_through():
    """A single text line is not grouped."""
    from uitag.group import group_text_blocks

    dets = [_det("Tools", x=25, y=19, w=72, h=26)]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 0
    assert len(result) == 1
    assert result[0].source == "vision_text"
    assert result[0].label == "Tools"


def test_confidence_is_min_of_group():
    """Grouped block confidence is the minimum of its lines."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Line one", x=10, y=10, w=200, h=20, conf=1.0),
        _det("Line two", x=10, y=35, w=200, h=20, conf=0.7),
    ]
    result, _ = group_text_blocks(dets)

    assert result[0].confidence == 0.7


def test_non_text_sources_untouched():
    """Florence and rect detections pass through ungrouped."""
    from uitag.group import group_text_blocks

    dets = [
        _det("button", x=10, y=10, w=50, h=20, source="florence2"),
        _det("rectangle", x=10, y=40, w=50, h=20, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 0
    assert len(result) == 2
    assert result[0].source == "florence2"
    assert result[1].source == "vision_rect"


def test_empty_input():
    """Empty detection list returns empty."""
    from uitag.group import group_text_blocks

    result, groups_formed = group_text_blocks([])

    assert result == []
    assert groups_formed == 0


def test_som_ids_reassigned():
    """SoM IDs are re-assigned sequentially after grouping."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Header", x=10, y=10, w=100, h=20, som_id=1),
        _det("Line one", x=10, y=100, w=200, h=16, som_id=2),
        _det("Line two", x=10, y=120, w=200, h=16, som_id=3),
    ]
    result, _ = group_text_blocks(dets)

    assert len(result) == 2
    assert result[0].som_id == 1  # Header
    assert result[1].som_id == 2  # Grouped block


def test_mixed_height_lines_group_correctly():
    """Lines with different heights group when gap < last line's height."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Bold heading text", x=10, y=10, w=200, h=24),
        _det("Regular body text below", x=10, y=44, w=200, h=16),
    ]
    # Gap = 44 - (10+24) = 10, last line height = 24, gap < 24 -> merge
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    assert len(result) == 1


def test_same_y_different_x_not_grouped():
    """Side-by-side text at same y but different x stays separate."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Left column", x=10, y=100, w=150, h=20),
        _det("Right column", x=300, y=100, w=150, h=20),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 0
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_group.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uitag.group'`

- [ ] **Step 3: Implement `group_text_blocks` in `uitag/group.py`**

Create `uitag/group.py`:

```python
"""Text block grouping — merges adjacent vision_text lines into paragraphs."""

from __future__ import annotations

from uitag.types import Detection


def group_text_blocks(
    detections: list[Detection],
    max_y_gap_factor: float = 1.0,
    x_align_tolerance: int = 20,
    containment_threshold: float = 0.85,
) -> tuple[list[Detection], int]:
    """Group adjacent vision_text lines into text blocks.

    Adjacent text lines are grouped when:
    - Vertical gap < max_y_gap_factor * last line's height
    - Left edge within x_align_tolerance of group's first line

    Groups of 2+ lines become a single detection with source
    ``"vision_text_block"`` and space-joined labels. Vision rectangles
    mostly contained within text blocks are absorbed.

    Returns:
        (updated_detections, groups_formed)
    """
    text_dets = [d for d in detections if d.source == "vision_text"]
    other_dets = [d for d in detections if d.source != "vision_text"]

    if len(text_dets) < 2:
        result = text_dets + other_dets
        result.sort(key=lambda d: (d.y, d.x))
        for i, det in enumerate(result):
            det.som_id = i + 1
        return result, 0

    # Sort text detections by vertical position
    text_dets.sort(key=lambda d: (d.y, d.x))

    # Build groups of adjacent, left-aligned lines
    groups: list[list[Detection]] = [[text_dets[0]]]

    for det in text_dets[1:]:
        last = groups[-1][-1]
        first = groups[-1][0]
        y_gap = det.y - (last.y + last.height)
        x_diff = abs(det.x - first.x)

        if y_gap < max_y_gap_factor * last.height and x_diff <= x_align_tolerance:
            groups[-1].append(det)
        else:
            groups.append([det])

    # Merge groups into text blocks
    merged_text: list[Detection] = []
    groups_formed = 0

    for group in groups:
        if len(group) == 1:
            merged_text.append(group[0])
            continue

        groups_formed += 1
        min_x = min(d.x for d in group)
        min_y = min(d.y for d in group)
        max_x2 = max(d.x + d.width for d in group)
        max_y2 = max(d.y + d.height for d in group)
        min_conf = min(d.confidence for d in group)
        label = " ".join(d.label for d in group)

        merged_text.append(
            Detection(
                label=label,
                x=min_x,
                y=min_y,
                width=max_x2 - min_x,
                height=max_y2 - min_y,
                confidence=min_conf,
                source="vision_text_block",
            )
        )

    # Absorb contained vision_rects
    text_blocks = [d for d in merged_text if d.source == "vision_text_block"]
    filtered_other: list[Detection] = []

    for det in other_dets:
        if det.source == "vision_rect" and text_blocks:
            if _is_contained_in_any(det, text_blocks, containment_threshold):
                continue
        filtered_other.append(det)

    # Combine, re-sort, re-assign SoM IDs
    result = merged_text + filtered_other
    result.sort(key=lambda d: (d.y, d.x))
    for i, det in enumerate(result):
        det.som_id = i + 1

    return result, groups_formed


def _is_contained_in_any(
    rect: Detection,
    blocks: list[Detection],
    threshold: float,
) -> bool:
    """Check if rect is mostly contained within any text block."""
    for block in blocks:
        ix1 = max(rect.x, block.x)
        iy1 = max(rect.y, block.y)
        ix2 = min(rect.x + rect.width, block.x + block.width)
        iy2 = min(rect.y + rect.height, block.y + block.height)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        rect_area = rect.width * rect.height
        if rect_area == 0:
            return True
        if inter_area / rect_area >= threshold:
            return True

    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_group.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -x -q`
Expected: 129 passed (118 existing + 11 new), 11 skipped

- [ ] **Step 6: Commit**

```bash
git add uitag/group.py tests/test_group.py
git commit -m "feat: text block grouping with tests (Stage 4d core)"
```

---

## Chunk 2: Rectangle Absorption Tests + Pipeline Integration

### Task 3: Rectangle absorption tests

**Files:**
- Modify: `tests/test_group.py`

- [ ] **Step 1: Add rectangle absorption tests to `tests/test_group.py`**

Append to `tests/test_group.py`:

```python
# --- Rectangle absorption ---


def test_contained_rect_absorbed():
    """vision_rect mostly inside a text block is removed."""
    from uitag.group import group_text_blocks

    dets = [
        # Two text lines that will group (gap=6, height=19)
        _det("Line one of the paragraph text here", x=46, y=189, w=361, h=19),
        _det("Line two of the paragraph continues", x=46, y=214, w=338, h=17),
        # Small rect fully inside the text block area
        _det("rectangle", x=100, y=190, w=50, h=15, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    # Text block + no rect (absorbed)
    assert len(result) == 1
    assert result[0].source == "vision_text_block"


def test_large_container_rect_preserved():
    """vision_rect larger than a text block is NOT absorbed."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Line one of the paragraph", x=46, y=189, w=361, h=19),
        _det("Line two continues here", x=46, y=214, w=338, h=17),
        # Large container rect that extends well beyond text block
        _det("rectangle", x=13, y=91, w=471, h=298, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    # Text block + container rect (preserved)
    assert len(result) == 2
    sources = {d.source for d in result}
    assert "vision_rect" in sources
    assert "vision_text_block" in sources


def test_florence_detections_never_absorbed():
    """Florence detections inside a text block area are NOT absorbed."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Line one text", x=10, y=10, w=200, h=20),
        _det("Line two text", x=10, y=35, w=200, h=20),
        # Florence detection inside the text block area
        _det("button", x=50, y=15, w=30, h=10, source="florence2"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    sources = [d.source for d in result]
    assert "florence2" in sources


def test_rect_outside_text_block_preserved():
    """vision_rect outside any text block passes through."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Line one", x=10, y=10, w=200, h=20),
        _det("Line two", x=10, y=35, w=200, h=20),
        # Rect far away from text block
        _det("rectangle", x=400, y=400, w=50, h=50, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    assert len(result) == 2
    assert any(d.source == "vision_rect" for d in result)
```

- [ ] **Step 2: Run new tests**

Run: `pytest tests/test_group.py -v`
Expected: All 15 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_group.py
git commit -m "test: add rectangle absorption tests for text block grouping"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `uitag/run.py`

- [ ] **Step 1: Add top-level import for group_text_blocks**

In `uitag/run.py`, after line 11 (`from uitag.correct import correct_detections`), add:
```python
from uitag.group import group_text_blocks
```

This is consistent with `correct_detections` which also uses a top-level import (not a local import like rescan).

- [ ] **Step 2: Add Stage 4d call after line 109**

In `uitag/run.py`, after line 109 (`timing["corrections"] = correction_count`), add:

```python

    # Stage 4d: Text block grouping
    t0 = time.perf_counter()
    merged, groups_formed = group_text_blocks(merged)
    timing["group_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timing["groups_formed"] = groups_formed
```

- [ ] **Step 3: Update docstring to include all sub-stages**

In `uitag/run.py`, replace the pipeline stages list in the docstring (lines 29-35). The current docstring only lists stages 1-6; update it to include the sub-stages 4b, 4c, 4d that already exist in the code but were never added to the docstring:

```python
    Pipeline stages:
    1. Apple Vision (text + rectangles)
    2. Quadrant split
    3. Florence-2 on each quadrant (via backend)
    4. Merge + deduplicate
    4b. Rescan (optional)
    4c. OCR correction
    4d. Text block grouping
    5. Annotate SoM
    6. Generate manifest
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -x -q`
Expected: 133 passed (118 existing + 15 from test_group.py), 11 skipped

- [ ] **Step 3: Run lint**

Run: `ruff check uitag/ tests/ && ruff format --check uitag/ tests/`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add uitag/run.py
git commit -m "feat: integrate text block grouping as Stage 4d in pipeline"
```

---

### Task 5: Integration verification on real images

**Files:** None (verification only)

- [ ] **Step 1: Re-run on tools-1.png, inspect results**

```bash
rm -rf /Users/swaylen/Desktop/image-files/output
uitag batch /Users/swaylen/Desktop/image-files -o /Users/swaylen/Desktop/image-files/output
```

Expected: All 7 images process successfully, element counts drop (fewer fragments).

- [ ] **Step 2: Verify Image Interpreter description is one block**

Read the `tools-1-uitag-manifest.json` and confirm:
- One detection with `source: "vision_text_block"` containing the full paragraph
- Word-level rectangles inside that area are gone
- "Image Interpreter" header remains a separate `"vision_text"` detection
- "CLIP Score" header remains separate from its description block

- [ ] **Step 3: Compare element counts before/after**

Before grouping: tools-1.png had 67 elements, tools-1-r1.png had 68.
After grouping: both should be significantly lower.

- [ ] **Step 4: Open annotated images to visual-check**

```bash
open /Users/swaylen/Desktop/image-files/output/tools-1-uitag.png
open /Users/swaylen/Desktop/image-files/output/tools-1-r1-uitag.png
```

Verify: text blocks have a single bounding box, not 4+ overlapping boxes.

- [ ] **Step 5: Final commit (if any fixups needed)**

Only if issues found in verification. Otherwise, done.