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


def test_near_miss_rect_absorbed_with_padding():
    """vision_rect extending a few pixels beyond text block is absorbed.

    Real-world case: Apple Vision's rect detector extends ~5px beyond its
    text detector boundary. Without padding the rect survives at ~80%
    containment (below the 85% threshold). With 5px block padding it's
    absorbed.
    """
    from uitag.group import group_text_blocks

    # Based on basic-settings-2: Sampler section, rect SoM 20 / block SoM 21
    # Block lines span y:692-803, but the rect starts at y:687 (5px above)
    dets = [
        _det("Different samplers can converge at", x=36, y=692, w=471, h=18),
        _det("different step counts and may result", x=36, y=716, w=471, h=18),
        _det("in different qualities.", x=36, y=740, w=200, h=18),
        # Rect starts 5px above the first text line
        _det("rectangle", x=36, y=687, w=87, h=24, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets)

    assert groups_formed == 1
    # Rect should be absorbed thanks to block_padding
    text_blocks = [d for d in result if d.source == "vision_text_block"]
    rects = [d for d in result if d.source == "vision_rect"]
    assert len(text_blocks) == 1
    assert len(rects) == 0


def test_near_miss_rect_without_padding_survives():
    """Verify near-miss rect survives when padding is explicitly zero."""
    from uitag.group import group_text_blocks

    dets = [
        _det("Different samplers can converge at", x=36, y=692, w=471, h=18),
        _det("different step counts and may result", x=36, y=716, w=471, h=18),
        _det("in different qualities.", x=36, y=740, w=200, h=18),
        _det("rectangle", x=36, y=687, w=87, h=24, source="vision_rect"),
    ]
    result, groups_formed = group_text_blocks(dets, block_padding=0)

    assert groups_formed == 1
    rects = [d for d in result if d.source == "vision_rect"]
    assert len(rects) == 1  # Rect survives without padding
