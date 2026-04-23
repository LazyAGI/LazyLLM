import copy
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from lazyllm import LOG

from .ocr_ir import (
    Block, BBox, PageRef, SectionPath,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)


# ---------- L1: Normalize ----------

def l1_normalize(blocks: List[Block], page_size: Optional[Tuple[float, float]] = None) -> List[Block]:
    blocks = _filter_noise(blocks, page_size)
    blocks = _reorder_reading(blocks)
    blocks = _normalize_headings(blocks)
    blocks = _drop_toc_pages(blocks)
    return blocks


def _filter_noise(blocks: List[Block], page_size: Optional[Tuple[float, float]]) -> List[Block]:
    if not page_size:
        page_size = (612.0, 792.0)  # Default A4
    page_width, page_height = page_size
    header_threshold = page_height * 0.08
    footer_threshold = page_height * 0.92

    result: List[Block] = []
    for b in blocks:
        bbox = b.page.bbox
        # Skip if fully in header/footer region and text is short
        if bbox.y1 < header_threshold or bbox.y0 > footer_threshold:
            text = b.text_content()
            if len(text) < 10:
                continue
        result.append(b)
    return result


def _reorder_reading(blocks: List[Block]) -> List[Block]:
    if not blocks:
        return blocks

    # Group by page
    pages: Dict[int, List[Block]] = {}
    for b in blocks:
        pages.setdefault(b.page.index, []).append(b)

    result: List[Block] = []
    for page_idx in sorted(pages.keys()):
        page_blocks = pages[page_idx]
        # Detect two-column layout by x0 distribution
        x0s = [b.page.bbox.x0 for b in page_blocks if b.page.bbox.x1 > b.page.bbox.x0]
        if len(x0s) < 2:
            result.extend(sorted(page_blocks, key=lambda b: (b.page.bbox.y0, b.page.bbox.x0)))
            continue

        x0s_sorted = sorted(x0s)
        mid_x = (x0s_sorted[0] + x0s_sorted[-1]) / 2
        left_count = sum(1 for x in x0s if x < mid_x)
        right_count = len(x0s) - left_count

        if left_count > 1 and right_count > 1 and min(left_count, right_count) >= len(x0s) * 0.25:
            # Two-column: sort left column first, then right
            left_blocks = [b for b in page_blocks if b.page.bbox.x0 < mid_x]
            right_blocks = [b for b in page_blocks if b.page.bbox.x0 >= mid_x]
            left_blocks.sort(key=lambda b: b.page.bbox.y0)
            right_blocks.sort(key=lambda b: b.page.bbox.y0)
            result.extend(left_blocks)
            result.extend(right_blocks)
        else:
            page_blocks.sort(key=lambda b: (b.page.bbox.y0, b.page.bbox.x0))
            result.extend(page_blocks)
    return result


def _normalize_headings(blocks: List[Block]) -> List[Block]:
    # Detect level gaps but do not insert virtual headings
    prev_level = 0
    for b in blocks:
        if isinstance(b, HeadingBlock):
            if prev_level > 0 and b.level > prev_level + 1:
                LOG.debug(f'[L1] Heading level gap detected: {prev_level} -> {b.level}')
            prev_level = b.level
    return blocks


def _drop_toc_pages(blocks: List[Block]) -> List[Block]:
    # Simple TOC detection: consecutive short blocks matching TOC patterns
    if not blocks:
        return blocks

    result: List[Block] = []
    i = 0
    while i < len(blocks):
        text = blocks[i].text_content().strip()
        if re.search(r'^目\s*[次录]', text):
            # Skip subsequent TOC entries (short text + number/page pattern)
            i += 1
            while i < len(blocks):
                ntext = blocks[i].text_content().strip()
                if len(ntext) < 50 and re.search(r'\d+$', ntext):
                    i += 1
                else:
                    break
            continue
        result.append(blocks[i])
        i += 1
    return result


# ---------- L2: Associate ----------

def l2_associate(blocks: List[Block]) -> List[Block]:
    blocks = _merge_cross_page_tables(blocks)
    _pair_captions(blocks)
    blocks = _inject_section_path(blocks)
    return blocks


def _merge_cross_page_tables(blocks: List[Block]) -> List[Block]:
    if not blocks:
        return blocks

    result: List[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if not isinstance(b, TableBlock):
            result.append(b)
            i += 1
            continue

        # Look ahead for next page table
        merged = False
        if i + 1 < len(blocks):
            next_b = blocks[i + 1]
            if isinstance(next_b, TableBlock) and next_b.page.index == b.page.index + 1:
                if _should_merge_tables(b, next_b):
                    merged_table = _merge_two_tables(b, next_b)
                    result.append(merged_table)
                    i += 2
                    merged = True

        if not merged:
            result.append(b)
            i += 1
    return result


def _should_merge_tables(a: TableBlock, b: TableBlock) -> bool:
    # Check header similarity using Jaccard
    a_header = _get_table_header_text(a)
    b_header = _get_table_header_text(b)
    if not a_header or not b_header:
        return False

    sim = _jaccard_similarity(a_header, b_header)
    if sim < 0.8:
        return False

    # Check position: a near bottom, b near top
    a_bottom = a.page.bbox.y1
    b_top = b.page.bbox.y0
    # Use normalized thresholds (assume 1000-scale or absolute; heuristic)
    # If values are small (< 1), they are normalized; else absolute
    scale = 1000.0 if a_bottom <= 1.0 else 792.0
    if a_bottom < scale * 0.85:
        return False
    if b_top > scale * 0.15:
        return False

    # Check no exact duplicate last/first row
    a_last = _get_last_row_text(a)
    b_first = _get_first_row_text(b)
    if a_last and b_first and a_last == b_first:
        return False

    return True


def _get_table_header_text(t: TableBlock) -> str:
    if not t.cells:
        return ''
    min_row = min(c.row for c in t.cells)
    header_cells = [c for c in t.cells if c.row == min_row]
    return ' '.join(c.text for c in sorted(header_cells, key=lambda x: x.col))


def _get_first_row_text(t: TableBlock) -> str:
    if not t.cells:
        return ''
    min_row = min(c.row for c in t.cells)
    cells = [c for c in t.cells if c.row == min_row]
    return ' '.join(c.text for c in sorted(cells, key=lambda x: x.col))


def _get_last_row_text(t: TableBlock) -> str:
    if not t.cells:
        return ''
    max_row = max(c.row for c in t.cells)
    cells = [c for c in t.cells if c.row == max_row]
    return ' '.join(c.text for c in sorted(cells, key=lambda x: x.col))


def _merge_two_tables(a: TableBlock, b: TableBlock) -> TableBlock:
    max_row_a = max((c.row for c in a.cells), default=-1)
    shifted_cells = []
    for c in b.cells:
        shifted_cells.append(copy.copy(c))
        shifted_cells[-1].row = c.row + max_row_a + 1

    # If b header matches a header, drop b header
    b_header_text = _get_table_header_text(b)
    a_header_text = _get_table_header_text(a)
    if b_header_text and b_header_text == a_header_text:
        min_b_row = min((c.row for c in b.cells), default=0)
        shifted_cells = [c for c in shifted_cells if c.row != min_b_row + max_row_a + 1]

    all_cells = list(a.cells) + shifted_cells

    return TableBlock(
        page=a.page,
        section=a.section,
        caption=a.caption or b.caption,
        footnote=a.footnote or b.footnote,
        cells=all_cells,
        page_range=(a.page_range[0], b.page_range[1]),
        merged_across_pages=True,
        source_pages=list(set((a.source_pages or [a.page.index]) + (b.source_pages or [b.page.index]))),
    )


def _pair_captions(blocks: List[Block]) -> None:
    for i, b in enumerate(blocks):
        if isinstance(b, FigureBlock):
            caption_idx = _find_nearest_caption(blocks, i, r'^(图|Figure|Fig\.?)\s*\d+')
            if caption_idx is not None and b.caption is None:
                b.caption = blocks[caption_idx].text_content()
        elif isinstance(b, TableBlock):
            caption_idx = _find_nearest_caption(blocks, i, r'^(表|Table)\s*\d+')
            if caption_idx is not None and b.caption is None:
                b.caption = blocks[caption_idx].text_content()


def _find_nearest_caption(blocks: List[Block], idx: int, pattern: str) -> Optional[int]:
    window = 3
    for offset in range(1, window + 1):
        for sign in (-1, 1):
            j = idx + sign * offset
            if 0 <= j < len(blocks):
                text = blocks[j].text_content().strip()
                if re.search(pattern, text, re.IGNORECASE) and len(text) < 200:
                    return j
    return None


def _inject_section_path(blocks: List[Block]) -> List[Block]:
    heading_stack: List[Tuple[int, str, str]] = []  # (level, text, anchor)

    for b in blocks:
        if isinstance(b, HeadingBlock):
            level = b.level
            # Pop until stack top level < current level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, b.text, b.anchor))
            b.section = SectionPath(
                anchors=[t for _, t, _ in heading_stack],
                level=len(heading_stack),
            )
        else:
            b.section = SectionPath(
                anchors=[t for _, t, _ in heading_stack],
                level=len(heading_stack),
            )
    return blocks


# ---------- Utilities ----------

def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union) if union else 0.0
