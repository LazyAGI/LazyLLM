import copy
import re
from typing import List, Tuple, Optional, Dict

from lazyllm import LOG

from .ocr_ir import (
    Block, SectionPath,
    HeadingBlock, ParagraphBlock, TableBlock,
    FigureBlock, CodeBlock,
)


# ---------- L1: Normalize ----------

def l1_normalize(blocks: List[Block], fix_reader_order: bool = False) -> List[Block]:
    '''L1 normalization: reorder blocks by reading order, validate heading levels, drop TOC pages.'''
    if fix_reader_order:
        blocks = _two_column_reorder_reading(blocks)
        blocks = _detect_heading_level_gap(blocks)
    blocks = _drop_toc_pages(blocks)
    return blocks


def _two_column_reorder_reading(blocks: List[Block]) -> List[Block]:
    '''Restore natural reading order per page.

    Groups blocks by page, then detects two-column layouts by analysing the
    x0 distribution. If both left and right columns contain enough blocks,
    sorts each column top-to-bottom and concatenates left before right.
    Otherwise falls back to standard top-to-bottom, left-to-right ordering.
    '''
    pages: Dict[int, List[Block]] = {}
    for b in blocks:
        pages.setdefault(b.page.index, []).append(b)

    result: List[Block] = []
    for page_idx in sorted(pages.keys()):
        page_blocks = pages[page_idx]
        x0s = [b.page.bbox.x0 for b in page_blocks if b.page.bbox.x1 > b.page.bbox.x0]
        if len(x0s) < 2:
            result.extend(sorted(page_blocks, key=lambda b: (b.page.bbox.y0, b.page.bbox.x0)))
            continue

        x0s_sorted = sorted(x0s)
        mid_x = (x0s_sorted[0] + x0s_sorted[-1]) / 2
        left_count = sum(1 for x in x0s if x < mid_x)
        right_count = len(x0s) - left_count

        if left_count > 1 and right_count > 1 and min(left_count, right_count) >= len(x0s) * 0.25:
            left_x0s = [x for x in x0s if x < mid_x]
            right_x0s = [x for x in x0s if x >= mid_x]
            page_width = max(x0s) - min(x0s)
            min_col_width = page_width * 0.15
            left_width = max(left_x0s) - min(left_x0s)
            right_width = max(right_x0s) - min(right_x0s)

            if left_width >= min_col_width and right_width >= min_col_width:
                left_blocks = [b for b in page_blocks if b.page.bbox.x0 < mid_x]
                right_blocks = [b for b in page_blocks if b.page.bbox.x0 >= mid_x]
                left_blocks.sort(key=lambda b: b.page.bbox.y0)
                right_blocks.sort(key=lambda b: b.page.bbox.y0)
                result.extend(left_blocks)
                result.extend(right_blocks)
            else:
                page_blocks.sort(key=lambda b: (b.page.bbox.y0, b.page.bbox.x0))
                result.extend(page_blocks)
        else:
            page_blocks.sort(key=lambda b: (b.page.bbox.y0, b.page.bbox.x0))
            result.extend(page_blocks)
    return result


def _detect_heading_level_gap(blocks: List[Block]) -> List[Block]:
    '''Detect heading level gaps and log anomalies.

    Scans heading blocks in document order. If a heading level jumps by more
    than one (e.g. level 1 -> level 3), logs a debug message but does not
    insert virtual headings.
    '''
    prev_level = 0
    for b in blocks:
        if isinstance(b, HeadingBlock):
            if prev_level > 0 and b.level > prev_level + 1:
                LOG.debug(f'[L1] Heading level gap detected: {prev_level} -> {b.level}')
            prev_level = b.level
    return blocks


def _drop_toc_pages(blocks: List[Block]) -> List[Block]:
    '''Drop table-of-contents pages.

    Detects a TOC start by the Chinese pattern '目[次录]'. All subsequent
    short lines ending with a number are treated as TOC entries and removed.
    '''
    if not blocks:
        return blocks

    result: List[Block] = []
    i = 0
    while i < len(blocks):
        text = blocks[i].text_content().strip()
        if re.search(r'^目\s*[次录]|^Table\s*of\s*Contents|^Contents', text, re.IGNORECASE):
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

def l2_associate(blocks: List[Block], merge_table: bool = False) -> List[Block]:
    '''L2 association: merge cross-page tables, pair captions, inject section paths, merge paragraphs.'''
    if merge_table:
        blocks = _merge_cross_page_tables(blocks)
    caption_indices = _pair_captions(blocks)
    blocks = [b for i, b in enumerate(blocks) if i not in caption_indices]
    blocks = _inject_section_path(blocks)
    blocks = _merge_consecutive_paragraphs(blocks)
    for idx, b in enumerate(blocks):
        b.index = idx
    return blocks


def _merge_cross_page_tables(blocks: List[Block]) -> List[Block]:
    '''Merge tables that span across consecutive pages.

    Scans for adjacent TableBlocks on consecutive pages. If header similarity,
    vertical position, and row-deduplication heuristics all pass, the two
    tables are merged into a single TableBlock with shifted row indices.
    '''
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
    '''Return True if two consecutive page tables should be merged.

    Checks Jaccard header similarity (>0.8), vertical position heuristics
    (a near page bottom, b near page top), and avoids exact duplicate
    last/first rows.

    Also merges when the first table has very few data rows (≤1), which
    indicates a header-only continuation page with a different header.
    '''
    a_header = _get_table_header_text(a)
    b_header = _get_table_header_text(b)

    a_bottom = a.page.bbox.y1
    b_top = b.page.bbox.y0
    scale = 1000.0 if a_bottom <= 1.0 else 792.0
    if a_bottom < scale * 0.85:
        return False
    if b_top > scale * 0.15:
        return False

    a_last = _get_last_row_text(a)
    b_first = _get_first_row_text(b)
    if a_last and b_first and a_last == b_first:
        return False

    if a_header and b_header:
        sim = _jaccard_similarity(a_header, b_header)
        if sim >= 0.8:
            return True

    # Allow merge when the first table has ≤1 data row (likely a
    # header-only continuation page with a different header).
    a_data_rows = len(set(c.row for c in a.cells)) - (1 if a_header else 0)
    return a_data_rows <= 1


def _get_table_header_text(t: TableBlock) -> str:
    '''Concatenate header cell texts of a table.'''
    if not t.cells:
        return ''
    min_row = min(c.row for c in t.cells)
    header_cells = [c for c in t.cells if c.row == min_row]
    return ' '.join(c.text for c in sorted(header_cells, key=lambda x: x.col))


def _get_first_row_text(t: TableBlock) -> str:
    '''Concatenate first-row cell texts of a table.'''
    if not t.cells:
        return ''
    min_row = min(c.row for c in t.cells)
    cells = [c for c in t.cells if c.row == min_row]
    return ' '.join(c.text for c in sorted(cells, key=lambda x: x.col))


def _get_last_row_text(t: TableBlock) -> str:
    '''Concatenate last-row cell texts of a table.'''
    if not t.cells:
        return ''
    max_row = max(c.row for c in t.cells)
    cells = [c for c in t.cells if c.row == max_row]
    return ' '.join(c.text for c in sorted(cells, key=lambda x: x.col))


def _merge_two_tables(a: TableBlock, b: TableBlock) -> TableBlock:
    '''Merge table b into table a, shifting row indices and deduplicating headers.'''
    max_row_a = max((c.row for c in a.cells), default=-1)
    shifted_cells = []
    for c in b.cells:
        shifted_cells.append(copy.copy(c))
        shifted_cells[-1].row = c.row + max_row_a + 1

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


def _pair_captions(blocks: List[Block]) -> set:
    '''Pair captions with nearby figure, table, and code blocks.

    Scans a +/-3 window around each target block for a text line matching
    the corresponding caption pattern (e.g. 'Figure 1', 'Table 2').
    When found, the caption text is injected into the block's caption field
    and the original caption block index is returned for removal.

    Returns a set of indices that should be removed from the block list.
    '''
    to_remove: set = set()
    for i, b in enumerate(blocks):
        if isinstance(b, FigureBlock):
            caption_idx = _find_nearest_caption(blocks, i, r'^(图|Figure|Fig\.?)\s*\d+')
            if caption_idx is not None and b.caption is None:
                b.caption = blocks[caption_idx].text_content()
                to_remove.add(caption_idx)
        elif isinstance(b, TableBlock):
            caption_idx = _find_nearest_caption(blocks, i, r'^(表|Table)\s*\d+')
            if caption_idx is not None and b.caption is None:
                b.caption = blocks[caption_idx].text_content()
                to_remove.add(caption_idx)
        elif isinstance(b, CodeBlock):
            caption_idx = _find_nearest_caption(blocks, i, r'^(代码|Code|Listing|Algorithm)\s*\d+')
            if caption_idx is not None and b.caption is None:
                b.caption = blocks[caption_idx].text_content()
                to_remove.add(caption_idx)
    return to_remove


def _find_nearest_caption(blocks: List[Block], idx: int, pattern: str) -> Optional[int]:
    '''Search up to 3 positions around idx for a block whose text matches pattern.

    Returns the index of the first matching block, or None if none found.
    Only considers blocks with text shorter than 200 chars.
    '''
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
    '''Inject heading hierarchy (section path) into every block.

    Maintains a stack of active headings. For each heading encountered,
    pops higher-or-equal levels and pushes the current one. Every block
    receives a SectionPath containing the stacked heading texts.
    '''
    heading_stack: List[Tuple[int, str, str]] = []  # (level, text, anchor)

    for b in blocks:
        if isinstance(b, HeadingBlock):
            level = b.level
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


def _merge_consecutive_paragraphs(blocks: List[Block]) -> List[Block]:
    '''Merge consecutive ParagraphBlocks that share the same section path.

    Groups consecutive ParagraphBlocks by identical section anchors.
    Groups with more than one block are merged into a single ParagraphBlock
    whose text is joined by '\\n\\n'. The first block's page and bbox are retained.
    If a block carries a '_lines' attribute (patch mode), those lines are concatenated.
    '''
    if not blocks:
        return blocks

    result: List[Block] = []
    group: List[ParagraphBlock] = []

    def flush_group():
        if len(group) > 1:
            merged = ParagraphBlock(
                page=group[0].page,
                section=group[0].section,
                text='\n\n'.join(p.text for p in group),
            )
            all_lines = []
            for p in group:
                lines = getattr(p, '_lines', None)
                if lines:
                    all_lines.extend(lines)
            if all_lines:
                object.__setattr__(merged, '_lines', all_lines)
            result.append(merged)
        elif group:
            result.append(group[0])
        group.clear()

    for b in blocks:
        if isinstance(b, ParagraphBlock):
            if group and group[-1].section.anchors != b.section.anchors:
                flush_group()
            group.append(b)
        else:
            flush_group()
            result.append(b)

    flush_group()
    return result


# ---------- Utilities ----------

def _jaccard_similarity(a: str, b: str) -> float:
    '''Compute Jaccard similarity between two strings (word-level).'''
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union) if union else 0.0
