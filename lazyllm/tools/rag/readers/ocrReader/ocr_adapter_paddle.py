import re
from typing import List, Optional
from pathlib import Path

from lazyllm import LOG
from lazyllm.thirdparty import bs4

from .ocr_ir import (
    Block, BBox, PageRef, Cell, SectionPath,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_adapter import ServiceAdapter, ServiceConfig, ServiceKind, PaddleVariant


class PaddleOCRAdapter(ServiceAdapter):
    @property
    def name(self) -> str:
        return 'paddleocr'

    def adapt(self, raw: dict, config: ServiceConfig) -> List[Block]:
        blocks: List[Block] = []
        # Handle both wrapped and unwrapped formats
        pages = raw.get('pages', [raw]) if isinstance(raw, dict) else [raw]
        for page_data in pages:
            if not isinstance(page_data, dict):
                continue
            page_idx = page_data.get('page_idx', 0)
            page_blocks = page_data.get('blocks', [])
            for item in page_blocks:
                block = self._adapt_one(item, page_idx, config)
                if block is not None:
                    blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict, page_idx: int, config: ServiceConfig) -> Optional[Block]:
        label = item.get('label', item.get('block_label', ''))
        content = item.get('content', item.get('block_content', ''))
        bbox = BBox.from_list(item.get('bbox', item.get('block_bbox', [])))
        page = PageRef(index=page_idx, bbox=bbox)

        if label in ('paragraph_title', 'doc_title'):
            level, text = _parse_markdown_heading(content)
            if level == 0:
                level = item.get('text_level', 1) or 1
                text = content
            return HeadingBlock(
                page=page, level=int(level), text=text,
                anchor=_make_anchor(text),
            )
        elif label == 'text':
            return ParagraphBlock(page=page, text=content)
        elif label in ('image', 'figure'):
            img_path = _extract_img_path(content)
            return FigureBlock(page=page, image_path=Path(img_path) if img_path else None)
        elif label == 'table':
            return TableBlock(
                page=page,
                cells=_parse_table_html(content),
                page_range=(page_idx, page_idx),
            )
        elif label == 'formula':
            return FormulaBlock(page=page, latex=content, inline=False)
        elif label == 'code':
            return CodeBlock(page=page, text=content)
        elif label in ('header', 'footer', 'page_number', 'aside_text', 'seal', 'number'):
            return None
        return None


def _parse_markdown_heading(content: str) -> tuple:
    m = re.match(r'^(#+)\s*(.*)$', content.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return 0, content


def _make_anchor(text: str) -> str:
    return text.strip().replace(' ', '-').replace('\n', '-')[:64]


def _extract_img_path(html: str) -> Optional[str]:
    try:
        soup = bs4.BeautifulSoup(html, 'html.parser')
        img = soup.find('img')
        if img:
            return img.get('src', '')
    except Exception:
        pass
    return None


def _parse_table_html(html_text: str) -> List[Cell]:
    cells: List[Cell] = []
    if not html_text:
        return cells
    try:
        soup = bs4.BeautifulSoup(html_text, 'html.parser')
        table = soup.find('table')
        if not table:
            return cells
        for row_idx, tr in enumerate(table.find_all('tr')):
            for col_idx, td in enumerate(tr.find_all(['td', 'th'])):
                rowspan = int(td.get('rowspan', 1))
                colspan = int(td.get('colspan', 1))
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    text=td.get_text(strip=True),
                ))
    except Exception as e:
        LOG.warning(f'[PaddleOCRAdapter] Failed to parse table HTML: {e}')
    return cells
