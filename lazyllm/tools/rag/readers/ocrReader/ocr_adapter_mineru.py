import copy
from typing import List, Optional
from pathlib import Path

from lazyllm import LOG
from lazyllm.thirdparty import bs4

from .ocr_ir import (
    Block, BBox, PageRef, Cell, SectionPath,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_adapter import ServiceAdapter, ServiceConfig, ServiceKind, MinerUVariant


class MinerUAdapter(ServiceAdapter):
    @property
    def name(self) -> str:
        return 'mineru'

    def adapt(self, raw: dict, config: ServiceConfig) -> List[Block]:
        # Support both direct content_list and wrapped result formats
        if isinstance(raw, list):
            content_list = raw
        elif isinstance(raw, dict):
            if 'result' in raw and isinstance(raw['result'], list):
                content_list = raw['result'][0].get('content_list', [])
            elif 'content_list' in raw:
                content_list = raw['content_list']
            else:
                content_list = []
        else:
            content_list = []

        blocks: List[Block] = []
        for item in content_list:
            block = self._adapt_one(item, config)
            if block is not None:
                blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict, config: ServiceConfig) -> Optional[Block]:
        ty = item.get('type', '')
        text_level = item.get('text_level', 0) or 0
        text = item.get('text', '')
        page_idx = item.get('page_idx', 0)
        bbox = BBox.from_list(item.get('bbox', []))

        # patch-specific fields
        lines = None
        page_width = page_height = None
        if config.mineru_variant == MinerUVariant.OFFLINE_LAZYLLM_PATCHED:
            lines = item.get('lines')
            page_width = item.get('page_width')
            page_height = item.get('page_height')

        # Update config.page_size from patch fields if not already set
        if page_width and page_height and not config.page_size:
            config.page_size = (float(page_width), float(page_height))

        page = PageRef(index=page_idx, bbox=bbox)

        # Map to RawBlockType
        if ty in ('text', 'title') and text_level > 0:
            return HeadingBlock(
                page=page, level=int(text_level), text=text,
                anchor=_make_anchor(text),
            )
        elif ty == 'text':
            return ParagraphBlock(page=page, text=text)
        elif ty == 'image':
            return FigureBlock(
                page=page,
                image_path=Path(img_path) if (img_path := item.get('img_path')) else None,
                caption=_first(item.get('image_caption')),
                footnote=_first(item.get('image_footnote')),
            )
        elif ty == 'table':
            return TableBlock(
                page=page,
                caption=_first(item.get('table_caption')),
                footnote=_first(item.get('table_footnote')),
                cells=_parse_table_html(item.get('table_body', '')),
                page_range=(page_idx, page_idx),
            )
        elif ty == 'equation':
            return FormulaBlock(page=page, latex=text, inline=False)
        elif ty == 'code':
            return CodeBlock(
                page=page, text=item.get('code_body', ''),
                language=item.get('guess_lang'),
                caption=_first(item.get('code_caption')),
            )
        elif ty == 'list':
            return ListBlock(
                page=page,
                items=item.get('list_items', []),
                ordered=False,
            )
        elif ty in ('header', 'footer', 'page_number', 'aside_text', 'page_footnote',
                    'discard', 'discarded'):
            return None  # Drop noise blocks at adapter level
        return None


def _make_anchor(text: str) -> str:
    return text.strip().replace(' ', '-').replace('\n', '-')[:64]


def _first(val):
    if isinstance(val, list) and val:
        return val[0]
    if isinstance(val, str):
        return val
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
        LOG.warning(f'[MinerUAdapter] Failed to parse table HTML: {e}')
    return cells
