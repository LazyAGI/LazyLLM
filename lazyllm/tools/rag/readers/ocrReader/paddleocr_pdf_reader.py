import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set
from typing_extensions import override

import lazyllm
from lazyllm import LOG
from lazyllm.thirdparty import bs4
from lazyllm.tools.http_request import post_sync

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef, Cell,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_postprocessor import l1_normalize, l2_associate
from .ocr_reader_base import _OcrReaderBase

lazyllm.config.add('paddle_api_key', str, None, 'PADDLE_API_KEY', description='The API key for PaddleOCR')


class PaddleOCRPDFReader(_OcrReaderBase):
    def __init__(self,
                 url: str = 'https://k4q3k6o0l1hbx6jc.aistudio-app.com/layout-parsing',
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 format_block_content: bool = True,
                 use_layout_detection: bool = True,
                 use_chart_recognition: bool = True,
                 droped_types: Set[str] = {'aside_text', 'header', 'footer', 'number', 'header_image', 'seal'},
                 **kwargs):
        super().__init__(url=url, droped_types=droped_types, **kwargs)
        api_key = kwargs.get('api_key') or lazyllm.config['paddle_api_key']
        if not url and not api_key:
            raise ValueError('Either url or api_key must be provided')

        if url:
            self._url = url.rstrip('/') + '/layout-parsing'
        else:
            self._url = 'https://k4q3k6o0l1hbx6jc.aistudio-app.com/layout-parsing'

        if api_key:
            self._headers = {
                'Authorization': f'token {api_key}',
                'Content-Type': 'application/json'
            }
        else:
            self._headers = {'Content-Type': 'application/json'}

        self._format_block_content = format_block_content
        self._use_layout_detection = use_layout_detection
        self._use_chart_recognition = use_chart_recognition
        self._callback = callback
        self._page_size = None

    @override
    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        if not file.exists():
            raise FileNotFoundError(f'File not found: {file}')
        with open(file, 'rb') as f:
            file_bytes = f.read()
            file_data = base64.b64encode(file_bytes).decode('ascii')

        payload = {
            'file': file_data,
            'fileType': 0 if str(file).endswith('.pdf') else 1,
            'formatBlockContent': self._format_block_content,
            'useLayoutDetection': self._use_layout_detection,
            'useChartRecognition': self._use_chart_recognition,
            'prettifyMarkdown': True,
        }

        response = post_sync(self._url, json_payload=payload, headers=self._headers, timeout=600)
        return response.text

    @override
    def _build_nodes_from_response(self, response_json: str, file: Path,
                       extra_info: Optional[Dict] = None) -> List[DocNode]:
        try:
            if isinstance(file, str):
                file = Path(file)
            raw = json.loads(response_json)
            blocks = self._adapt_raw(raw)
            blocks = l1_normalize(blocks, self._page_size)
            blocks, relations = l2_associate(blocks)
            docs = self._build_nodes_from_blocks(blocks, file, extra_info)
            if not docs:
                LOG.warning(f'[PaddleOCRPDFReader] No elements found in response for: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[PaddleOCRPDFReader] Error parsing response for {file}: {e}')
            return []

    @override
    def _adapt_raw(self, raw: dict) -> List[Block]:
        blocks: List[Block] = []
        # Handle both wrapped and unwrapped formats
        pages = raw.get('pages', [raw]) if isinstance(raw, dict) else [raw]
        for page_data in pages:
            if not isinstance(page_data, dict):
                continue
            page_idx = page_data.get('page_idx', 0)
            page_blocks = page_data.get('blocks', [])
            for item in page_blocks:
                block = self._adapt_one(item, page_idx)
                if block is not None:
                    blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict, page_idx: int) -> Optional[Block]:
        label = item.get('label', item.get('block_label', ''))
        content = item.get('content', item.get('block_content', ''))
        bbox = BBox.from_list(item.get('bbox', item.get('block_bbox', [])))
        page = PageRef(index=page_idx, bbox=bbox)

        if label in ('paragraph_title', 'doc_title'):
            level, text = self._parse_markdown_heading(content)
            if level == 0:
                level = item.get('text_level', 1) or 1
                text = content
            return HeadingBlock(
                page=page, level=int(level), text=text,
                anchor=self._make_anchor(text),
            )
        elif label == 'text':
            return ParagraphBlock(page=page, text=content)
        elif label in ('image', 'figure'):
            img_src = self._extract_img_path(content)
            image_path = self._resolve_image_path(img_src, page_idx) if img_src else None
            return FigureBlock(page=page, image_path=image_path)
        elif label == 'table':
            return TableBlock(
                page=page,
                cells=self._parse_table_html(content),
                page_range=(page_idx, page_idx),
            )
        elif label == 'formula':
            return FormulaBlock(page=page, latex=content, inline=False)
        elif label == 'code':
            return CodeBlock(page=page, text=content)
        elif label in ('header', 'footer', 'page_number', 'aside_text', 'seal', 'number'):
            return None
        return None

    def _resolve_image_path(self, img_src: str, page_idx: int) -> Optional[Path]:
        if not img_src:
            return None
        if img_src.startswith('data:'):
            match = re.match(r'^data:([^;]+);base64,(.+)$', img_src)
            if match:
                mime_type, b64_data = match.group(1), match.group(2)
                ext = mime_type.split('/')[-1]
                if ext in ('jpeg', 'jpg'):
                    ext = 'jpg'
                elif ext not in ('png', 'gif', 'webp', 'bmp'):
                    ext = 'png'
                filename = f'page{page_idx}_fig_{uuid.uuid4().hex[:8]}.{ext}'
                file_path = self._image_cache_dir / filename
                file_path.write_bytes(base64.b64decode(b64_data))
                return Path(filename)
            return None
        p = Path(img_src)
        if p.is_absolute():
            try:
                return p.relative_to(self._image_cache_dir)
            except ValueError:
                return Path(p.name)
        return p

    @staticmethod
    def _parse_markdown_heading(content: str) -> tuple:
        m = re.match(r'^(#+)\s*(.*)$', content.strip())
        if m:
            return len(m.group(1)), m.group(2).strip()
        return 0, content

    @staticmethod
    def _make_anchor(text: str) -> str:
        return text.strip().replace(' ', '-').replace('\n', '-')[:64]

    @staticmethod
    def _extract_img_path(html: str) -> Optional[str]:
        try:
            soup = bs4.BeautifulSoup(html, 'html.parser')
            img = soup.find('img')
            if img:
                return img.get('src', '')
        except Exception:
            pass
        return None

    def _parse_table_html(self, html_text: str) -> List[Cell]:
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
            LOG.warning(f'[PaddleOCRPDFReader] Failed to parse table HTML: {e}')
        return cells

    @override
    def _build_nodes_from_blocks(self, blocks: List[Block], file: Path,
                                  extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []
        global_metadata = dict(extra_info) if extra_info else {}
        global_metadata['image_cache_dir'] = str(self._image_cache_dir)

        for b in blocks:
            text = b.text_content()
            metadata = {
                'file_name': file.name,
                'file_path': str(file),
                'type': b.ty,
                'page': b.page.index,
                'bbox': [b.page.bbox.x0, b.page.bbox.y0, b.page.bbox.x1, b.page.bbox.y1],
                'section_path': b.section.anchors,
            }
            if isinstance(b, HeadingBlock):
                metadata['text_level'] = b.level
            elif isinstance(b, TableBlock):
                metadata['table_caption'] = b.caption
                metadata['table_footnote'] = b.footnote
                if b.cells:
                    metadata['cells'] = [c.__dict__ for c in b.cells]
            elif isinstance(b, FigureBlock):
                metadata['image_path'] = str(b.image_path) if b.image_path else None
                metadata['image_caption'] = b.caption
            elif isinstance(b, FormulaBlock):
                metadata['latex'] = b.latex
            elif isinstance(b, CodeBlock):
                metadata['code_type'] = b.language
            elif isinstance(b, ListBlock):
                metadata['list_items'] = b.items
            node = DocNode(text=text, metadata=metadata, global_metadata=global_metadata)
            node.excluded_embed_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            node.excluded_llm_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            docs.append(node)

        return docs
