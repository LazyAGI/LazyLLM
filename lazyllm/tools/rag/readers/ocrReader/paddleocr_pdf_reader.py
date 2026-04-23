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
                 droped_types: Set[str] = {'aside_text', 'header', 'footer', 'number', 'header_image', 'seal'},
                 **kwargs):
        super().__init__(url=url, droped_types=droped_types, **kwargs)
        self._api_key = lazyllm.config['paddle_api_key']
        self._headers = {
            'Authorization': f'token {self._api_key or ""}',
            'Content-Type': 'application/json'
        }

    @override
    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        with open(file, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('ascii')

        payload = {
            'file': file_data,
            'fileType': 0 if str(file).endswith('.pdf') else 1,
            'formatBlockContent': True,
            'useLayoutDetection': True,
            'useChartRecognition': True,
            'prettifyMarkdown': True,
        }

        response = post_sync(self._url, json_payload=payload, headers=self._headers, timeout=600)
        return response.text

    @override
    def _build_nodes_from_response(self, response_json: str, file: Path,
                       extra_info: Optional[Dict] = None) -> List[DocNode]:
        raw = json.loads(response_json)
        blocks = self._adapt_raw(raw)
        blocks = l1_normalize(blocks, self._page_size)
        blocks, relations = l2_associate(blocks)
        return self._build_nodes_from_blocks(blocks, file, extra_info)

    @override
    def _adapt_raw(self, raw: dict) -> List[Block]:
        blocks: List[Block] = []
        for page_idx, page_data in enumerate(raw['result']['layoutParsingResults']):
            markdown_images = page_data['markdown']['images']
            for item in page_data['prunedResult']['parsing_res_list']:
                block = self._adapt_one(item, page_idx, markdown_images)
                if block is not None:
                    blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict, page_idx: int,
                   markdown_images: Dict[str, str]) -> Optional[Block]:
        label = item['block_label']
        content = item['block_content']
        bbox = BBox.from_list(item['block_bbox'])
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
            image_path = self._resolve_image_path(img_src, markdown_images)
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

    def _resolve_image_path(self, img_src: str, markdown_images: Dict[str, str]) -> Path:
        img_b64 = markdown_images[img_src]
        rel_path = Path(img_src)
        clean_path = rel_path.as_posix().lstrip('./')
        save_path = self._image_cache_dir / clean_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(base64.b64decode(img_b64))
        return Path(clean_path)

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
    def _extract_img_path(html: str) -> str:
        soup = bs4.BeautifulSoup(html, 'html.parser')
        return soup.find('img')['src']

    def _parse_table_html(self, html_text: str) -> List[Cell]:
        soup = bs4.BeautifulSoup(html_text, 'html.parser')
        cells: List[Cell] = []
        for row_idx, tr in enumerate(soup.find('table').find_all('tr')):
            for col_idx, td in enumerate(tr.find_all(['td', 'th'])):
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=int(td.get('rowspan', 1)),
                    colspan=int(td.get('colspan', 1)),
                    text=td.get_text(strip=True),
                ))
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
                metadata['image_path'] = str(b.image_path)
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
