import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor

import lazyllm
from lazyllm.thirdparty import bs4
from lazyllm.tools.http_request import post_sync

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef, Cell,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock,
)
from .ocr_reader_base import _OcrReaderBase

lazyllm.config.add('paddle_api_key', str, None, 'PADDLE_API_KEY', description='The API key for PaddleOCR')


class PaddleOCRPDFReader(_OcrReaderBase):
    def __init__(self,
            url: str = 'https://k4q3k6o0l1hbx6jc.aistudio-app.com/layout-parsing',
            droped_types: Set[str] = {'aside_text', 'header', 'footer', 'number', 'header_image', 'seal'},
            **kwargs):
        super().__init__(url=url, droped_types=droped_types, **kwargs)
        self._api_key = lazyllm.config['paddle_api_key']

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

        response = post_sync(self._url, json_payload=payload, headers={
            'Authorization': f'token {self._api_key or ""}',
            'Content-Type': 'application/json'
        }, timeout=600)
        return response.text

    @override
    def _adapt_json_to_IR(self, raw: dict) -> List[Block]:
        blocks: List[Block] = []
        image_tasks: List[tuple] = []
        for page_idx, page_data in enumerate(raw['result']['layoutParsingResults']):
            markdown_images = page_data['markdown']['images']
            for item in page_data['prunedResult']['parsing_res_list']:
                block = self._adapt_one(item, page_idx, markdown_images, image_tasks)
                if block is not None:
                    blocks.append(block)
        self._download_images(image_tasks)
        return blocks

    def _adapt_one(self, item: dict, page_idx: int, markdown_images: Dict[str, str],
                   image_tasks: List[tuple]) -> Optional[Block]:
        label = item['block_label']
        if label in self._droped_types:
            return None

        content = item['block_content']
        bbox = BBox.from_list(item['block_bbox'])
        page = PageRef(index=page_idx, bbox=bbox)

        if label in ('paragraph_title', 'doc_title'):
            return HeadingBlock(page=page, text=content)
        elif label == 'image':
            img_src = self._extract_img_path(content)
            rel_path = Path(img_src).as_posix().lstrip('./')
            save_path = self._image_cache_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_tasks.append((markdown_images[img_src], save_path))
            return FigureBlock(page=page, image_path=Path(rel_path))
        elif label in ('table', 'chart'):
            return TableBlock(
                page=page,
                cells=self._parse_table_html(content),
                page_range=(page_idx, page_idx),
            )
        elif label in ('display_formula', 'inline_formula'):
            return FormulaBlock(page=page, latex=content, inline=False)
        elif label in ('code', 'algorithm'):
            return CodeBlock(page=page, text=content)
        else:
            return ParagraphBlock(page=page, text=content)

    @staticmethod
    def _download_images(image_tasks: List[tuple]) -> None:
        def _download_one(task: tuple) -> None:
            img_url, save_path = task
            resp = requests.get(img_url, timeout=120)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(_download_one, image_tasks)

    @staticmethod
    def _extract_img_path(html: str) -> str:
        soup = bs4.BeautifulSoup(html, 'html.parser')
        return soup.find('img')['src']

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
                'bbox': b.page.bbox.to_list(),
                'section_path': b.section.anchors,
            }
            b.update_metadata(metadata)
            node = DocNode(text=text, metadata=metadata, global_metadata=global_metadata)
            node.excluded_embed_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            node.excluded_llm_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            docs.append(node)

        return docs
