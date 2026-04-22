import base64
import json
import requests
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable
from typing_extensions import override

from lazyllm.thirdparty import bs4

import lazyllm
from lazyllm import LOG
from ...doc_node import DocNode
from ..readerBase import _RichReader
from .ocr_ir import Block, HeadingBlock, TableBlock, FigureBlock, FormulaBlock, CodeBlock, ListBlock
from .ocr_adapter import ServiceVariant, ServiceConfig, ServiceKind, PaddleVariant, AdapterRegistry
from .ocr_postprocess import l1_normalize, l2_associate

lazyllm.config.add('paddleocr_api_key', str, None, 'PADDLEOCR_API_KEY', description='The API key for PaddleOCR')


class PaddleOCRPDFReader(_RichReader):
    def __init__(self, url: str = None, api_key: str = None,
                 service_variant: ServiceVariant = 'online',
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 format_block_content: bool = True,
                 use_layout_detection: bool = True,
                 use_chart_recognition: bool = True,
                 split_doc: bool = True,
                 drop_types: List[str] = None,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True,
                 images_dir: str = None):
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)
        api_key = api_key or lazyllm.config['paddleocr_api_key']
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
        self._variant = PaddleVariant(variant)
        self._format_block_content = format_block_content
        self._use_layout_detection = use_layout_detection
        self._use_chart_recognition = use_chart_recognition
        if images_dir:
            self._images_dir = Path(images_dir)
            self._images_dir.mkdir(exist_ok=True)
        else:
            self._images_dir = None
        self._drop_types = (
            list(drop_types)
            if drop_types is not None
            else ['aside_text', 'header', 'footer', 'number', 'header_image', 'seal']
        )
        self._callback = callback
        self._registry = AdapterRegistry()
        self._config = ServiceConfig(
            kind=ServiceKind.PADDLEOCR,
            paddle_variant=self._variant,
        )

    @override
    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   use_cache: bool = True, **kwargs) -> List[DocNode]:
        try:
            if isinstance(file, str):
                file = Path(file)
            response_json = self._fetch_response(file)
            return self.from_response(response_json, file, extra_info)
        except Exception as e:
            LOG.error(f'[PaddleOCRPDFReader] Error loading data from {file}: {e}')
            return []

    def from_response(self, response_json: str, file: Path,
                      extra_info: Optional[Dict] = None) -> List[DocNode]:
        """Parse a PaddleOCR service response (JSON string) into DocNodes.

        This is the primary entry point when the caller has already obtained
        the service response.
        """
        try:
            if isinstance(file, str):
                file = Path(file)
            raw = json.loads(response_json)
            adapter = self._registry.get(self._config)
            blocks = adapter.adapt(raw, self._config)
            blocks = l1_normalize(blocks, self._config.page_size)
            blocks, relations = l2_associate(blocks)
            docs = self._build_nodes_from_blocks(blocks, file, extra_info)
            if not docs:
                LOG.warning(f'[PaddleOCRPDFReader] No elements found in response for: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[PaddleOCRPDFReader] Error parsing response for {file}: {e}')
            return []

    def _fetch_response(self, file: Path) -> str:
        """Internal HTTP request."""
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

        response = requests.post(self._url, json=payload, headers=self._headers, timeout=600)
        response.raise_for_status()
        return response.text

    def _build_nodes_from_blocks(self, blocks: List[Block], file: Path,
                                  extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []
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

            node = DocNode(text=text, metadata=metadata, global_metadata=extra_info)
            node.excluded_embed_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            node.excluded_llm_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            docs.append(node)

        return docs
