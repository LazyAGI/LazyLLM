import json
import os
import requests
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Set
from typing_extensions import override

from lazyllm import LOG

from ...doc_node import DocNode
from .ocr_reader_base import _OcrReaderBase
from .ocr_ir import Block, HeadingBlock, TableBlock, FigureBlock, FormulaBlock, CodeBlock, ListBlock
from .ocr_adapter import ServiceVariant, ServiceConfig, ServiceKind, MinerUVariant, AdapterRegistry
from .ocr_postprocess import l1_normalize, l2_associate


class _MineruBackend(str, Enum):
    PIPELINE = 'pipeline'
    VLM_TRANSFORMERS = 'vlm-transformers'
    VLEM_VLLM_ASYNC_ENGINE = 'vlm-vllm-async-engine'
    HYBRID_AUTO_ENGINE = 'hybrid-auto-engine'

    @classmethod
    def _missing_(cls, value):
        supported = ', '.join(f'{m.value!r}' for m in cls)
        raise ValueError(f'Invalid backend: {value!r}, only support: {supported}')


class MineruPDFReader(_OcrReaderBase):
    def __init__(self,
            service_variant: ServiceVariant = 'online',
            apply_lazyllm_patch: bool = False,
            backend: _MineruBackend = 'hybrid-auto-engine',
            upload_mode: bool = False,
            timeout: Optional[int] = None,
            droped_types: Set[str] = Set('header', 'footer', 'page_number', 'aside_text', 'page_footnote'),
            **kwargs):
        super().__init__(droped_types=droped_types, **kwargs)
        self._backend = backend
        self._service_variant = service_variant
        self._upload_mode = upload_mode
        self._timeout = timeout if (timeout is not None and timeout > 0) else None

    @override
    def _load_data(self, file: Path, extra_info: Optional[Dict] = None, use_cache: bool = True) -> List[DocNode]:
        response_json = self._fetch_response(file, use_cache=use_cache)
        return self._from_response(response_json, file, extra_info)

    def _from_response(self, response_json: str, file: Path,
                      extra_info: Optional[Dict] = None) -> List[DocNode]:
        """Parse a MinerU service response (JSON string) into DocNodes.

        This is the primary entry point when the caller has already obtained
        the service response (e.g., from async polling or a downloaded zip).
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
                LOG.warning(f'[MineruPDFReader] No elements found in response for: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[MineruPDFReader] Error parsing response for {file}: {e}')
            return []

    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        """Internal HTTP request."""
        if not self._url:
            raise ValueError('url is required for internal HTTP mode')
        url = self._url.rstrip('/') + '/api/v1/pdf_parse'
        payload = {
            'return_content_list': True,
            'use_cache': use_cache,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
        }
        try:
            if not self._upload_mode:
                payload['files'] = [str(file)]
                response = requests.post(url, data=payload, timeout=self._timeout)
            else:
                with open(file, 'rb') as f:
                    files = {'upload_files': (os.path.basename(file), f)}
                    response = requests.post(url, data=payload, files=files, timeout=self._timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            LOG.error(f'[MineruPDFReader] POST request failed: {e}')
            raise e

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
