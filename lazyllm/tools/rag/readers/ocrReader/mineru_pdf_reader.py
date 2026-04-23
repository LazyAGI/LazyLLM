import io
import json
import os
import zipfile

from pathlib import Path
from typing import Dict, List, Optional, Set
from typing_extensions import override

import lazyllm
from lazyllm import LOG
from lazyllm.tools.http_request import post_sync, post_async

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef, Cell,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_postprocessor import l1_normalize, l2_associate
from .ocr_reader_base import _OcrReaderBase, _Adapter, ServiceVariant

lazyllm.config.add('mineru_api_key', str, None, 'MINERU_API_KEY', description='The API key for Mineru')

class MineruPDFReader(_OcrReaderBase, _Adapter):
    def __init__(self,
            url: str = 'https://mineru.net/api/v4/extract/task',
            lazyllm_patch_applied: bool = False,
            backend: str = 'hybrid-auto-engine',
            upload_mode: bool = False,
            timeout: Optional[int] = None,
            droped_types: Set[str] = {'header', 'footer', 'page_number', 'aside_text', 'page_footnote'},
            **kwargs):
        super().__init__(url=url, droped_types=droped_types, **kwargs)
        self._backend = backend
        self._upload_mode = upload_mode
        self._timeout = timeout if (timeout is not None and timeout > 0) else None
        self._lazyllm_patch_applied = lazyllm_patch_applied
        self._api_key = lazyllm.config['mineru_api_key']

    @override
    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        if self._service_variant == ServiceVariant.OFFLINE:
            return self._fetch_sync(file, use_cache)
        return self._fetch_async(file, use_cache)

    def _fetch_sync(self, file: Path, use_cache: bool) -> str:
        payload = {
            'return_content_list': True,
            'use_cache': use_cache,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
        }
        if not self._upload_mode:
            payload['files'] = [str(file)]
            response = post_sync(self._url, payload=payload, timeout=self._timeout)
        else:
            with open(file, 'rb') as f:
                files = {'upload_files': (os.path.basename(file), f)}
                response = post_sync(self._url, payload=payload, files=files, timeout=self._timeout)
        return response.text

    def _fetch_async(self, file: Path, use_cache: bool) -> str:
        base_url = self._url.rstrip('/')
        payload = {
            'return_md': True,
            'return_content_list': True,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
        }

        files_payload = None
        file_obj = None
        if self._upload_mode:
            file_obj = open(file, 'rb')
            files_payload = {'files': (os.path.basename(file), file_obj)}
        else:
            payload['files'] = [str(file)]

        result = post_async(
            submit_url=base_url + '/tasks',
            status_url=base_url + '/tasks/{task_id}',
            result_url=base_url + '/tasks/{task_id}/result',
            payload=payload,
            files=files_payload,
            headers={'Authorization': f'Bearer {self._api_key}'},
            timeout=self._timeout,
        )
        if file_obj:
            file_obj.close()

        return self._extract_content_from_zip(result.content)

    def _extract_content_from_zip(self, zip_bytes: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(self._image_cache_dir)

        with open(self._image_cache_dir.glob('auto/*_content_list.json'), 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f))

    @override
    def _build_nodes_from_response(self, response_json: str, file: Path,
            extra_info: Optional[Dict] = None) -> List[DocNode]:
        raw = json.loads(response_json)
        blocks = self._adapt_json_to_IR(raw)
        # Post processing
        blocks = l1_normalize(blocks, self._page_size)
        blocks = l2_associate(blocks)
        return self._build_nodes_from_blocks(blocks, file, extra_info)

    @override
    def _adapt_json_to_IR(self, raw: dict) -> List[Block]:
        content_list = raw['content_list']
        blocks: List[Block] = []
        for item in content_list:
            block = self._adapt_one(item)
            if block is not None:
                blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict) -> Optional[Block]:
        ty = item['type']
        if ty in self._droped_types:
            return None

        text_level = item.get('text_level', 0)
        text = item['text']
        page_idx = item['page_idx']
        bbox = BBox.from_list(item['bbox'])

        page = PageRef(index=page_idx, bbox=bbox)

        if ty in ('text', 'title') and text_level > 0:
            return HeadingBlock(
                page=page, level=int(text_level), text=text,
                anchor=self._make_anchor(text),
            )
        elif ty == 'text':
            return ParagraphBlock(page=page, text=text)
        elif ty == 'image':
            return FigureBlock(
                page=page,
                image_path=Path(item['img_path']),
                caption=self._first(item.get('image_caption')),
                footnote=self._first(item.get('image_footnote')),
            )
        elif ty == 'table':
            return TableBlock(
                page=page,
                caption=self._first(item.get('table_caption')),
                footnote=self._first(item.get('table_footnote')),
                cells=self._parse_table_html(item['table_body']),
                page_range=(page_idx, page_idx),
            )
        elif ty == 'equation':
            return FormulaBlock(page=page, latex=text, inline=False)
        elif ty == 'code':
            return CodeBlock(
                page=page, text=item['code_body'],
                language=item.get('guess_lang'),
                caption=self._first(item.get('code_caption')),
            )
        elif ty == 'list':
            return ListBlock(
                page=page,
                items=item['list_items'],
                ordered=False,
            )
        return None

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
