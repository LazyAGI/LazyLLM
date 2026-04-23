import io
import json
import os
import zipfile

import requests
from pathlib import Path
from typing import Dict, List, Optional, Set
from typing_extensions import override

import lazyllm
from lazyllm import LOG
from lazyllm.tools.http_request import post_sync, get_sync, post_async

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
        try:
            if not self._upload_mode:
                payload['files'] = [str(file)]
                response = post_sync(self._url, payload=payload, timeout=self._timeout)
            else:
                with open(file, 'rb') as f:
                    files = {'upload_files': (os.path.basename(file), f)}
                    response = post_sync(self._url, payload=payload, files=files, timeout=self._timeout)
            return response.text
        except requests.exceptions.RequestException as e:
            LOG.error(f'[MineruPDFReader] POST request failed: {e}')
            raise

    def _fetch_async(self, file: Path, use_cache: bool) -> str:
        base_url = self._url.rstrip('/')
        payload = {
            'return_md': True,
            'return_content_list': True,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
        }

        # Try local mineru-api pattern first, then cloud pattern
        try:
            if not self._upload_mode:
                payload['files'] = [str(file)]
                result = post_async(
                    submit_url=base_url + '/tasks',
                    status_url=base_url + '/tasks/{task_id}',
                    result_url=base_url + '/tasks/{task_id}/result',
                    payload=payload,
                    timeout=self._timeout,
                )
            else:
                with open(file, 'rb') as f:
                    files = {'files': (os.path.basename(file), f)}
                    result = post_async(
                        submit_url=base_url + '/tasks',
                        status_url=base_url + '/tasks/{task_id}',
                        result_url=base_url + '/tasks/{task_id}/result',
                        payload=payload,
                        files=files,
                        timeout=self._timeout,
                    )
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            LOG.warning(f'[MineruPDFReader] Local async failed: {e}, trying cloud endpoint')
            with open(file, 'rb') as f:
                files = {'file': (os.path.basename(file), f)}
                cloud_headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
                result = post_async(
                    submit_url=base_url + '/api/v4/extract/task',
                    status_url=base_url + '/api/v4/extract/task/{task_id}',
                    payload=payload,
                    files=files,
                    headers=cloud_headers,
                    timeout=self._timeout,
                    result_extractor=lambda resp: resp.json().get('data', {}).get('full_zip_url'),
                )

        if isinstance(result, str) and result.startswith('http'):
            zip_resp = get_sync(result, timeout=self._timeout)
            return self._extract_content_from_zip(zip_resp.content)
        if isinstance(result, bytes):
            return self._extract_content_from_zip(result)
        if isinstance(result, requests.Response):
            content_type = result.headers.get('Content-Type', '')
            if 'zip' in content_type:
                return self._extract_content_from_zip(result.content)
            try:
                return json.dumps(result.json())
            except Exception:
                return result.text
        return json.dumps(result)

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
                LOG.warning(f'[MineruPDFReader] No elements found in response for: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[MineruPDFReader] Error parsing response for {file}: {e}')
            return []

    @override
    def _adapt_raw(self, raw: dict) -> List[Block]:
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
            block = self._adapt_one(item)
            if block is not None:
                blocks.append(block)
        return blocks

    def _adapt_one(self, item: dict) -> Optional[Block]:
        ty = item.get('type', '')
        text_level = item.get('text_level', 0) or 0
        text = item.get('text', '')
        page_idx = item.get('page_idx', 0)
        bbox = BBox.from_list(item.get('bbox', []))

        # patch-specific fields (only for offline with lazyllm_patch_applied)
        page_width = page_height = None
        if (self._service_variant == ServiceVariant.OFFLINE and self._lazyllm_patch_applied):
            page_width = item.get('page_width')
            page_height = item.get('page_height')

        if page_width and page_height and not self._page_size:
            self._page_size = (float(page_width), float(page_height))

        page = PageRef(index=page_idx, bbox=bbox)

        if ty in ('text', 'title') and text_level > 0:
            return HeadingBlock(
                page=page, level=int(text_level), text=text,
                anchor=self._make_anchor(text),
            )
        elif ty == 'text':
            return ParagraphBlock(page=page, text=text)
        elif ty == 'image':
            raw_img_path = item.get('img_path')
            img_path: Optional[Path] = None
            if raw_img_path:
                p = Path(raw_img_path)
                if p.is_absolute():
                    try:
                        img_path = p.relative_to(self._image_cache_dir)
                    except ValueError:
                        img_path = Path(p.name)
                else:
                    img_path = p
            return FigureBlock(
                page=page,
                image_path=img_path,
                caption=self._first(item.get('image_caption')),
                footnote=self._first(item.get('image_footnote')),
            )
        elif ty == 'table':
            return TableBlock(
                page=page,
                caption=self._first(item.get('table_caption')),
                footnote=self._first(item.get('table_footnote')),
                cells=self._parse_table_html(item.get('table_body', '')),
                page_range=(page_idx, page_idx),
            )
        elif ty == 'equation':
            return FormulaBlock(page=page, latex=text, inline=False)
        elif ty == 'code':
            return CodeBlock(
                page=page, text=item.get('code_body', ''),
                language=item.get('guess_lang'),
                caption=self._first(item.get('code_caption')),
            )
        elif ty == 'list':
            return ListBlock(
                page=page,
                items=item.get('list_items', []),
                ordered=False,
            )
        elif ty in ('header', 'footer', 'page_number', 'aside_text', 'page_footnote',
                    'discard', 'discarded'):
            return None
        return None

    @staticmethod
    def _make_anchor(text: str) -> str:
        return text.strip().replace(' ', '-').replace('\n', '-')[:64]

    @staticmethod
    def _first(val):
        if isinstance(val, list) and val:
            return val[0]
        if isinstance(val, str):
            return val
        return None

    def _extract_content_from_zip(self, zip_bytes: bytes) -> str:
        """Extract zip to image_cache_dir and return content_list.json as string."""
        if self._image_cache_dir:
            extract_dir = self._image_cache_dir
        else:
            import tempfile
            extract_dir = Path(tempfile.mkdtemp())

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(extract_dir)

        # Prefer *_content_list.json
        for pattern in ('**/*_content_list.json', '**/content_list.json', '**/*.json', '**/*.md'):
            candidates = list(extract_dir.glob(pattern))
            if candidates:
                candidate = candidates[0]
                if candidate.suffix == '.json':
                    with open(candidate, 'r', encoding='utf-8') as f:
                        return json.dumps(json.load(f))
                elif candidate.suffix == '.md':
                    with open(candidate, 'r', encoding='utf-8') as f:
                        return json.dumps([{'type': 'text', 'text': f.read()}])

        raise ValueError('[MineruPDFReader] No parseable content found in zip response')

    def _parse_table_html(self, html_text: str) -> List[Cell]:
        cells: List[Cell] = []
        if not html_text:
            return cells
        try:
            from lazyllm.thirdparty import bs4
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
            LOG.warning(f'[MineruPDFReader] Failed to parse table HTML: {e}')
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
