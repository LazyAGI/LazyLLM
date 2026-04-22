import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set
from typing_extensions import override

import requests

from lazyllm import LOG

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef, Cell,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_postprocessor import l1_normalize, l2_associate
from .ocr_reader_base import _OcrReaderBase, _Adapter, ServiceVariant


class MineruPDFReader(_OcrReaderBase, _Adapter):
    def __init__(self,
            apply_lazyllm_patch: bool = False,
            backend: str = 'hybrid-auto-engine',
            upload_mode: bool = False,
            timeout: Optional[int] = None,
            droped_types: Set[str] = {'header', 'footer', 'page_number', 'aside_text', 'page_footnote'},
            **kwargs):
        super().__init__(droped_types=droped_types, **kwargs)
        self._backend = backend
        self._upload_mode = upload_mode
        self._timeout = timeout if (timeout is not None and timeout > 0) else None
        self._apply_lazyllm_patch = apply_lazyllm_patch
        self._page_size = None

    @override
    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        if self._service_variant == ServiceVariant.OFFLINE:
            return self._fetch_sync(file, use_cache)
        return self._fetch_async(file, use_cache)

    def _fetch_sync(self, file: Path, use_cache: bool) -> str:
        if not self._url:
            raise ValueError('url is required for MineruPDFReader')
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
            raise

    def _fetch_async(self, file: Path, use_cache: bool) -> str:
        if not self._url:
            raise ValueError('url is required for MineruPDFReader')
        base_url = self._url.rstrip('/')

        # Try local mineru-api pattern first, then cloud pattern
        task_id = self._submit_task(base_url, file, use_cache)
        if not task_id:
            raise RuntimeError('[MineruPDFReader] Failed to submit async task')

        LOG.info(f'[MineruPDFReader] Task {task_id} submitted, polling...')
        result = self._poll_task(base_url, task_id)
        if not result:
            raise RuntimeError(f'[MineruPDFReader] Task {task_id} failed or timed out')

        # result may be a JSON string, a zip bytes, or a full_zip_url
        if isinstance(result, str):
            # full_zip_url from cloud API
            if result.startswith('http'):
                zip_resp = requests.get(result, timeout=self._timeout)
                zip_resp.raise_for_status()
                return self._extract_content_from_zip(zip_resp.content)
            # Already JSON string
            return result
        if isinstance(result, bytes):
            return self._extract_content_from_zip(result)
        # dict or list - serialize to JSON string
        return json.dumps(result)

    def _submit_task(self, base_url: str, file: Path, use_cache: bool) -> Optional[str]:
        # Local mineru-api style
        submit_url = base_url + '/tasks'
        payload = {
            'return_md': True,
            'return_content_list': True,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
        }
        try:
            if not self._upload_mode:
                payload['files'] = [str(file)]
                resp = requests.post(submit_url, data=payload, timeout=self._timeout)
            else:
                with open(file, 'rb') as f:
                    files = {'files': (os.path.basename(file), f)}
                    resp = requests.post(submit_url, data=payload, files=files, timeout=self._timeout)
            if resp.status_code == 404:
                return self._submit_task_cloud(base_url, file, use_cache)
            resp.raise_for_status()
            data = resp.json()
            return data.get('task_id')
        except requests.exceptions.RequestException as e:
            LOG.warning(f'[MineruPDFReader] Local async submit failed: {e}, trying cloud endpoint')
            return self._submit_task_cloud(base_url, file, use_cache)

    def _submit_task_cloud(self, base_url: str, file: Path, use_cache: bool) -> Optional[str]:
        submit_url = base_url + '/api/v4/extract/task'
        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
        try:
            with open(file, 'rb') as f:
                files = {'file': (os.path.basename(file), f)}
                resp = requests.post(submit_url, files=files, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get('data', {}).get('task_id')
        except requests.exceptions.RequestException as e:
            LOG.error(f'[MineruPDFReader] Cloud async submit failed: {e}')
            return None

    def _poll_task(self, base_url: str, task_id: str, max_retries: int = 120, interval: int = 3):
        # Try local status endpoint first
        status_url = base_url + f'/tasks/{task_id}'
        for _ in range(max_retries):
            try:
                resp = requests.get(status_url, timeout=self._timeout)
                if resp.status_code == 404:
                    return self._poll_task_cloud(base_url, task_id, max_retries, interval)
                resp.raise_for_status()
                data = resp.json()
                status = data.get('status', data.get('state', ''))
                if status in ('completed', 'done', 'success'):
                    # Try to get result directly
                    result_url = base_url + f'/tasks/{task_id}/result'
                    result_resp = requests.get(result_url, timeout=self._timeout)
                    result_resp.raise_for_status()
                    content_type = result_resp.headers.get('Content-Type', '')
                    if 'zip' in content_type:
                        return result_resp.content
                    try:
                        return result_resp.json()
                    except Exception:
                        return result_resp.text
                if status in ('failed', 'error', 'failure'):
                    LOG.error(f'[MineruPDFReader] Task {task_id} failed: {data}')
                    return None
            except requests.exceptions.RequestException as e:
                LOG.warning(f'[MineruPDFReader] Poll error: {e}')
            time.sleep(interval)
        LOG.error(f'[MineruPDFReader] Task {task_id} polling timed out')
        return None

    def _poll_task_cloud(self, base_url: str, task_id: str, max_retries: int = 120, interval: int = 3):
        status_url = base_url + f'/api/v4/extract/task/{task_id}'
        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}
        for _ in range(max_retries):
            try:
                resp = requests.get(status_url, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json().get('data', {})
                state = data.get('state', '')
                if state == 'done':
                    return data.get('full_zip_url')
                if state == 'failed':
                    LOG.error(f'[MineruPDFReader] Cloud task {task_id} failed: {data.get("err_msg")}')
                    return None
            except requests.exceptions.RequestException as e:
                LOG.warning(f'[MineruPDFReader] Cloud poll error: {e}')
            time.sleep(interval)
        LOG.error(f'[MineruPDFReader] Cloud task {task_id} polling timed out')
        return None

    def _extract_content_from_zip(self, zip_bytes: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Prefer content_list.json
            for name in zf.namelist():
                if 'content_list.json' in name:
                    return zf.read(name).decode('utf-8')
            # Fallback to any json
            for name in zf.namelist():
                if name.endswith('.json'):
                    return zf.read(name).decode('utf-8')
            # Fallback to first md file
            for name in zf.namelist():
                if name.endswith('.md'):
                    return json.dumps([{'type': 'text', 'text': zf.read(name).decode('utf-8')}])
        raise ValueError('[MineruPDFReader] No parseable content found in zip response')

    @override
    def _from_response(self, response_json: str, file: Path,
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

        # patch-specific fields (only for offline with apply_lazyllm_patch)
        page_width = page_height = None
        if (
            self._service_variant == ServiceVariant.OFFLINE
            and self._apply_lazyllm_patch
        ):
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
            return FigureBlock(
                page=page,
                image_path=Path(img_path) if (img_path := item.get('img_path')) else None,
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
