import io
import json
import os
import uuid
import zipfile
import requests
import time

from concurrent.futures import as_completed
from lazyllm.common.threading import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from typing_extensions import override

import lazyllm
from lazyllm.common import retry_transient
from lazyllm.tools.http_request import post_sync, get_sync
from lazyllm import LOG

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock, ListBlock,
)
from .ocr_reader_base import _OcrReaderBase, ServiceVariant

lazyllm.config.add('mineru_api_key', str, None, 'MINERU_API_KEY', description='The API key for Mineru')


class MineruPDFReader(_OcrReaderBase):
    def __init__(self,
                 url: Optional[str] = None,
                 backend: str = 'hybrid-auto-engine',
                 upload_mode: bool = False,
                 extract_table: bool = True,
                 extract_formula: bool = True,
                 split_doc: bool = True,
                 clean_content: bool = True,
                 timeout: Optional[int] = None,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True,
                 dropped_types: Optional[Set[str]] = None,
                 patch_applied: bool = False,
                 **kwargs):
        super().__init__(url=url,
            dropped_types=dropped_types or {
            'header', 'footer', 'page_number', 'aside_text', 'page_footnote'},
            return_trace=return_trace,
            post_func=post_func,
            image_cache_dir=kwargs.pop('image_cache_dir', os.path.join(
                lazyllm.config['home'], 'mineru_cache')),
            **kwargs)
        self._backend = backend
        self._upload_mode = upload_mode
        self._timeout = timeout if (timeout is not None and timeout > 0) else None
        self._patch_applied = patch_applied
        self._api_key = lazyllm.config['mineru_api_key']

    @override
    def _load_data(self, file, extra_info: Optional[Dict] = None, use_cache: bool = True
                   ) -> List['DocNode']:
        file_path = Path(file)
        if self._service_variant == ServiceVariant.OFFLINE:
            response_text = self._fetch_sync(file_path, use_cache)
            task_dir = None
        else:
            response_text, task_dir = self._fetch_async(file_path, use_cache)
        merged_info = dict(extra_info) if extra_info else {}
        if task_dir is not None:
            merged_info['image_cache_dir'] = str(task_dir)
        return self._build_nodes_from_response(response_text, file_path, merged_info)

    def _fetch_sync(self, file: Path, use_cache: bool) -> str:
        if self._patch_applied:
            # Patch-deployed local server: form-encoded.
            payload = {
                'return_content_list': 'true',
                'use_cache': 'false' if not use_cache else 'true',
                'backend': self._backend,
                'table_enable': 'true',
                'formula_enable': 'true',
            }
            if not self._upload_mode:
                payload['files'] = str(file)
                response = post_sync(self._url, payload=payload, timeout=self._timeout)
            else:
                with open(file, 'rb') as f:
                    files = {'upload_files': (os.path.basename(file), f)}
                    response = post_sync(self._url, payload=payload, files=files, timeout=self._timeout)
            return response.text

        # Original local server: JSON payload.
        payload = {
            'return_content_list': True,
            'use_cache': use_cache,
            'backend': self._backend,
            'table_enable': True,
            'formula_enable': True,
            'files': [str(file)],
        }
        response = post_sync(self._url, json_payload=payload, timeout=self._timeout)
        return response.text

    def _fetch_async(self, file, use_cache: bool = True):
        file_str = str(file)
        splits = self._split_large_pdf(file_str)
        task_dir = self._image_cache_dir / str(uuid.uuid4())

        if len(splits) == 1:
            return retry_transient(
                self._fetch_async_by_upload,
                log_prefix=f'[MineruPDFReader] {os.path.basename(file_str)} ')(
                    splits[0][0], task_dir=task_dir)

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(splits), 5)) as executor:
            futures = {
                executor.submit(
                    retry_transient(
                        self._fetch_async_by_upload,
                        log_prefix=f'[MineruPDFReader] {os.path.basename(sub_path)} '),
                    sub_path, task_dir=task_dir,
                ): start_page
                for sub_path, start_page in splits
            }
            for future in as_completed(futures):
                start_page = futures[future]
                results[start_page] = future.result()

        return self._merge_split_results(results)

    def _merge_split_results(self, results: dict):
        sorted_pages = sorted(results.keys())
        all_content = []
        first_task_dir = None

        for start_page in sorted_pages:
            json_str, task_dir = results[start_page]
            if first_task_dir is None:
                first_task_dir = task_dir
            content = json.loads(json_str)
            items = content

            for item in items:
                if 'page_idx' in item:
                    item['page_idx'] += start_page
                all_content.append(item)

        merged_json = json.dumps(all_content)

        return merged_json, first_task_dir

    def _fetch_async_by_upload(self, file_path: str, task_dir: Optional['Path'] = None):
        '''Upload a local file via batch presigned URL and fetch result.'''
        fname = os.path.basename(file_path)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self._api_key}'}

        # Step 1: Request presigned upload URL
        payload = {
            'files': [{'name': fname}],
            'model_version': 'vlm',
        }
        resp = post_sync(
            'https://mineru.net/api/v4/file-urls/batch',
            json_payload=payload,
            headers=headers,
            timeout=self._timeout,
        )
        data = resp.json()
        batch_id = data['data']['batch_id']
        file_url = data['data']['file_urls'][0]

        # Step 2: Upload file to OSS
        with open(file_path, 'rb') as f:
            upload_resp = requests.put(file_url, data=f, timeout=self._timeout or 300)
            upload_resp.raise_for_status()

        # Step 3: Poll batch results
        status_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
        for _ in range(120):
            status_resp = requests.get(status_url, headers=headers, timeout=self._timeout or 30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            extract_result = status_data.get('data', {}).get('extract_result', [])
            if extract_result:
                state = extract_result[0].get('state')
                if state == 'done':
                    full_zip_url = extract_result[0].get('full_zip_url')
                    zip_resp = requests.get(full_zip_url, timeout=self._timeout or 120)
                    zip_resp.raise_for_status()
                    return self._extract_content_from_zip(zip_resp.content, task_dir=task_dir)
                elif state == 'failed':
                    raise RuntimeError(
                        f'[MineruPDFReader] Batch task failed: '
                        f'{extract_result[0].get("err_msg", "Unknown error")}')
            time.sleep(3)

        raise TimeoutError('[MineruPDFReader] Batch polling timed out')

    def _extract_content_from_zip(self, zip_bytes: bytes, task_dir: Optional['Path'] = None):
        if task_dir is None:
            task_dir = self._image_cache_dir / str(uuid.uuid4())
        task_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or '..' in member_path.parts:
                    raise ValueError(f'Path traversal detected in zip: {member.filename}')
            json_members = [m for m in zf.infolist() if m.filename.endswith('_content_list.json')]
            if not json_members:
                raise ValueError('No *_content_list.json found in zip')
            content = json.loads(zf.read(json_members[0]))
            for member in zf.infolist():
                if not member.filename.endswith('_content_list.json'):
                    zf.extract(member, task_dir)
        return json.dumps(content), task_dir

    @override
    def _adapt_json_to_IR(self, raw) -> List[Block]:
        # Online API (zip extraction) returns a list directly.
        # Patch-deployed local server returns {'result': [{'content_list': [...]}]}.
        # Prefer structural detection over the patch_applied flag so that
        # online-API responses are handled correctly regardless of configuration.
        if self._patch_applied and isinstance(raw, dict):
            content_list = raw['result'][0]['content_list']
        else:
            content_list = raw

        blocks: List[Block] = []
        for item in content_list:
            block = self._adapt_one(item)
            if block is not None:
                if self._service_variant == ServiceVariant.OFFLINE \
                   and self._patch_applied and 'lines' in item:
                    block.lines = self._normalize_content(item['lines'])
                blocks.append(block)
        return blocks

    def _normalize_content(self, content) -> List:
        if isinstance(content, str):
            return [content.encode('utf-8', 'replace').decode('utf-8')]
        elif isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, str):
                    result.append(item.encode('utf-8', 'replace').decode('utf-8'))
                elif isinstance(item, dict):
                    normalized = dict(item)
                    if 'content' in normalized and isinstance(normalized['content'], str):
                        normalized['content'] = normalized['content'].encode(
                            'utf-8', 'replace').decode('utf-8')
                    result.append(normalized)
                else:
                    result.append(item)
            return result
        raise TypeError(f'Not supported type: {type(content)}.')

    def _adapt_one(self, item: dict) -> Optional[Block]:
        ty = item['type']
        if ty in self._dropped_types:
            return None

        text_level = item.get('text_level', -1)
        text = item.get('text', '')
        page_idx = item['page_idx']
        page = PageRef(index=page_idx, bbox=BBox.from_list(item['bbox']))

        if ty == 'title':
            return HeadingBlock(page=page, level=text_level, text=text)
        elif ty in ('text', 'ref_text', 'phonetic'):
            return ParagraphBlock(page=page, text=text)
        elif ty == 'image':
            return self._adapt_image(item, page, page_idx)
        elif ty == 'table':
            return self._adapt_table(item, page, page_idx)
        elif ty == 'equation':
            return FormulaBlock(page=page, latex=text, inline=False)
        elif ty == 'code':
            return self._adapt_code(item, page, page_idx)
        elif ty == 'list':
            return self._adapt_list(item, page, page_idx)
        return None

    def _adapt_image(self, item: dict, page: PageRef, page_idx: int) -> Optional[Block]:
        img_path = item.get('img_path')
        if img_path is None:
            LOG.warning(f'[MineruPDFReader] image block on page {page_idx} missing img_path, skipped')
            return None
        return FigureBlock(
            page=page,
            image_path=Path(img_path),
            caption=self._first(item.get('image_caption')),
            footnote=self._first(item.get('image_footnote')),
        )

    def _adapt_table(self, item: dict, page: PageRef, page_idx: int) -> TableBlock:
        table_body = item.get('table_body')
        if table_body is None:
            LOG.warning(f'[MineruPDFReader] table block on page {page_idx} missing table_body, '
                        f'caption={self._first(item.get("table_caption"))}')
        return TableBlock(
            page=page,
            caption=self._first(item.get('table_caption')),
            footnote=self._first(item.get('table_footnote')),
            cells=self._parse_table_html(table_body or ''),
            page_range=(page_idx, page_idx),
        )

    def _adapt_code(self, item: dict, page: PageRef, page_idx: int) -> Optional[Block]:
        code_body = item.get('code_body')
        if code_body is None:
            LOG.warning(f'[MineruPDFReader] code block on page {page_idx} missing code_body, skipped')
            return None
        return CodeBlock(
            page=page, text=code_body,
            language=item.get('guess_lang'),
            caption=self._first(item.get('code_caption')),
        )

    def _adapt_list(self, item: dict, page: PageRef, page_idx: int) -> Optional[Block]:
        list_items = item.get('list_items')
        if list_items is None:
            LOG.warning(f'[MineruPDFReader] list block on page {page_idx} missing list_items, skipped')
            return None
        return ListBlock(page=page, items=list_items, ordered=False)

    @override
    def _build_nodes_from_blocks(self, blocks: List[Block], file,
                                 extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []

        global_metadata = dict(extra_info) if extra_info else {}
        # image_cache_dir is injected into extra_info by _load_data for async requests
        if 'image_cache_dir' not in global_metadata:
            global_metadata['image_cache_dir'] = str(self._image_cache_dir)

        file_name = Path(file).name if not isinstance(file, str) else Path(file).name
        file_path = str(file)

        for b in blocks:
            text = b.text_content()
            metadata = {
                'file_name': file_name,
                'file_path': file_path,
                'type': b.ty,
                'page': b.page.index,
                'bbox': b.page.bbox.to_list(),
                'section_path': b.section.anchors,
            }
            b.update_metadata(metadata)
            if b.lines:
                metadata['lines'] = b.lines
            node = DocNode(text=text, metadata=metadata, global_metadata=global_metadata)
            node.excluded_embed_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            node.excluded_llm_metadata_keys = [k for k in metadata if k not in ('file_name', 'text')]
            docs.append(node)

        return docs


def post_async(submit_url: str, status_url: str, result_url: str = None,
               payload: dict = None, files: dict = None, headers: dict = None,
               timeout: int = 60,
               success_states: tuple = ('completed', 'done', 'success'),
               failure_states: tuple = ('failed', 'error', 'failure'),
               max_retries: int = 120, interval: int = 3,
               total_timeout: Optional[int] = None,
               result_extractor: Optional[Callable[[requests.Response], any]] = None,
               json_payload: dict = None) -> any:
    '''Submit an async task, poll status, and fetch the final result.

    Args:
        submit_url: URL to submit the task.
        status_url: Status polling URL containing ``{task_id}`` placeholder.
        result_url: Optional result URL containing ``{task_id}`` placeholder.
        result_extractor: Optional callable to extract result from the status
            response when ``result_url`` is not provided.
        json_payload: Optional JSON payload for the submit request (preferred
            over ``payload`` for APIs that expect ``application/json``).
    '''
    resp = post_sync(submit_url, payload=payload, files=files, headers=headers,
                     json_payload=json_payload, timeout=timeout, raise_for_status=False)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f'[HttpRequest] Unexpected response format: {data}')
    task_id = data.get('task_id')
    if not task_id and 'data' in data:
        task_id = data['data'].get('task_id')
    if not task_id:
        raise ValueError(f'[HttpRequest] No task_id in submit response: {data}')

    deadline = time.time() + total_timeout if total_timeout else None
    for _ in range(max_retries):
        if deadline and time.time() > deadline:
            raise TimeoutError(f'[HttpRequest] Task polling timed out after {total_timeout}s')
        status_resp = get_sync(status_url.format(task_id=task_id), headers=headers,
                               timeout=timeout, raise_for_status=False)
        if status_resp.status_code == 404:
            LOG.error(f'[HttpRequest] Status endpoint 404: {status_url.format(task_id=task_id)}')
            raise RuntimeError(f'[HttpRequest] Status endpoint 404: {status_url.format(task_id=task_id)}')
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get('status', status_data.get('state', ''))
        if 'data' in status_data and isinstance(status_data['data'], dict):
            status = status_data['data'].get('state', status)
        if status in success_states:
            if result_url:
                return get_sync(result_url.format(task_id=task_id), headers=headers, timeout=timeout)
            if result_extractor:
                return result_extractor(status_resp)
            return status_resp
        if status in failure_states:
            raise RuntimeError(f'[HttpRequest] Task failed: {status_data}')
        time.sleep(interval)

    raise TimeoutError(f'[HttpRequest] Task polling timed out after {max_retries * interval}s')
