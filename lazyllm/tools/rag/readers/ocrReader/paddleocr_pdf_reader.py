import json
import os
import time
import requests
from concurrent.futures import as_completed
from lazyllm.common.threading import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from typing_extensions import override

import lazyllm
from lazyllm.tools.http_request import post_sync, get_sync
from lazyllm.common import retry_transient
from lazyllm import LOG

from ...doc_node import DocNode
from .ocr_ir import (
    Block, BBox, PageRef,
    HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock,
    FigureBlock, CodeBlock,
)
from .ocr_reader_base import _OcrReaderBase, ServiceVariant

lazyllm.config.add('paddle_api_key', str, None, 'PADDLE_API_KEY', description='The API key for PaddleOCR')

JOB_URL = 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs'
DEFAULT_MODEL = 'PaddleOCR-VL-1.6'
MAX_SIZE_MB = 500
MAX_PAGES = 100


class PaddleOCRPDFReader(_OcrReaderBase):
    def __init__(self,
                 url: Optional[str] = None,
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 format_block_content: bool = True,
                 use_layout_detection: bool = True,
                 use_chart_recognition: bool = True,
                 split_doc: bool = True,
                 drop_types: List[str] = None,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True,
                 images_dir: str = None,
                 dropped_types: Optional[Set[str]] = None,
                 **kwargs):
        super().__init__(url=url,
                         dropped_types=drop_types or dropped_types or {
                             'aside_text', 'header', 'footer', 'number', 'header_image', 'seal'},
                         return_trace=return_trace,
                         image_cache_dir=images_dir or os.path.join(
                             lazyllm.config['home'], 'paddleocr_cache'),
                         **kwargs)
        if self._service_variant != ServiceVariant.ONLINE:
            raise ValueError(
                f'PaddleOCRPDFReader only supports service_variant="online", '
                f'got {self._service_variant.value!r}')
        self._api_key = lazyllm.config['paddle_api_key']
        self._model = kwargs.pop('model', DEFAULT_MODEL)
        self._timeout = kwargs.pop('timeout', None)

    @override
    def _load_data(self, file, extra_info: Optional[Dict] = None, **kwargs
                   ) -> List['DocNode']:
        file_path = Path(file)
        _t0 = time.time()
        response_text, task_dir = self._fetch_async(file_path)
        _t_fetch = time.time() - _t0
        merged_info = dict(extra_info) if extra_info else {}
        if task_dir is not None:
            merged_info['image_cache_dir'] = str(task_dir)
        _t1 = time.time()
        nodes = self._build_nodes_from_response(response_text, file_path, merged_info)
        _t_build = time.time() - _t1
        LOG.info(f'[BENCHMARK] file={file_path.name} phase=fetch elapsed={_t_fetch:.3f}s')
        LOG.info(f'[BENCHMARK] file={file_path.name} phase=parse elapsed={_t_build:.3f}s')
        return nodes

    def _fetch_async(self, file):
        file_str = str(file)
        splits = self._split_large_pdf(file_str, max_size_mb=MAX_SIZE_MB, max_pages=MAX_PAGES)

        if len(splits) == 1:
            return retry_transient(
                self._fetch_job,
                log_prefix=f'[PaddleOCRPDFReader] {os.path.basename(file_str)} ')(
                    splits[0][0])

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(splits), 5)) as executor:
            futures = {
                executor.submit(
                    retry_transient(
                        self._fetch_job,
                        log_prefix=f'[PaddleOCRPDFReader] {os.path.basename(sub_path)} '),
                    sub_path,
                ): start_page
                for sub_path, start_page in splits
            }
            for future in as_completed(futures):
                start_page = futures[future]
                results[start_page] = future.result()

        return self._merge_split_results(results)

    def _fetch_job(self, file_path: str):
        fname = os.path.basename(file_path)
        headers = {'Authorization': f'bearer {self._api_key or ""}'}

        optional_payload = {
            'useDocOrientationClassify': False,
            'useDocUnwarping': False,
            'useChartRecognition': True,
        }

        data = {
            'model': self._model,
            'optionalPayload': json.dumps(optional_payload),
        }
        with open(file_path, 'rb') as f:
            resp = post_sync(
                JOB_URL,
                payload=data,
                files={'file': (fname, f)},
                headers=headers,
                timeout=self._timeout or 120,
            )
        job_data = resp.json()
        job_id = job_data['data']['jobId']
        LOG.info(f'[PaddleOCRPDFReader] Job submitted: {job_id} for {fname}')

        _t_poll = time.time()
        for _ in range(240):
            status_resp = get_sync(
                f'{JOB_URL}/{job_id}',
                headers=headers,
                timeout=self._timeout or 30,
            )
            status_data = status_resp.json()
            state = status_data['data']['state']

            if state == 'done':
                jsonl_url = status_data['data']['resultUrl']['jsonUrl']
                _t_wait = time.time() - _t_poll
                LOG.info(f'[BENCHMARK] file={fname} phase=wait elapsed={_t_wait:.3f}s')

                jsonl_resp = get_sync(jsonl_url, timeout=self._timeout or 120)
                merged = self._merge_jsonl_lines(jsonl_resp.text)
                return merged, None

            elif state == 'failed':
                error_msg = status_data['data'].get('errorMsg', 'Unknown error')
                raise RuntimeError(
                    f'[PaddleOCRPDFReader] Job {job_id} failed: {error_msg}')

            elif state == 'running':
                try:
                    progress = status_data['data']['extractProgress']
                    total = progress.get('totalPages', '?')
                    extracted = progress.get('extractedPages', '?')
                    LOG.debug(f'[PaddleOCRPDFReader] Job {job_id}: {extracted}/{total} pages')
                except KeyError:
                    pass

            time.sleep(5)

        raise TimeoutError(f'[PaddleOCRPDFReader] Job {job_id} polling timed out')

    @staticmethod
    def _merge_jsonl_lines(jsonl_text: str) -> str:
        lines = jsonl_text.strip().split('\n')
        all_results: List[Dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for entry in obj['result']['layoutParsingResults']:
                all_results.append(entry)
        merged = {'result': {'layoutParsingResults': all_results}}
        return json.dumps(merged)

    def _merge_split_results(self, results: dict):
        sorted_pages = sorted(results.keys())
        all_results: List[Dict] = []

        for start_page in sorted_pages:
            json_str, _ = results[start_page]
            data = json.loads(json_str)
            for entry in data['result']['layoutParsingResults']:
                all_results.append(entry)

        merged = {'result': {'layoutParsingResults': all_results}}
        return json.dumps(merged), None

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
        if label in self._dropped_types:
            return None

        content = item['block_content']
        bbox = BBox.from_list(item['block_bbox'])
        page = PageRef(index=page_idx, bbox=bbox)

        if label in ('paragraph_title', 'doc_title'):
            return HeadingBlock(page=page, text=content)
        elif label == 'image':
            img_src, img_url = self._resolve_image(item['block_bbox'], markdown_images)
            if img_src is None:
                return None
            rel_path = Path(img_src).as_posix().removeprefix('./')
            save_path = self._image_cache_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_tasks.append((img_url, save_path))
            return FigureBlock(page=page, image_path=Path(rel_path))
        elif label in ('table', 'chart'):
            return TableBlock(
                page=page,
                cells=self._parse_table_html(content),
                page_range=(page_idx, page_idx),
            )
        elif label == 'display_formula':
            return FormulaBlock(page=page, latex=content, inline=False)
        elif label == 'inline_formula':
            return FormulaBlock(page=page, latex=content, inline=True)
        elif label in ('code', 'algorithm'):
            return CodeBlock(page=page, text=content)
        else:
            return ParagraphBlock(page=page, text=content)

    @staticmethod
    def _resolve_image(block_bbox: List[float],
                       markdown_images: Dict[str, str]) -> Optional[tuple]:
        '''Resolve image src and URL from markdown_images by matching bbox in key.

        Image keys follow the pattern: imgs/img_in_image_box_{x0}_{y0}_{x1}_{y1}.jpg
        '''
        if not markdown_images:
            LOG.warning('[PaddleOCRPDFReader] image block has no markdown_images, skipped')
            return None

        if len(markdown_images) == 1:
            key = next(iter(markdown_images))
            return key, markdown_images[key]

        x0, y0, x1, y1 = int(block_bbox[0]), int(block_bbox[1]), int(block_bbox[2]), int(block_bbox[3])
        bbox_suffix = f'{x0}_{y0}_{x1}_{y1}'
        for key, url in markdown_images.items():
            if bbox_suffix in key:
                return key, url

        key = next(iter(markdown_images))
        LOG.warning(f'[PaddleOCRPDFReader] could not match image by bbox {bbox_suffix}, '
                    f'using first: {key}')
        return key, markdown_images[key]

    @staticmethod
    def _download_images(image_tasks: List[tuple]) -> None:
        def _download_one(task: tuple) -> None:
            img_url, save_path = task
            resp = requests.get(img_url, timeout=120)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_download_one, image_tasks))

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
