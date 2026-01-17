import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable

from lazyllm import LOG
from ..doc_node import DocNode
from .readerBase import LazyLLMReaderBase


class MineruPDFReader(LazyLLMReaderBase):
    def __init__(self, url, backend='hybrid-auto-engine',
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 upload_mode: bool = False,
                 extract_table: bool = True,
                 extract_formula: bool = True,
                 split_doc: bool = True,
                 clean_content: bool = True,
                 process_blocks_func: Optional[Callable] = None,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        if backend not in ['pipeline', 'vlm-transformers', 'vlm-vllm-async-engine', 'hybrid-auto-engine']:
            raise ValueError(f'Invalid backend: {backend}, \
                             only support pipeline, vlm-transformers, vlm-vllm-async-engine, hybrid-auto-engine')
        self._url = url + '/api/v1/pdf_parse'
        self._drop_types = ['header', 'footer', 'page_number', 'aside_text', 'page_footnote']
        self._upload_mode = upload_mode
        self._backend = backend
        self._extract_table = extract_table
        self._extract_formula = extract_formula
        self._split_doc = split_doc
        self._post_func = post_func
        self._clean_content = clean_content
        self._type_processors = {
            'base': self._process_base,   # base processor for all content types
            'text': self._process_text,
            'image': self._process_image,
            'table': self._process_table,
            'equation': self._process_equation,
            'code': self._process_code,
            'list': self._process_list,
            'default': self._process_default,   # default processor for unknown content types
        }

    def set_type_processor(self, content_type: str, processor: Callable):
        # set custom processor for a specific content type, output should be a dict with the following keys:
        # - text: will be set to content of DocNode
        # - other keys: will be added to metadata of DocNode
        self._type_processors[content_type] = processor

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   use_cache: bool = True, **kwargs) -> List[DocNode]:
        try:
            if isinstance(file, str):
                file = Path(file)
            elements = self._parse_pdf_elements(file, use_cache=use_cache)
            docs = self._build_nodes(elements, file, extra_info)

            if self._post_func:
                docs = self._post_func(docs)
                assert isinstance(docs, list), f'Expected list, got {type(docs)}, please check your post function'
                for node in docs:
                    assert isinstance(node, DocNode), f'Expected DocNode, got {type(node)}, \
                        please check your post function'
                    node.global_metadata = extra_info
            if not docs:
                LOG.warning(f'[MineruPDFReader] No elements found in PDF: {file}')
            return docs
        except Exception as e:
            LOG.error(f'[MineruPDFReader] Error loading data from {file}: {e}')
            return []

    def _parse_pdf_elements(self, pdf_path: Path, use_cache: bool = True) -> List[dict]:
        payload = {'return_content_list': True,
                   'use_cache': use_cache,
                   'backend': self._backend,
                   'table_enable': self._extract_table,
                   'formula_enable': self._extract_formula}
        try:
            if not self._upload_mode:
                payload['files'] = [str(pdf_path)]
                response = requests.post(self._url, data=payload, timeout=3600)
            else:
                with open(pdf_path, 'rb') as f:
                    files = {'upload_files': (os.path.basename(pdf_path), f)}
                    response = requests.post(self._url, data=payload, files=files, timeout=3600)
            if response.status_code != 200:
                LOG.error(f'[MineruPDFReader] POST request failed with status '
                          f'{response.status_code}: {response.text}')
                return []
            res = response.json()
            if not isinstance(res, dict) or not res.get('result'):
                LOG.error(f'[MineruPDFReader] Invalid response: {res}')
                return []
            res = res['result'][0].get('content_list', [])
            if not res:
                LOG.warning(f'[MineruPDFReader] No content list found in response: {pdf_path}')
        except requests.exceptions.RequestException as e:
            LOG.error(f'[MineruPDFReader] POST or parse response failed: {e}')
            return []
        res = self._extract_content_blocks(res)
        return res

    def _extract_content_blocks(self, content_list: List[dict]) -> List[dict]:
        blocks = []
        for content in content_list:
            if self._clean_content and content.get('type') in self._drop_types:
                continue

            content_type = content.get('type', 'default')
            processor = self._type_processors.get(content_type)
            processed_block = processor(content)

            if processed_block is not None:
                blocks.append(processed_block)

        return blocks

    def _process_base(self, content: dict) -> dict:
        block = {
            'bbox': content.get('bbox', []),
            'type': content.get('type', 'text'),
            'page': content.get('page_idx', 0),
            'lines': content.get('lines', [])
        }

        for line in block['lines']:
            if 'content' in line:
                line['content'] = self._normalize_content_recursively(line['content'])

        block['text'] = self._normalize_content_recursively(content.get('text', '')).strip()
        return block

    def _process_text(self, content: dict) -> Optional[dict]:
        block = self._process_base(content)
        block['text'] = self._normalize_content_recursively(content.get('text', '')).strip()
        if not block['text']:
            return None

        if 'text_level' in content:
            block['text_level'] = content['text_level']

        return block

    def _process_image(self, content: dict) -> Optional[dict]:
        block = self._process_base(content)
        img_path = content.get('img_path')
        if not img_path:
            return None

        block['image_path'] = img_path
        block['img_caption'] = self._normalize_and_join(content.get('image_caption', []), separator=' ')
        block['img_footnote'] = self._normalize_and_join(content.get('image_footnote', []))

        block['text'] = f'![{block["img_caption"]}]({block["image_path"]})'
        if block['img_footnote']:
            block['text'] += f'\n{block["img_footnote"]}\n'
        else:
            block['text'] += '\n'

        return block

    def _process_table(self, content) -> Optional[dict]:
        block = self._process_base(content)
        if self._extract_table:
            table_body = content.get('table_body')
            if not table_body:
                return None

            block['table_body'] = self._normalize_content_recursively(table_body)
            block['table_caption'] = self._normalize_and_join(content.get('table_caption', []), separator=' ')
            block['table_footnote'] = self._normalize_and_join(content.get('table_footnote', []))

            text_parts = [t for t in [block['table_caption'], block['table_body'], block['table_footnote']] if t]
            block['text'] = '\n'.join(text_parts).lstrip('\n') + '\n'
        else:
            img_path = content.get('img_path')
            if not img_path:
                return None

            block['image_path'] = img_path
            block['text'] = f'![table]({block["image_path"]})'

        return block

    def _process_equation(self, content: dict) -> Optional[dict]:
        block = self._process_base(content)
        if self._extract_formula:
            text = content.get('text')
            if not text:
                return None
            block['text'] = text
        else:
            img_path = content.get('img_path')
            if not img_path:
                return None
            block['image_path'] = img_path
            block['text'] = f'![formula]({block["image_path"]})'

        return block

    def _process_code(self, content: dict) -> Optional[dict]:
        block = self._process_base(content)
        code_body = content.get('code_body')
        if not code_body:
            return None

        block['text'] = code_body
        block['code_type'] = content.get('sub_type', '')

        code_caption = content.get('code_caption')
        if code_caption:
            normalized_caption = self._normalize_and_join(code_caption)
            block['text'] = f"{normalized_caption}\n{block['text']}"

        return block

    def _process_list(self, content: dict) -> Optional[dict]:
        block = self._process_base(content)
        list_items = content.get('list_items')
        if not list_items:
            return None

        block['list_type'] = content.get('sub_type', '')
        block['text'] = self._normalize_and_join(list_items)

        return block

    def _process_default(self, content: dict) -> dict:
        block = self._process_base(content)
        for k, v in content.items():
            if k not in block:
                block[k] = v
        return block

    def _normalize_and_join(self, content_list: List, separator: str = '\n') -> str:
        if not content_list:
            return ''

        normalized = self._normalize_content_recursively(content_list)
        if isinstance(normalized, list):
            return separator.join(str(item) for item in normalized if item)
        return str(normalized) if normalized else ''

    def _normalize_content_recursively(self, content) -> str:
        if isinstance(content, str):
            content = content.encode('utf-8', 'replace').decode('utf-8')
        if isinstance(content, list):
            return [self._normalize_content_recursively(t) for t in content]
        return content

    def _build_nodes(self, elements: List[dict], file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []
        if self._split_doc:
            for e in elements:
                metadata = {'file_name': file.name}
                metadata.update({k: v for k, v in e.items() if k != 'text'})
                metadata.update({'file_path': str(file)})
                node = DocNode(text=e.get('text', ''), metadata=metadata, global_metadata=extra_info)
                node.excluded_embed_metadata_keys = [type for type in e.keys() if type not in ['file_name', 'text']]
                node.excluded_llm_metadata_keys = [type for type in e.keys() if type not in ['file_name', 'text']]
                docs.append(node)
        else:
            text_chunks = [el['text'] for el in elements if 'text' in el]
            node = DocNode(text='\n'.join(text_chunks), metadata={'file_name': file.name})
            node.excluded_embed_metadata_keys = [type for type in e.keys() if type not in ['file_name', 'text']]
            node.excluded_llm_metadata_keys = [type for type in e.keys() if type not in ['file_name', 'text']]
            docs.append(node)
        return docs
