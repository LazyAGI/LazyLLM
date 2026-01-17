import os
import requests
from pathlib import Path
from lazyllm.thirdparty import bs4
import copy
from typing import Dict, List, Optional, Callable
import unicodedata

from lazyllm import LOG
from ..doc_node import DocNode
from .pdfReader import _RichPDFReader


class MineruPDFReader(_RichPDFReader):
    def __init__(self, url, backend='pipeline',
                 callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 upload_mode: bool = False,
                 extract_table: bool = True,
                 extract_formula: bool = True,
                 split_doc: bool = True,
                 clean_content: bool = True,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True):
        super().__init__(post_func=post_func, return_trace=return_trace)
        self._url = url + '/api/v1/pdf_parse'
        self._drop_types = ['header', 'footer', 'page_number', 'aside_text', 'page_footnote']
        self._upload_mode = upload_mode
        self._backend = backend
        self._extract_table = extract_table
        self._extract_formula = extract_formula
        self._split_doc = split_doc
        self._clean_content = clean_content

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   use_cache: bool = True, **kwargs) -> List[DocNode]:
        try:
            if isinstance(file, str):
                file = Path(file)
            elements = self._parse_pdf_elements(file, use_cache=use_cache)
            docs = self._build_nodes(elements, file, extra_info)

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
                response = requests.post(self._url, data=payload, timeout=600)
            else:
                with open(pdf_path, 'rb') as f:
                    files = {'upload_files': (os.path.basename(pdf_path), f)}
                    response = requests.post(self._url, data=payload, files=files, timeout=600)
            if response.status_code != 200:
                LOG.error(f'[MineruPDFReader] POST request failed with status '
                          f'{response.status_code}: {response.text}')
            response.raise_for_status()
            res = response.json()
            if not isinstance(res, dict) or not res.get('result'):
                LOG.error(f'[MineruPDFReader] Invalid response: {res}')
                return []
            res = res['result'][0].get('content_list', [])
            if not res:
                LOG.warning(f'[MineruPDFReader] No elements found in PDF: {pdf_path}')
                return []
        except requests.exceptions.RequestException as e:
            LOG.error(f'[MineruPDFReader] POST failed: {e}')
            return []
        res = self._extract_content_blocks(res)
        return res

    def _extract_content_blocks(self, content_list) -> List[dict]:  # noqa: C901
        blocks = []
        cur_title = ''
        cur_level = -1
        for content in content_list:
            if self._clean_content and content.get('type') in self._drop_types:
                continue
            block = {}
            block['bbox'] = content.get('bbox', [])
            block['type'] = content.get('type', 'text')
            block['page'] = content.get('page_idx', 0)
            block['lines'] = content['lines'] if 'lines' in content else []
            for line in block['lines']:
                if 'content' in line:
                    line['content'] = self._normalize_content_recursively(line['content'])
            if content['type'] == 'text':
                block['text'] = self._normalize_content_recursively(content['text']).strip()
                if not block['text']:
                    continue
                if 'text_level' in content:
                    if cur_title and content['text_level'] > cur_level:
                        block['title'] = cur_title
                    cur_title = content['text']
                    cur_level = content['text_level']
                    block['text_level'] = content['text_level']
                else:
                    if cur_title:
                        block['title'] = cur_title
                blocks.append(block)
            elif content['type'] == 'image':
                if not content.get('img_path', None):
                    continue
                block['image_path'] = content['img_path']
                block['img_caption'] = '\n'.join(self._normalize_content_recursively(content.get('image_caption', [])))
                block['img_footnote'] = '\n'.join(self._normalize_content_recursively(content.get('image_footnote', [])))
                if cur_title:
                    block['title'] = cur_title
                block['text'] = f'![{block["img_caption"]}]({block["image_path"]})'
                block['text'] += f'\n{block["img_footnote"]}\n' if block['img_footnote'] else '\n'
                blocks.append(block)
            elif content['type'] == 'table':
                if self._extract_table:
                    block['text'] = self._html_table_to_markdown(
                        self._normalize_content_recursively(content.get('table_body', '')))
                    block['table_caption'] = '\n'.join(
                        self._normalize_content_recursively(content.get('table_caption', [])))
                    block['table_footnote'] = '\n'.join(
                        self._normalize_content_recursively(content.get('table_footnote', [])))
                    if block.get('text', None):
                        block['text'] = f'{block["table_caption"]}\n{block["text"]}'.lstrip('\n')
                        block['text'] += f'\n{block["table_footnote"]}\n' if block['table_footnote'] else '\n'
                else:
                    block['image_path'] = content.get('img_path', '')
                    block['text'] = f'![table]({block["image_path"]})'
                if cur_title:
                    block['title'] = cur_title
                blocks.append(block)
            elif content['type'] == 'equation':
                if self._extract_formula:
                    block['text'] = content.get('text', '')
                else:
                    block['image_path'] = content.get('img_path', '')
                    if not block['image_path']:
                        continue
                    block['text'] = f'![formula]({block["image_path"]})'
                if cur_title:
                    block['title'] = cur_title
                blocks.append(block)
            elif content['type'] == 'code':
                block['text'] = content['code_body']
                if not block['text']:
                    continue
                if content.get('code_caption', None):
                    code_caption = '\n'.join(self._normalize_content_recursively(content['code_caption']))
                    block['text'] = f"{code_caption}\n{block['text']}"
                block['code_type'] = content.get('sub_type', '')
                if cur_title:
                    block['title'] = cur_title
                blocks.append(block)
                LOG.info(f'[MineruPDFReader] Found code block: {block["text"][:100]}...')
            elif content['type'] == 'list':
                block['list_type'] = content.get('sub_type', '')
                block['text'] = '\n'.join(self._normalize_content_recursively(content.get('list_items', [])))
                if cur_title:
                    block['title'] = cur_title
                blocks.append(block)
            else:
                block = copy.deepcopy(content)
                block['type'] = content['type']
                if 'content' in block:
                    block['text'] = block['content']
                    del block['content']
                if cur_title:
                    block['title'] = cur_title
                blocks.append(block)
        return blocks

    def _normalize_content_recursively(self, content) -> str:
        if isinstance(content, str):
            content = content.encode('utf-8', 'replace').decode('utf-8')
            return unicodedata.normalize('NFKC', content)
        if isinstance(content, list):
            return [self._normalize_content_recursively(t) for t in content]
        return content

    def _html_table_to_markdown(self, html_table) -> str:  # noqa: C901
        if not html_table:
            return ''
        try:
            soup = bs4.BeautifulSoup(html_table.strip(), 'html.parser')
            table = soup.find('table')
            if not table:
                raise ValueError('No <table> found in the HTML.')

            rows = []
            max_cols = 0

            for row in table.find_all('tr'):
                cells = []
                for cell in row.find_all(['td', 'th']):
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    text = cell.get_text(strip=True)

                    for _ in range(colspan):
                        cells.append({'text': text, 'rowspan': rowspan})
                rows.append(cells)
                max_cols = max(max_cols, len(cells))

            expanded_rows = []
            rowspan_tracker = [0] * max_cols
            for row in rows:
                expanded_row = []
                col_idx = 0
                for cell in row:
                    while col_idx < max_cols and rowspan_tracker[col_idx] > 0:
                        expanded_row.append(None)
                        rowspan_tracker[col_idx] -= 1
                        col_idx += 1

                    expanded_row.append(cell['text'])
                    if cell['rowspan'] > 1:
                        rowspan_tracker[col_idx] = cell['rowspan'] - 1
                    col_idx += 1

                while col_idx < max_cols:
                    if rowspan_tracker[col_idx] > 0:
                        expanded_row.append(None)
                        rowspan_tracker[col_idx] -= 1
                    else:
                        expanded_row.append('')
                    col_idx += 1

                expanded_rows.append(expanded_row)

            markdown = ''
            if not expanded_rows:
                return ''

            headers = expanded_rows[0]
            body_rows = expanded_rows[1:]
            if headers:
                markdown += '| ' + ' | '.join(h if h else '' for h in headers) + ' |\n'
                markdown += '| ' + ' | '.join(['-' * (len(h) if h else 3) for h in headers]) + ' |\n'
            for row in body_rows:
                markdown += '| ' + ' | '.join(cell if cell else '' for cell in row) + ' |\n'

            return markdown

        except Exception as e:
            LOG.error(f'Error parsing table: {e}')
            return str(html_table)

    def _build_nodes(self, elements: List[dict], file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
        docs = []
        if self._split_doc:
            for e in elements:
                metadata = {'file_name': file.name}
                metadata.update({k: v for k, v in e.items() if k != 'text'})
                metadata.update({'file_path': str(file)})
                node = DocNode(text=e.get('text', ''), metadata=metadata, global_metadata=extra_info)
                node.excluded_embed_metadata_keys = ['type', 'index', 'text_level', 'bbox', 'lines']
                node.excluded_llm_metadata_keys = ['type', 'index', 'text_level', 'bbox', 'lines']
                docs.append(node)
        else:
            text_chunks = [el['text'] for el in elements if 'text' in el]
            nodes = DocNode(text='\n'.join(text_chunks), metadata={'file_name': file.name})
            nodes.excluded_embed_metadata_keys = ['type', 'index', 'text_level', 'bbox', 'lines']
            nodes.excluded_llm_metadata_keys = ['type', 'index', 'text_level', 'bbox', 'lines']
            docs.append(nodes)
        return docs
