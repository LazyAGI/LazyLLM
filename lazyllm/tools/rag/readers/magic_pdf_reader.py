import os
import copy
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Callable

import unicodedata
from ..doc_node import DocNode
from lazyllm import LOG
import requests

class MagicPDFReader:

    def __init__(self, magic_url, callback: Optional[Callable[[List[dict], Path, dict], List[DocNode]]] = None,
                 upload_mode: bool = False):
        self._magic_url = magic_url
        self._upload_mode = upload_mode
        if callback is not None:
            self._callback = callback
        else:
            def default_callback(elements: List[dict], file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
                text_chunks = [el["text"] for el in elements if "text" in el]
                return [DocNode(text="\n".join(text_chunks), metadata={"file_name": file.name})]
            self._callback = default_callback

    def __call__(self, file: Path, **kwargs) -> List[DocNode]:
        try:
            return self._load_data(file, **kwargs)
        except Exception as e:
            LOG.error(f"[MagicPDFReader] Error loading data from {file}: {e}")
            return []

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None, **kwargs) -> List[DocNode]:
        if isinstance(file, str):
            file = Path(file)
        if self._upload_mode:
            elements = self._upload_parse_pdf_elements(file)
        else:
            elements = self._parse_pdf_elements(file)
        docs: List[DocNode] = self._callback(elements, file, extra_info)
        return docs

    def _parse_pdf_elements(self, pdf_path: Path) -> List[dict]:
        payload = {"files": [str(pdf_path)], "reserve_image": True}
        try:
            response = requests.post(self._magic_url, json=payload)
            response.raise_for_status()
            res = response.json()
            if not isinstance(res, list) or not res:
                LOG.info(f"[MagicPDFReader] No elements found in PDF: {pdf_path}")
                return []
        except requests.exceptions.RequestException as e:
            LOG.error(f"[MagicPDFReader] POST failed: {e}")
            return []
        return self._extract_content_blocks(res[0])

    def _upload_parse_pdf_elements(self, pdf_path: Path) -> List[dict]:
        try:
            with open(pdf_path, "rb") as f:
                files = {'file': (os.path.basename(pdf_path), f)}
                response = requests.post(self._magic_url, files=files)
                response.raise_for_status()
                res = response.json()
                if not isinstance(res, list) or not res:
                    LOG.info(f"[MagicPDFReader] No elements found in PDF: {pdf_path}")
                    return []
        except requests.exceptions.RequestException as e:
            LOG.error(f"[MagicPDFReader] POST failed: {e}")
            return []
        return self._extract_content_blocks(res[0])

    def _extract_content_blocks(self, content_list) -> List[dict]:  # noqa: C901
        blocks = []
        cur_title = ""
        cur_level = -1
        for content in content_list:
            block = {}
            block["bbox"] = content["bbox"]
            block["lines"] = content["lines"] if 'lines' in content else []
            for line in block['lines']:
                line['content'] = self._clean_content(line['content'])
            if content["type"] == "text":
                content["text"] = self._clean_content(content["text"]).strip()
                if not content["text"]:
                    continue
                if "text_level" in content:
                    if cur_title and content["text_level"] > cur_level:
                        content["title"] = cur_title
                    cur_title = content["text"]
                    cur_level = content["text_level"]
                else:
                    if cur_title:
                        content["title"] = cur_title
                block = copy.deepcopy(content)
                block["page"] = content["page_idx"]
                del block["page_idx"]
                blocks.append(block)
            elif content["type"] == "image":
                if not content["img_path"]:
                    continue
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                block["image_path"] = os.path.basename(content["img_path"])
                block['img_caption'] = self._clean_content(content['img_caption'])
                block['img_footnote'] = self._clean_content(content['img_footnote'])
                if cur_title:
                    block["title"] = cur_title
                img_title = block["img_caption"][0] if len(block["img_caption"]) > 0 else ""
                block["text"] = f"![{img_title}]({block['image_path']})"
                blocks.append(block)
            elif content["type"] == "table":
                block["type"] = content["type"]
                block["page"] = content["page_idx"]
                if self.extract_table:
                    block["text"] = self._html_table_to_markdown(self._clean_content(content["table_body"])
                                                                 ) if "table_body" in content else ""
                else:
                    block['image_path'] = os.path.basename(content['img_path'])
                if cur_title:
                    block["title"] = cur_title
                block['table_caption'] = self._clean_content(content['table_caption'])
                block['table_footnote'] = self._clean_content(content['table_footnote'])
                blocks.append(block)
        return blocks

    def _clean_content(self, content) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8", "replace").decode("utf-8")
            return unicodedata.normalize("NFKC", content)
        if isinstance(content, list):
            return [self._clean_content(t) for t in content]
        return content

    def _html_table_to_markdown(self, html_table) -> str:  # noqa: C901
        try:
            soup = BeautifulSoup(html_table.strip(), 'html.parser')
            table = soup.find('table')
            if not table:
                raise ValueError("No <table> found in the HTML.")

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
                        expanded_row.append("")
                    col_idx += 1

                expanded_rows.append(expanded_row)

            markdown = ''
            if not expanded_rows:
                return ""

            headers = expanded_rows[0]
            body_rows = expanded_rows[1:]
            if headers:
                markdown += '| ' + ' | '.join(h if h else '' for h in headers) + ' |\n'
                markdown += '| ' + ' | '.join(['-' * (len(h) if h else 3) for h in headers]) + ' |\n'
            for row in body_rows:
                markdown += '| ' + ' | '.join(cell if cell else '' for cell in row) + ' |\n'

            return markdown

        except Exception as e:
            LOG.error(f"Error parsing table: {e}")
            return ''
