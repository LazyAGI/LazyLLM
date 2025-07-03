import unittest
from lazyllm.tools.rag.readers import MagicPDFReader
from lazyllm.tools.rag.doc_node import DocNode
from unittest.mock import patch, MagicMock
from typing import List, Optional, Dict
from pathlib import Path


EGS_RESULT_LIST = [[
    {
        'text': '铁路信号设计规范',
        'file_name': 'TB 10007-2017 铁路信号设计规范.pdf',
        'type': 'text',
        'text_level': 1,
        'bbox': [267, 407, 630, 462],
        'lines': [{'bbox': [273, 414, 625, 457], 'content': '铁路信号设计规范', 'type': 'text', 'page': 0}],
        'page_idx': 0,
        'file_path': '/home/mnt/jisiyuan/projects/lazyllm-project/dataset/rag_data/pdf_data/TB 10007-2017 铁路信号设计规范.pdf'
    },
    {
        "text": "Code for Design of Railway Signaling",
        "file_name": "TB 10007-2017 铁路信号设计规范.pdf",
        "type": "text",
        "bbox": [
            253,
            478,
            644,
            508
        ],
        "lines": [{"bbox": [256, 480, 643, 508], "content": "Code for Design of Railway Signaling",
                   "type": "text", "page": 0}],
        "title": "铁路信号设计规范",
        "page_idx": 0,
        "file_path": "/home/mnt/jisiyuan/projects/lazyllm-project/dataset/rag_data/pdf_data/TB 10007-2017 铁路信号设计规范.pdf"
    }]]


class TestMagicPDFReader(unittest.TestCase):
    magic_url = "http://127.0.0.1:20231/api/v1/pdf_parse"

    @patch("lazyllm.tools.rag.readers.magic_pdf_reader.requests.post")
    def test_read(self, mock_post: MagicMock):
        magic_reader = MagicPDFReader(self.magic_url)
        mock_response = MagicMock()
        mock_response.json.return_value = EGS_RESULT_LIST
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = magic_reader("./demo.pdf")
        self.assertTrue(result[0].text.startswith("铁路信号设计规范"))

    @patch("lazyllm.tools.rag.readers.magic_pdf_reader.requests.post")
    def test_read_with_callback(self, mock_post: MagicMock):
        def custom_callback(elements: List[dict], file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
            text_chunks = [f"page_{el['page']}:{el['text']}" for el in elements if "text" in el]
            return [DocNode(text="\n".join(text_chunks), metadata={"file_name": file.name})]

        magic_reader = MagicPDFReader(self.magic_url, custom_callback)
        mock_response = MagicMock()
        mock_response.json.return_value = EGS_RESULT_LIST
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = magic_reader("./demo.pdf")
        self.assertTrue(result[0].text.startswith("page"))
