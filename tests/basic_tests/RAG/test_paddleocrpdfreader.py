import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import lazyllm
from lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader import (
    PaddleOCRPDFReader, JOB_URL,
)
from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.doc_node import RichDocNode
from lazyllm.tools.rag.transform import RichTransform
from lazyllm.tools.rag.transform.sentence import SentenceSplitter


# ---------------------------------------------------------------------------
# Real PDF generation (cached to disk)
# ---------------------------------------------------------------------------

_TEST_PDF_DIR = Path(__file__).resolve().parent.parent.parent / 'ci_data' / 'test_paddleocr'
_TEST_PDF_PATH = _TEST_PDF_DIR / 'test_paddleocr.pdf'


def _helvetica_content(texts):
    '''Build a PDF content stream with positionable Helvetica text.

    texts: list of (font_size, x, y, text) — y=0 is bottom of page.
    '''
    lines = ['BT']
    for size, x, y, text in texts:
        escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        lines.append(f'/F1 {size} Tf')
        lines.append(f'{x} {y} Td')
        lines.append(f'({escaped}) Tj')
        lines.append('T*')
    lines.append('ET')
    return '\n'.join(lines)


def _build_minimal_pdf(pages_texts, width=612, height=792):
    '''Build a valid multi-page PDF from lists of (font_size, x, y, text) tuples.'''
    objects = {}
    font_obj = 6

    objects[font_obj] = (
        f'{font_obj} 0 obj\n'
        f'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\n'
        f'endobj'
    )

    page_refs = []
    for i, texts in enumerate(pages_texts):
        content_stream = _helvetica_content(texts)
        content_obj = font_obj + 1 + i * 3
        page_obj = content_obj + 1

        objects[content_obj] = (
            f'{content_obj} 0 obj\n'
            f'<< /Length {len(content_stream)} >>\n'
            f'stream\n{content_stream}\nendstream\nendobj'
        )
        objects[page_obj] = (
            f'{page_obj} 0 obj\n'
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] '
            f'/Contents {content_obj} 0 R '
            f'/Resources << /Font << /F1 {font_obj} 0 R >> >> >>\n'
            f'endobj'
        )
        page_refs.append(f'{page_obj} 0 R')

    objects[2] = (
        f'2 0 obj\n'
        f'<< /Type /Pages /Kids [{" ".join(page_refs)}] /Count {len(pages_texts)} >>\n'
        f'endobj'
    )
    objects[1] = '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj'

    lines = ['%PDF-1.4']
    xref_offsets = {}
    for obj_num in sorted(objects):
        xref_offsets[obj_num] = len('\n'.join(lines)) + 1
        lines.append(objects[obj_num])

    xref_start = len('\n'.join(lines)) + 1
    max_obj = max(objects.keys())
    xref_lines = []
    for i in range(max_obj + 1):
        if i == 0:
            xref_lines.append('0000000000 65535 f ')
        elif i in xref_offsets:
            xref_lines.append(f'{xref_offsets[i]:010d} 00000 n ')
        else:
            xref_lines.append('0000000000 65535 f ')

    lines.append('xref')
    lines.append(f'0 {len(xref_lines)}')
    for entry in xref_lines:
        lines.append(entry)
    lines.append('trailer')
    lines.append(f'<< /Size {len(xref_lines)} /Root 1 0 R >>')
    lines.append('startxref')
    lines.append(str(xref_start))
    lines.append('%%EOF')

    return '\n'.join(lines).encode('latin-1')


def _make_test_pdf():
    '''Return path to a real 2-page PDF (cached).'''
    if _TEST_PDF_PATH.exists():
        return _TEST_PDF_PATH

    _TEST_PDF_DIR.mkdir(parents=True, exist_ok=True)

    h = 792
    pages = [
        [
            (20, 50, h - 60, 'Test Title: PaddleOCR Integration Test'),
            (12, 50, h - 100, 'This is the first paragraph of the test document.'),
            (12, 50, h - 120, 'It contains text that PaddleOCR should recognize as a paragraph.'),
            (16, 50, h - 160, 'Section 1.1'),
            (12, 50, h - 180, 'This section discusses the detailed aspects of OCR testing.'),
            (12, 50, h - 200, 'PaddleOCR is a powerful tool for document parsing and layout analysis.'),
        ],
        [
            (18, 50, h - 60, 'Section 2: Additional Content'),
            (12, 50, h - 100, 'The second page contains more content for multi-page testing.'),
            (12, 50, h - 120, 'This ensures that multi-page PDF parsing works correctly.'),
            (14, 50, h - 160, 'Subsection 2.1: Details'),
            (12, 50, h - 180, 'Here are some detailed testing notes about the integration.'),
            (12, 50, h - 200, 'The system should extract all headings and paragraphs accurately.'),
        ],
    ]

    pdf_bytes = _build_minimal_pdf(pages)
    _TEST_PDF_PATH.write_bytes(pdf_bytes)
    return _TEST_PDF_PATH


# ---------------------------------------------------------------------------
# mock helpers
# ---------------------------------------------------------------------------

def _make_job_response(layout_parsing_results: list) -> str:
    return json.dumps({'result': {'layoutParsingResults': layout_parsing_results}})


def _make_page_result(blocks: list, images: dict = None) -> dict:
    return {
        'markdown': {'images': images or {}},
        'prunedResult': {'parsing_res_list': blocks},
    }


def _make_mock_response() -> str:
    return _make_job_response([
        _make_page_result([
            {'block_label': 'paragraph_title', 'block_content': '# Test Heading',
             'block_bbox': [0, 100, 400, 130]},
            {'block_label': 'text', 'block_content': 'This is test paragraph content.',
             'block_bbox': [0, 130, 400, 160]},
            {'block_label': 'image', 'block_content': '',
             'block_bbox': [100, 200, 300, 400]},
            {'block_label': 'table',
             'block_content': '<table><tr><th>A</th><th>B</th></tr>'
                              '<tr><td>1</td><td>2</td></tr></table>',
             'block_bbox': [0, 400, 400, 460]},
        ], images={
            'imgs/img_in_image_box_100_200_300_400.jpg': 'http://mock-url/test_img.jpg',
        }),
    ])


# ---------------------------------------------------------------------------
# Mock-based tests (mock _fetch_async, real PDF on disk)
# ---------------------------------------------------------------------------

class TestPaddleOCRPDFReaderMock:

    def test_load_data_mock(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader()

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = (_make_mock_response(), None)

            docs = reader._load_data(str(pdf))

        assert isinstance(docs, list)
        assert len(docs) > 0
        types = {d.metadata.get('type') for d in docs}
        assert 'heading' in types
        assert 'paragraph' in types
        assert 'figure' in types
        assert 'table' in types

    def test_resolve_image_by_bbox(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader()

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = (_make_mock_response(), None)

            docs = reader._load_data(str(pdf))

        image_nodes = [d for d in docs if d.metadata.get('type') == 'figure']
        assert len(image_nodes) == 1
        assert image_nodes[0].metadata['image_path'] is not None

    def test_split_doc_false(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader(split_doc=False)

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = (_make_mock_response(), None)

            docs = reader._load_data(str(pdf))

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], DocNode)

    def test_drop_types(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader(drop_types=['header', 'footer', 'aside_text'])

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = (_make_mock_response(), None)

            docs = reader._load_data(str(pdf))

        for d in docs:
            if 'type' in d.metadata:
                assert d.metadata['type'] not in ('header', 'footer', 'aside_text')

    def test_image_cache_dir(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader()

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = (_make_mock_response(), None)

            docs = reader._load_data(str(pdf))

        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_merge_jsonl_lines(self):
        jsonl = (
            '{"result": {"layoutParsingResults": [{"a": 1}]}}\n'
            '{"result": {"layoutParsingResults": [{"b": 2}, {"c": 3}]}}\n'
        )
        merged = PaddleOCRPDFReader._merge_jsonl_lines(jsonl)
        data = json.loads(merged)
        assert data['result']['layoutParsingResults'] == [{'a': 1}, {'b': 2}, {'c': 3}]

    def test_merge_jsonl_lines_skips_empty(self):
        jsonl = '\n{"result": {"layoutParsingResults": [{"x": 1}]}}\n\n'
        merged = PaddleOCRPDFReader._merge_jsonl_lines(jsonl)
        data = json.loads(merged)
        assert len(data['result']['layoutParsingResults']) == 1

    def test_resolve_image_single(self):
        images = {'img.png': 'http://u'}
        result = PaddleOCRPDFReader._resolve_image([0, 0, 100, 100], images)
        assert result == ('img.png', 'http://u')

    def test_resolve_image_by_bbox_match(self):
        images = {
            'imgs/img_in_image_box_10_20_30_40.jpg': 'http://a',
            'imgs/img_in_image_box_50_60_70_80.jpg': 'http://b',
        }
        result = PaddleOCRPDFReader._resolve_image([50, 60, 70, 80], images)
        assert result == ('imgs/img_in_image_box_50_60_70_80.jpg', 'http://b')

    def test_resolve_image_empty_returns_none(self):
        result = PaddleOCRPDFReader._resolve_image([0, 0, 100, 100], {})
        assert result is None

    def test_fetch_job_submits_and_polls(self):
        pdf = _make_test_pdf()

        with patch('lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader.post_sync') as mock_post, \
             patch('lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader.get_sync') as mock_get:
            submit_resp = mock_post.return_value
            submit_resp.json.return_value = {'data': {'jobId': 'test-job-123'}}

            poll_resp_mock = MagicMock()
            poll_resp_mock.json.return_value = {
                'data': {
                    'state': 'done',
                    'resultUrl': {'jsonUrl': 'http://mock/jsonl'},
                }
            }
            jsonl_resp_mock = MagicMock()
            jsonl_resp_mock.text = '{"result": {"layoutParsingResults": [{"x": 1}]}}'
            mock_get.side_effect = [poll_resp_mock, jsonl_resp_mock]

            reader = PaddleOCRPDFReader()
            result, task_dir = reader._fetch_job(str(pdf))

        assert mock_post.called
        submit_url = mock_post.call_args[0][0]
        assert submit_url == JOB_URL

        assert mock_get.call_count >= 1
        poll_url = mock_get.call_args_list[0][0][0]
        assert 'test-job-123' in poll_url

        data = json.loads(result)
        assert data['result']['layoutParsingResults'] == [{'x': 1}]
        assert task_dir is None


# ---------------------------------------------------------------------------
# Live-service tests (skip when API key unavailable)
# ---------------------------------------------------------------------------

class TestPaddleOCRPDFReaderLive:

    def setup_method(self):
        try:
            self.api_key = lazyllm.config['paddle_api_key']
        except KeyError:
            self.api_key = os.environ.get('LAZYLLM_PADDLE_API_KEY', '')
        if not self.api_key:
            pytest.skip('LAZYLLM_PADDLE_API_KEY not set')

    def test_load_data_online(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader()
        docs = reader._load_data(str(pdf))

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], DocNode)

    def test_richdocnode_compatible(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader(split_doc=True)
        docs = reader._load_data(str(pdf))

        assert len(docs) >= 1
        assert isinstance(docs[0], DocNode)

        # RichDocNode only when multi-page content has enough structure
        if isinstance(docs[0], RichDocNode):
            split_nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=10)([docs[0]])
            assert isinstance(split_nodes, list)
            assert len(split_nodes) > 0

            rich_nodes = RichTransform()([docs[0]])
            assert isinstance(rich_nodes, list)
            assert len(rich_nodes) > 0
        else:
            # Simple PDF: plain DocNode list, verify at least one has text
            assert any(len(node.text) > 0 for node in docs)

    def test_split_doc_false_live(self):
        pdf = _make_test_pdf()
        reader = PaddleOCRPDFReader(split_doc=False)
        docs = reader._load_data(str(pdf))

        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], DocNode)
