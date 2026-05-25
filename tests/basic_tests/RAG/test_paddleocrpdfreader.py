import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

import lazyllm
from lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader import (
    PaddleOCRPDFReader, JOB_URL,
)
from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.doc_node import RichDocNode
from lazyllm.tools.rag.transform import RichTransform
from lazyllm.tools.rag.transform.sentence import SentenceSplitter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_job_response(layout_parsing_results: list) -> str:
    '''Build merged JSONL result matching _merge_jsonl_lines output.'''
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


def _make_test_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / 'test.pdf'
    pdf.write_bytes(b'%PDF-1.4 fake pdf content for testing')
    return pdf


def _mock_fetch_job_return(mock_response: str):
    '''Return a (merged_json_str, None) tuple matching _fetch_job signature.'''
    return mock_response, None


# ---------------------------------------------------------------------------
# Mock-based tests
# ---------------------------------------------------------------------------

class TestPaddleOCRPDFReaderMock:

    def test_load_data_mock(self, tmp_path):
        pdf = _make_test_pdf(tmp_path)
        reader = PaddleOCRPDFReader()

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = _mock_fetch_job_return(_make_mock_response())

            docs = reader._load_data(pdf)

        assert isinstance(docs, list)
        assert len(docs) > 0
        types = {d.metadata.get('type') for d in docs}
        assert 'heading' in types
        assert 'paragraph' in types
        assert 'figure' in types
        assert 'table' in types

    def test_resolve_image_by_bbox(self, tmp_path):
        pdf = _make_test_pdf(tmp_path)
        reader = PaddleOCRPDFReader()

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = _mock_fetch_job_return(_make_mock_response())

            docs = reader._load_data(pdf)

        image_nodes = [d for d in docs if d.metadata.get('type') == 'figure']
        assert len(image_nodes) == 1
        assert image_nodes[0].metadata['image_path'] is not None

    def test_offline_rejected(self):
        with pytest.raises(ValueError, match='only supports service_variant="online"'):
            PaddleOCRPDFReader(service_variant='offline')

    def test_split_doc_false(self, tmp_path):
        pdf = _make_test_pdf(tmp_path)
        reader = PaddleOCRPDFReader(split_doc=False)

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = _mock_fetch_job_return(_make_mock_response())

            docs = reader._load_data(pdf)

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], DocNode)

    def test_drop_types(self, tmp_path):
        pdf = _make_test_pdf(tmp_path)
        reader = PaddleOCRPDFReader(drop_types=['header', 'footer', 'aside_text'])

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = _mock_fetch_job_return(_make_mock_response())

            docs = reader._load_data(pdf)

        for d in docs:
            if 'type' in d.metadata:
                assert d.metadata['type'] not in ('header', 'footer', 'aside_text')

    def test_images_dir(self, tmp_path):
        pdf = _make_test_pdf(tmp_path)
        images_dir = tmp_path / 'images'
        reader = PaddleOCRPDFReader(images_dir=str(images_dir))

        with patch.object(reader, '_fetch_async') as mock_fetch, \
             patch.object(PaddleOCRPDFReader, '_download_images'):
            mock_fetch.return_value = _mock_fetch_job_return(_make_mock_response())

            docs = reader._load_data(pdf)

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

    def test_fetch_job_submits_and_polls(self, tmp_path):
        '''Verify _fetch_job calls the correct PaddleOCR Job API endpoints.'''
        pdf = _make_test_pdf(tmp_path)

        with patch('lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader.post_sync') as mock_post, \
             patch('lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader.get_sync') as mock_get, \
             patch('lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader.requests.get') as mock_requests_get:
            # Submit response
            submit_resp = mock_post.return_value
            submit_resp.json.return_value = {'data': {'jobId': 'test-job-123'}}

            # Poll response (done)
            poll_resp = mock_get.return_value
            poll_resp.json.return_value = {
                'data': {
                    'state': 'done',
                    'resultUrl': {'jsonUrl': 'http://mock/jsonl'},
                }
            }

            # JSONL download response
            jsonl_resp = mock_requests_get.return_value
            jsonl_resp.text = '{"result": {"layoutParsingResults": [{"x": 1}]}}'

            reader = PaddleOCRPDFReader()
            result, task_dir = reader._fetch_job(str(pdf))

        # Verify submit called correctly
        assert mock_post.called
        submit_url = mock_post.call_args[0][0]
        assert submit_url == JOB_URL

        # Verify poll called
        assert mock_get.called
        poll_url = mock_get.call_args[0][0]
        assert 'test-job-123' in poll_url

        # Verify result
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

        self.test_pdf = os.path.join(
            lazyllm.config['data_path'], 'ci_data/test_paddleocr/test_paddleocr.pdf')
        if not os.path.exists(self.test_pdf):
            pytest.skip(f'Test PDF not found: {self.test_pdf}')

    def test_load_data_online(self, tmp_path):
        pdf = Path(self.test_pdf)

        reader = PaddleOCRPDFReader()
        docs = reader._load_data(pdf)

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], DocNode)

    def test_richdocnode_compatible(self, tmp_path):
        pdf = Path(self.test_pdf)

        reader = PaddleOCRPDFReader(split_doc=True)
        docs = reader._load_data(pdf)

        assert len(docs) == 1
        assert isinstance(docs[0], RichDocNode)

        split_nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=10)([docs[0]])
        assert isinstance(split_nodes, list)
        assert len(split_nodes) > 0

        rich_nodes = RichTransform()([docs[0]])
        assert isinstance(rich_nodes, list)
        assert len(rich_nodes) > 0

    def test_split_doc_false_live(self, tmp_path):
        pdf = Path(self.test_pdf)

        reader = PaddleOCRPDFReader(split_doc=False)
        docs = reader._load_data(pdf)

        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], DocNode)
