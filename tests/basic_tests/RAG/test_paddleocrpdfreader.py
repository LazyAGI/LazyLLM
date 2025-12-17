import os
import pytest
import tempfile
import shutil
import requests

import lazyllm
from lazyllm.tools.rag.readers.paddleocr_pdf_reader import PaddleOCRPDFReader
from lazyllm.tools.rag import DocNode


def _is_url_accessible(base_url: str, timeout: float = 1.0) -> bool:
    if not base_url or not base_url.strip():
        return False

    health_url = base_url.rstrip('/') + '/health'

    try:
        resp = requests.get(health_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get('errorCode') == 0

    except (requests.exceptions.RequestException, ValueError):
        return False


class TestPaddleOCRPDFReader(object):
    def setup_method(self):
        # Skip all tests if paddleocrvl_url is empty or not accessible
        self.url = os.environ.get('LAZYLLM_PADDLEOCRVL_URL')
        if not self.url or not self.url.strip():
            pytest.skip('paddleocrvl_url is empty or not set')
        if not _is_url_accessible(self.url):
            pytest.skip(f'paddleocrvl_url is not accessible: {self.url}')

        self.test_pdf = os.path.join(lazyllm.config['data_path'], 'ci_data/test_paddleocr/test_paddleocr.pdf')
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_data_without_images_dir(self):
        if not os.path.exists(self.test_pdf):
            pytest.skip(f'Test file does not exist: {self.test_pdf}')
        reader = PaddleOCRPDFReader(url=self.url)
        docs = reader(self.test_pdf)
        assert isinstance(docs, list)
        assert len(docs) > 0, 'Return result should not be empty'

    def test_load_data_with_images_dir(self):
        if not os.path.exists(self.test_pdf):
            pytest.skip(f'Test file does not exist: {self.test_pdf}')
        images_dir = os.path.join(self.temp_dir, 'images')
        reader = PaddleOCRPDFReader(
            url=self.url,
            images_dir=images_dir
        )
        docs = reader(self.test_pdf)
        assert isinstance(docs, list)
        assert len(docs) > 0, 'Return result should not be empty'
        # Check if image_path in image node metadata actually exists
        image_nodes = [doc for doc in docs if doc.metadata.get('type') == 'image']
        for image_node in image_nodes:
            image_paths = image_node.metadata.get('image_path', [])
            assert isinstance(image_paths, list), 'image_path should be a list'
            for image_path in image_paths:
                assert os.path.exists(image_path), f'Image path does not exist: {image_path}'

    def test_load_data_with_split_doc_false(self):
        if not os.path.exists(self.test_pdf):
            pytest.skip(f'Test file does not exist: {self.test_pdf}')
        reader = PaddleOCRPDFReader(url=self.url, split_doc=False)
        docs = reader(self.test_pdf)
        assert isinstance(docs, list)
        assert len(docs) > 0, 'Return result should not be empty'
        assert len(docs) == 1, 'When split_doc=False, should return only one node'
        assert isinstance(docs[0], DocNode)

    def test_load_data_with_different_init_parameters(self):
        if not os.path.exists(self.test_pdf):
            pytest.skip(f'Test file does not exist: {self.test_pdf}')

        # Test format_block_content=False
        reader1 = PaddleOCRPDFReader(
            url=self.url,
            format_block_content=False
        )
        docs1 = reader1(self.test_pdf)
        assert isinstance(docs1, list)
        assert len(docs1) > 0, 'Return result should not be empty when format_block_content=False'

        # Test use_layout_detection=False
        reader2 = PaddleOCRPDFReader(
            url=self.url,
            use_layout_detection=False
        )
        docs2 = reader2(self.test_pdf)
        assert isinstance(docs2, list)
        assert len(docs2) > 0 and len(docs2) < 15, 'Return result should not be empty when use_layout_detection=False'

        # Test use_chart_recognition=False
        reader3 = PaddleOCRPDFReader(
            url=self.url,
            use_chart_recognition=False
        )
        docs3 = reader3(self.test_pdf)
        assert isinstance(docs3, list)
        assert len(docs3) > 0, 'Return result should not be empty when use_chart_recognition=False'

        # Test drop_types parameter
        reader4 = PaddleOCRPDFReader(
            url=self.url,
            drop_types=['header', 'footer', 'aside_text']
        )
        docs4 = reader4(self.test_pdf)
        assert isinstance(docs4, list)
        assert len(docs4) > 0, 'Return result should not be empty when using drop_types'
        # Check that metadata should not contain filtered types
        for doc in docs4:
            if 'type' in doc.metadata:
                assert doc.metadata['type'] not in ['header', 'footer', 'aside_text']
