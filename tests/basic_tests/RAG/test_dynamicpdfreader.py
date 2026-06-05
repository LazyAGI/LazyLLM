from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest

from lazyllm import globals as lazyllm_globals
from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader import DynamicPDFReader
from lazyllm.tools.rag.readers.ocrReader.mineru_pdf_reader import MineruPDFReader
from lazyllm.tools.rag.readers.ocrReader.ocr_service import (
    OcrServiceVariant,
    default_online_url,
    resolve_ocr_variant,
)


@contextmanager
def _dynamic_ocr_configs(value):
    old = lazyllm_globals.config['dynamic_ocr_configs']
    lazyllm_globals.config['dynamic_ocr_configs'] = value
    try:
        yield
    finally:
        lazyllm_globals.config['dynamic_ocr_configs'] = old


class TestDynamicPDFReader:

    def test_resolve_route_priority(self):
        reader = DynamicPDFReader(ocr_type='none')
        with _dynamic_ocr_configs({
                'ocr_type': 'paddleocr',
                'ocr_url': 'http://dynamic-service',
        }):
            ocr_type, ocr_url = reader._resolve_route({
                'ocr_type': 'mineru',
                'ocr_url': 'http://extra-service',
            })

        assert ocr_type == 'mineru'
        assert ocr_url == 'http://extra-service'

    def test_normalize_paddle_alias(self):
        reader = DynamicPDFReader(ocr_type='paddle', ocr_url='http://mock')
        with _dynamic_ocr_configs(None):
            ocr_type, ocr_url = reader._resolve_route(None)

        assert ocr_type == 'paddleocr'
        assert ocr_url == 'http://mock'

    def test_reader_cache_reuse_same_options(self):
        reader = DynamicPDFReader(ocr_type='mineru', ocr_url='http://mock')
        fake_reader = MagicMock()
        fake_reader.forward.return_value = [DocNode(text='ok')]

        with patch.object(reader, '_build_reader', return_value=fake_reader) as mock_build:
            result_1 = reader._load_data('fake.pdf', extra_info=None, use_cache=True)
            result_2 = reader._load_data('fake.pdf', extra_info=None, use_cache=True)

        assert mock_build.call_count == 1
        assert len(result_1) == 1
        assert len(result_2) == 1
        assert fake_reader.forward.call_count == 2

    def test_build_reader_uses_dynamic_auth(self):
        reader = DynamicPDFReader(ocr_type='paddleocr', ocr_url='http://mock-paddle')
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.PaddleOCRPDFReader'
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            reader._build_reader('paddleocr', 'http://mock-paddle')
            mock_cls.assert_called_once_with(
                url='http://mock-paddle',
                dynamic_auth=True,
            )

    def test_unsupported_type_raises(self):
        reader = DynamicPDFReader()
        with pytest.raises(ValueError, match='Unsupported OCR server type'):
            reader._build_reader('badtype', 'http://fake')

    def test_build_mineru_dynamic_auth(self):
        reader = DynamicPDFReader(ocr_type='mineru', ocr_url='http://mock-mineru')
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.MineruPDFReader'
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            reader._build_reader('mineru', 'http://mock-mineru')
            assert mock_cls.call_args.kwargs['dynamic_auth'] is True

    def test_dynamic_mineru_type_without_url_uses_official(self):
        reader = DynamicPDFReader(
            ocr_type='none',
            ocr_url='http://host.docker.internal:8000/api/v1/pdf_parse',
        )
        with _dynamic_ocr_configs({'ocr_type': 'mineru'}):
            ocr_type, ocr_url = reader._resolve_route({'mineru_api_key': 'mineru-key'})

        assert ocr_type == 'mineru'
        assert ocr_url == ''

    def test_dynamic_paddle_type_without_url_uses_official(self):
        reader = DynamicPDFReader(
            ocr_type='none',
            ocr_url='http://host.docker.internal:8000/api/v1/pdf_parse',
        )
        with _dynamic_ocr_configs({'ocr_type': 'paddleocr'}):
            ocr_type, ocr_url = reader._resolve_route({'paddle_api_key': 'paddle-key'})

        assert ocr_type == 'paddleocr'
        assert ocr_url == ''

    def test_explicit_empty_ocr_url(self):
        reader = DynamicPDFReader(
            ocr_url='http://host.docker.internal:8000/api/v1/pdf_parse',
        )
        with _dynamic_ocr_configs({'ocr_type': 'mineru', 'ocr_url': ''}):
            ocr_type, ocr_url = reader._resolve_route(None)

        assert ocr_type == 'mineru'
        assert ocr_url == ''

    def test_mineru_reader_defaults_empty_url_to_official(self):
        reader = MineruPDFReader(url='', dynamic_auth=True)
        assert reader._url == 'https://mineru.net'
        assert reader._offline_mode is False

    def test_static_route_uses_env_url_without_dynamic_request(self):
        reader = DynamicPDFReader(
            ocr_type='mineru',
            ocr_url='http://local-mineru:8000/api/v1/pdf_parse',
        )
        with _dynamic_ocr_configs(None):
            ocr_type, ocr_url = reader._resolve_route(None)

        assert ocr_type == 'mineru'
        assert ocr_url == 'http://local-mineru:8000/api/v1/pdf_parse'

    def test_online_url_detection(self):
        assert resolve_ocr_variant('mineru', 'https://mineru.net/api/v4/foo') == OcrServiceVariant.ONLINE
        assert resolve_ocr_variant(
            'mineru', 'http://172.24.176.1:20234/api/v1/pdf_parse'
        ) == OcrServiceVariant.OFFLINE
        assert resolve_ocr_variant(
            'mineru', 'http://host.docker.internal:8000/api/v1/pdf_parse'
        ) == OcrServiceVariant.OFFLINE
        assert resolve_ocr_variant('mineru', '') == OcrServiceVariant.ONLINE
        assert resolve_ocr_variant('paddleocr', default_online_url('paddleocr')) == OcrServiceVariant.ONLINE
        assert resolve_ocr_variant(
            'paddleocr', 'http://host.docker.internal:8000/api/v1/pdf_parse'
        ) == OcrServiceVariant.OFFLINE
        assert resolve_ocr_variant('paddleocr', '') == OcrServiceVariant.ONLINE

    def test_offline_mineru_enables_upload(self):
        reader = MineruPDFReader(
            url='http://172.24.176.1:20234/api/v1/pdf_parse',
        )
        assert reader._variant == OcrServiceVariant.OFFLINE
        assert reader._upload_mode is True

    def test_online_mineru_disables_upload(self):
        reader = MineruPDFReader(url='https://mineru.net')
        assert reader._variant == OcrServiceVariant.ONLINE
        assert reader._upload_mode is False
