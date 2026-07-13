from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest

from lazyllm import globals as lazyllm_globals
from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader import DynamicPDFReader
from lazyllm.tools.rag.readers.ocrReader.mineru_pdf_reader import MineruPDFReader
from lazyllm.tools.rag.readers.ocrReader.mineru_ppt_reader import MineruPPTReader
from lazyllm.tools.rag.readers.ocrReader.ocr_ir import FigureBlock, ParagraphBlock
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
            result_1 = reader._load_data('fake.pdf', extra_info=None)
            result_2 = reader._load_data('fake.pdf', extra_info=None)

        assert mock_build.call_count == 1
        assert len(result_1) == 1
        assert len(result_2) == 1
        assert fake_reader.forward.call_count == 2

    def test_build_reader_uses_dynamic_auth(self):
        post_func = MagicMock()
        reader = DynamicPDFReader(
            ocr_type='paddleocr',
            ocr_url='http://mock-paddle',
            post_func=post_func,
            timeout=3600,
        )
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.PaddleOCRPDFReader'
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            reader._build_reader('paddleocr', 'http://mock-paddle')
            mock_cls.assert_called_once_with(
                url='http://mock-paddle',
                dynamic_auth=True,
                timeout=3600,
                post_func=post_func,
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

    def test_build_mineru_ppt_reader(self):
        reader = DynamicPDFReader(ocr_type='mineru', ocr_url='http://mock-mineru')
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.MineruPPTReader'
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            reader._build_reader('mineru_ppt', 'http://mock-mineru')
            assert mock_cls.call_args.kwargs['dynamic_auth'] is True

    def test_ppt_file_uses_separate_reader_cache(self):
        reader = DynamicPDFReader(ocr_type='mineru', ocr_url='http://mock-mineru')
        pdf_reader = MagicMock()
        ppt_reader = MagicMock()
        pdf_reader.forward.return_value = [DocNode(text='pdf')]
        ppt_reader.forward.return_value = [DocNode(text='ppt')]

        with patch.object(reader, '_build_reader', side_effect=[pdf_reader, ppt_reader]) as mock_build:
            reader._load_data('/tmp/demo.pdf', extra_info=None)
            reader._load_data('/tmp/demo.pptx', extra_info=None)

        assert mock_build.call_count == 2
        assert mock_build.call_args_list[0].args == ('mineru', 'http://mock-mineru')
        assert mock_build.call_args_list[1].args == ('mineru_ppt', 'http://mock-mineru')

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

    def test_mineru_reader_content_cache_skips_load(self, tmp_path):
        import lazyllm
        from lazyllm.module.module import module_cache

        pdf_path = tmp_path / 'demo.pdf'
        pdf_path.write_bytes(b'%PDF-1.4 demo')

        old_cache_mode = lazyllm.config['cache_mode']
        lazyllm.config['cache_mode'] = 'RW'
        module_cache.close()
        try:
            lazyllm.config['reader_use_cache'] = True
            reader = MineruPDFReader(url='https://mineru.net')
            with patch.object(
                MineruPDFReader, '_load_data', return_value=[DocNode(text='cached')]
            ) as mock_load:
                reader(pdf_path)
                reader(pdf_path)
            assert mock_load.call_count == 1
        finally:
            lazyllm.config['cache_mode'] = old_cache_mode
            module_cache.close()

    def test_ppt_reader_reuses_mineru_auth_key(self):
        reader = MineruPPTReader(url='https://mineru.net', dynamic_auth=True)
        assert reader._auth_source_key == 'mineru'

    def test_ppt_skips_pdf_split(self):
        ppt_path = '/tmp/demo.pptx'
        assert MineruPPTReader._split_large_pdf(ppt_path) == [(ppt_path, 0)]

    def test_ppt_missing_bbox_uses_zero_bbox(self):
        reader = MineruPPTReader(url='https://mineru.net')
        text_block = reader._adapt_one({
            'type': 'text',
            'text': 'hello',
            'page_idx': 0,
        })
        image_block = reader._adapt_one({
            'type': 'image',
            'img_path': 'images/demo.jpg',
            'page_idx': 1,
        })
        assert isinstance(text_block, ParagraphBlock)
        assert text_block.page.bbox.to_list() == [0, 0, 0, 0]
        assert isinstance(image_block, FigureBlock)
        assert image_block.page.bbox.to_list() == [0, 0, 0, 0]

    def test_pdf_official_missing_bbox_still_skipped(self):
        reader = MineruPDFReader(url='https://mineru.net')
        assert reader._adapt_one({'type': 'text', 'text': 'hello', 'page_idx': 0}) is None

    def test_pdf_official_keeps_returned_bbox(self):
        reader = MineruPDFReader(url='https://mineru.net')
        block = reader._adapt_one({
            'type': 'text',
            'text': 'hello',
            'page_idx': 2,
            'bbox': [10.0, 20.0, 100.0, 50.0],
        })
        assert isinstance(block, ParagraphBlock)
        assert block.page.index == 2
        assert block.page.bbox.to_list() == [10.0, 20.0, 100.0, 50.0]

    def test_normalize_online_content_bboxes_to_pdf_space(self):
        content = [
            {'type': 'text', 'text': 'title', 'page_idx': 0, 'bbox': [361, 80, 636, 99]},
            {'type': 'text', 'text': 'body', 'page_idx': 0, 'bbox': [174, 224, 825, 502]},
        ]
        layout = {'pdf_info': [{'page_idx': 0, 'page_size': [595, 841]}]}
        model = [[
            {'type': 'doc_title', 'bbox': [0.363, 0.081, 0.637, 0.101], 'content': 'title'},
            {'type': 'text', 'bbox': [0.175, 0.225, 0.826, 0.503], 'content': 'body'},
        ]]
        out = MineruPDFReader._normalize_online_content_bboxes(content, layout, model)
        # OCR canvas inferred from max extent ≈ 1000; result should be near PDF points.
        assert out[0]['bbox'][0] == pytest.approx(215.0, abs=2.0)
        assert out[0]['bbox'][1] == pytest.approx(67.5, abs=2.0)
        assert out[0]['bbox'][2] == pytest.approx(379.0, abs=2.0)
        assert out[0]['bbox'][3] == pytest.approx(83.5, abs=2.0)

    def test_pdf_offline_missing_bbox_still_skipped(self):
        reader = MineruPDFReader(url='http://local-mineru:8000/api/v1/pdf_parse')
        assert reader._adapt_one({'type': 'text', 'text': 'hello', 'page_idx': 0}) is None

    def test_online_ssl_verify_respects_config(self, monkeypatch):
        monkeypatch.setenv('LAZYLLM_MINERU_SSL_VERIFY', 'false')
        reader = MineruPDFReader(url='https://mineru.net')
        assert reader._online_request_kwargs() == {'verify': False}
