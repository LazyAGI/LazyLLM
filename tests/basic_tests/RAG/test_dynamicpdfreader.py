from unittest.mock import patch, MagicMock

from lazyllm.tools.rag import DocNode
from lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader import DynamicPDFReader


class TestDynamicPDFReader:

    def test_resolve_options_priority(self):
        reader = DynamicPDFReader(ocr_type='none', ocr_dynamic=False)
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_dynamic_ocr_configs'
        ) as mock_dynamic, patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_static_api_key'
        ) as mock_static:
            mock_dynamic.return_value = {
                'ocr_type': 'paddleocr',
                'ocr_url': 'http://dynamic-service',
                'mineru_upload_mode': 'false',
                'ocr_dynamic': True,
            }
            mock_static.side_effect = lambda key: 'static-mineru' if key == 'mineru_api_key' else 'static-paddle'

            resolved = reader._resolve_options({
                'ocr_type': 'mineru',
                'ocr_url': 'http://extra-service',
                'mineru_upload_mode': 'true',
            })

        assert resolved[0] == 'mineru'
        assert resolved[1] == 'http://extra-service'
        assert resolved[2] is True
        assert resolved[3] == 'static-mineru'
        assert resolved[4] == 'static-paddle'

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

    def test_extra_info_overrides_dynamic_config(self):
        reader = DynamicPDFReader(ocr_type='none')
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_dynamic_ocr_configs'
        ) as mock_dynamic, patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_static_api_key'
        ) as mock_static:
            mock_dynamic.return_value = {
                'ocr_type': 'paddleocr',
                'ocr_url': 'http://dynamic-url',
            }
            mock_static.return_value = None

            ocr_type, ocr_url, _, _, _ = reader._resolve_options({
                'ocr_type': 'mineru',
                'ocr_url': 'http://extra-url',
            })

        assert ocr_type == 'mineru'
        assert ocr_url == 'http://extra-url'

    def test_mineru_key_without_url_falls_back_to_official(self):
        reader = DynamicPDFReader(ocr_type='mineru')
        with patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_dynamic_ocr_configs'
        ) as mock_dynamic, patch(
            'lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader.read_static_api_key'
        ) as mock_static:
            mock_dynamic.return_value = None
            mock_static.side_effect = lambda key: 'mineru-key' if key == 'mineru_api_key' else None

            ocr_type, ocr_url, _, _, _ = reader._resolve_options({})

        assert ocr_type == 'mineru'
        assert ocr_url == 'https://mineru.net'
