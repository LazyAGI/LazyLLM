from lazyllm import globals
from lazyllm.tools.rag.readers.reader_config_inject import inject_reader_config


def test_inject_reader_config_clears_previous_ocr_when_missing():
    inject_reader_config(ocr_config={
        'ocr_type': 'mineru',
        'ocr_url': 'https://mineru.net/api/v4/',
        'ocr_auth': {'mineru': 'token-a'},
    })
    assert globals.config['dynamic_ocr_configs']['ocr_type'] == 'mineru'
    assert globals.config['dynamic_ocr_auth']['mineru'] == 'token-a'

    inject_reader_config(ocr_config=None)

    assert globals.config['dynamic_ocr_configs'] is None
    assert globals.config['dynamic_ocr_auth'] is None


def test_inject_reader_config_replaces_previous_ocr():
    inject_reader_config(ocr_config={
        'ocr_type': 'mineru',
        'ocr_url': 'https://mineru.net/api/v4/',
        'ocr_auth': {'mineru': 'token-a'},
    })
    inject_reader_config(ocr_config={
        'ocr_type': 'paddleocr',
        'ocr_url': 'http://paddle:8000',
        'ocr_auth': {'paddleocr': 'token-b'},
    })

    assert globals.config['dynamic_ocr_configs'] == {
        'ocr_type': 'paddleocr',
        'ocr_url': 'http://paddle:8000',
    }
    assert globals.config['dynamic_ocr_auth'] == {'paddleocr': 'token-b'}
