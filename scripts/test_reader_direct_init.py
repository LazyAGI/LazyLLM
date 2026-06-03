#!/usr/bin/env python3
'''Smoke-test direct MineruPDFReader / PaddleOCRPDFReader construction in container.'''
import os
import sys
import traceback

from lazyllm import globals, inject_ocr_config
from lazyllm.tools.rag.readers.ocrReader.mineru_pdf_reader import MineruPDFReader
from lazyllm.tools.rag.readers.ocrReader.paddleocr_pdf_reader import PaddleOCRPDFReader

PDF = os.environ.get(
    'TEST_PDF',
    '/var/lib/lazymind/uploads/tenants/root/datasets/'
    'ds_dc26eb13c402b9a833e49a0983aaa23d/docs/files/upload_manual_dynamic/'
    'test_dynamic.pdf',
)
PADDLE_KEY = os.environ.get('PADDLE_KEY', '')
MINERU_KEY = os.environ.get('MINERU_KEY', '')
ENV_OCR_URL = os.environ.get(
    'LAZYMIND_OCR_SERVER_URL',
    'http://host.docker.internal:8000/api/v1/pdf_parse',
)


def clear_ocr_inject():
    for key in ('dynamic_ocr_configs', 'dynamic_ocr_auth'):
        try:
            del globals.config[key]
        except Exception:
            globals.config[key] = None


def run_init(label, factory):
    try:
        reader = factory()
        attrs = {}
        if isinstance(reader, MineruPDFReader):
            attrs = {
                '_url': reader._url,
                '_offline_mode': reader._offline_mode,
            }
        elif isinstance(reader, PaddleOCRPDFReader):
            attrs = {
                '_url': reader._url,
                '_job_url': reader._job_url,
            }
        print(f'OK   init[{label}] {attrs}')
        return reader, None
    except Exception as exc:
        print(f'FAIL init[{label}] {exc!r}')
        traceback.print_exc()
        return None, exc


def run_parse(label, reader):
    try:
        nodes = reader.forward(PDF, use_cache=False)
        print(f'OK   parse[{label}] nodes={len(nodes)}')
        return True
    except Exception as exc:
        print(f'FAIL parse[{label}] {type(exc).__name__}: {exc}')
        return False


def main():
    print('=== direct reader init / optional parse ===')
    print(f'PDF exists: {os.path.isfile(PDF)} path={PDF}')
    print(f'ENV LAZYMIND_OCR_SERVER_URL={ENV_OCR_URL!r}')
    print(f'keys: paddle={"set" if PADDLE_KEY else "empty"} mineru={"set" if MINERU_KEY else "empty"}')
    print()

    cases = [
        ('mineru_url_empty', lambda: MineruPDFReader(url='', dynamic_auth=True)),
        ('mineru_url_official', lambda: MineruPDFReader(url='https://mineru.net', dynamic_auth=True)),
        ('mineru_url_env_default', lambda: MineruPDFReader(url=ENV_OCR_URL, dynamic_auth=True)),
        ('paddle_url_empty', lambda: PaddleOCRPDFReader(url='', dynamic_auth=True)),
        (
            'paddle_url_official',
            lambda: PaddleOCRPDFReader(
                url='https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
                dynamic_auth=True,
            ),
        ),
        ('paddle_url_env_default', lambda: PaddleOCRPDFReader(url=ENV_OCR_URL, dynamic_auth=True)),
        ('mineru_no_dynamic_auth', lambda: MineruPDFReader(url='https://mineru.net', api_key=MINERU_KEY or None)),
        ('paddle_no_dynamic_auth', lambda: PaddleOCRPDFReader(url='', api_key=PADDLE_KEY or None)),
    ]

    readers = []
    for label, factory in cases:
        reader, err = run_init(label, factory)
        if reader is not None:
            readers.append((label, reader))

    if not PADDLE_KEY and not MINERU_KEY:
        print()
        print('SKIP parse: set PADDLE_KEY / MINERU_KEY env to test forward()')
        return 0

    print()
    print('=== parse with inject_ocr_config (dynamic_auth readers) ===')
    parse_cases = []
    if MINERU_KEY:
        parse_cases.extend([
            (
                'mineru_empty_url_parse',
                '',
                {'mineru': MINERU_KEY},
                lambda: MineruPDFReader(url='', dynamic_auth=True),
            ),
            (
                'mineru_official_url_parse',
                'https://mineru.net',
                {'mineru': MINERU_KEY},
                lambda: MineruPDFReader(url='https://mineru.net', dynamic_auth=True),
            ),
        ])
    if PADDLE_KEY:
        parse_cases.extend([
            (
                'paddle_empty_url_parse',
                '',
                {'paddleocr': PADDLE_KEY},
                lambda: PaddleOCRPDFReader(url='', dynamic_auth=True),
            ),
            (
                'paddle_official_url_parse',
                'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
                {'paddleocr': PADDLE_KEY},
                lambda: PaddleOCRPDFReader(
                    url='https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
                    dynamic_auth=True,
                ),
            ),
        ])

    failed = 0
    for label, ocr_url, auth, factory in parse_cases:
        clear_ocr_inject()
        inject_ocr_config({
            'ocr_type': 'mineru' if 'mineru' in label else 'paddleocr',
            'ocr_url': ocr_url,
            'ocr_auth': auth,
        })
        reader, err = run_init(label + '_before_parse', factory)
        if err:
            failed += 1
            continue
        if not run_parse(label, reader):
            failed += 1

    print()
    if failed:
        print(f'done: {failed} parse failure(s)')
        return 1
    print('done: all requested checks finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
