#!/usr/bin/env python3
'''Smoke-test direct OCR reader init/parse in container (llm_config inject format).'''
import os
import sys
import traceback

from lazyllm import globals, inject_ocr_config
from lazyllm.tools.rag.readers.ocrReader.dynamic_pdf_reader import DynamicPDFReader
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
MINERU_LOCAL = os.environ.get(
    'MINERU_LOCAL',
    'http://172.24.176.1:20234/api/v1/pdf_parse',
)
IMAGE_CACHE_DIR = os.environ.get(
    'OCR_CACHE_DIR',
    '/var/lib/lazymind/uploads/.image_cache',
)


def clear_ocr_inject():
    globals.config['dynamic_ocr_configs'] = None
    globals.config['dynamic_ocr_auth'] = None


def reader_attrs(reader):
    if isinstance(reader, DynamicPDFReader):
        return {'route': reader._resolve_route(None)}
    if isinstance(reader, MineruPDFReader):
        return {
            '_url': reader._url,
            '_offline_mode': reader._offline_mode,
            '_image_cache_dir': str(reader._image_cache_dir),
        }
    if isinstance(reader, PaddleOCRPDFReader):
        return {
            '_url': reader._url,
            '_job_url': reader._job_url,
            '_image_cache_dir': str(reader._image_cache_dir),
        }
    return {}


def run_init(label, factory):
    try:
        reader = factory()
        print(f'OK   init[{label}] {reader_attrs(reader)}')
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


def inject_llm(llm_config):
    clear_ocr_inject()
    inject_ocr_config(llm_config)


def build_dynamic():
    return DynamicPDFReader(
        ocr_type='none',
        ocr_url=ENV_OCR_URL,
        image_cache_dir=IMAGE_CACHE_DIR,
        timeout=3600,
    )


def main():
    failed = 0
    print('=== direct reader init / parse ===')
    print(f'PDF exists: {os.path.isfile(PDF)} path={PDF}')
    print(f'ENV LAZYMIND_OCR_SERVER_URL={ENV_OCR_URL!r}')
    print(f'MINERU_LOCAL={MINERU_LOCAL!r}')
    print(f'IMAGE_CACHE_DIR={IMAGE_CACHE_DIR!r}')
    print(f'keys: paddle={"set" if PADDLE_KEY else "empty"} mineru={"set" if MINERU_KEY else "empty"}')
    print()

    print('=== Part 1: direct init (no inject) ===')
    init_cases = [
        ('mineru_url_empty', lambda: MineruPDFReader(url='', dynamic_auth=True)),
        ('mineru_url_local', lambda: MineruPDFReader(url=MINERU_LOCAL, dynamic_auth=True)),
        ('mineru_url_env', lambda: MineruPDFReader(url=ENV_OCR_URL, dynamic_auth=True)),
        ('paddle_url_empty', lambda: PaddleOCRPDFReader(url='', dynamic_auth=True)),
        (
            'paddle_url_official',
            lambda: PaddleOCRPDFReader(
                url='https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
                dynamic_auth=True,
            ),
        ),
        ('paddle_url_env', lambda: PaddleOCRPDFReader(url=ENV_OCR_URL, dynamic_auth=True)),
    ]
    for label, factory in init_cases:
        _, err = run_init(label, factory)
        if err:
            failed += 1

    if not PADDLE_KEY and not MINERU_KEY:
        print()
        print('SKIP parse: set PADDLE_KEY / MINERU_KEY env to test forward()')
        return 1 if failed else 0

    print()
    print('=== Part 2: DynamicPDFReader route + parse (llm_config inject) ===')
    dynamic = build_dynamic()
    dynamic_cases = [
        ('none', {'ocr_config': {'ocr_type': 'none', 'ocr_url': ''}}, True),
        (
            'paddle_key_only',
            {'ocr_config': {'ocr_type': 'paddleocr', 'paddle_api_key': PADDLE_KEY}},
            bool(PADDLE_KEY),
        ),
        (
            'paddle_alias',
            {'ocr_config': {'ocr_type': 'paddle', 'paddle_api_key': PADDLE_KEY}},
            bool(PADDLE_KEY),
        ),
        (
            'mineru_local',
            {
                'ocr_config': {
                    'ocr_type': 'mineru',
                    'ocr_url': MINERU_LOCAL,
                    'mineru_api_key': MINERU_KEY,
                },
            },
            bool(MINERU_KEY),
        ),
        (
            'mineru_key_only_offline_fallback',
            {
                'ocr_config': {
                    'ocr_type': 'mineru',
                    'ocr_url': MINERU_LOCAL,
                    'mineru_api_key': MINERU_KEY,
                },
            },
            bool(MINERU_KEY),
        ),
    ]
    for label, llm_config, enabled in dynamic_cases:
        if not enabled:
            print(f'SKIP [{label}] missing key')
            continue
        inject_llm(llm_config)
        route = dynamic._resolve_route(None)
        child = dynamic._get_reader(*route)
        print(f'route[{label}] -> {route} child={reader_attrs(child)}')
        if not run_parse(label, dynamic):
            failed += 1

    print()
    print('=== Part 3: direct reader parse + llm_config inject ===')
    direct_cases = [
        (
            'paddle_empty',
            {'ocr_config': {'ocr_type': 'paddleocr', 'paddle_api_key': PADDLE_KEY}},
            lambda: PaddleOCRPDFReader(url='', dynamic_auth=True),
            bool(PADDLE_KEY),
        ),
        (
            'mineru_local',
            {
                'ocr_config': {
                    'ocr_type': 'mineru',
                    'ocr_url': MINERU_LOCAL,
                    'mineru_api_key': MINERU_KEY,
                },
            },
            lambda: MineruPDFReader(url=MINERU_LOCAL, dynamic_auth=True),
            bool(MINERU_KEY),
        ),
    ]
    for label, llm_config, factory, enabled in direct_cases:
        if not enabled:
            print(f'SKIP [{label}] missing key')
            continue
        inject_llm(llm_config)
        reader, err = run_init(label, factory)
        if err:
            failed += 1
            continue
        if not run_parse(label, reader):
            failed += 1

    print()
    if failed:
        print(f'done: {failed} failure(s)')
        return 1
    print('done: all requested checks finished')
    return 0


if __name__ == '__main__':
    sys.exit(main())
