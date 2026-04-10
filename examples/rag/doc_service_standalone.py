'''Start a standalone DocServer example.

Examples:
    python examples/rag/doc_service_standalone.py --wait
    python examples/rag/doc_service_standalone.py --parser-url http://127.0.0.1:9966 --wait
    python examples/rag/doc_service_standalone.py --export-openapi
'''

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import requests

from lazyllm import Document
from lazyllm.tools.rag.doc_service import DocServer
from lazyllm.tools.rag.doc_service.doc_server import DEFAULT_OPENAPI_OUTPUT_PATH
from lazyllm.tools.rag.parsing_service import DocumentProcessor

REAL_ALGO_ID = 'real-standalone-algo'
FIXED_DB_ROOT = './tmp/db'
DEFAULT_OPENAPI_PATH = DEFAULT_OPENAPI_OUTPUT_PATH


def _make_db_config(db_name: str) -> Dict[str, Any]:
    return {'db_type': 'sqlite', 'user': None, 'password': None, 'host': None, 'port': None, 'db_name': db_name}


def _prepare_runtime_paths() -> Dict[str, str]:
    os.makedirs(FIXED_DB_ROOT, exist_ok=True)
    paths = {
        'root_dir': FIXED_DB_ROOT,
        'storage_dir': os.path.join(FIXED_DB_ROOT, 'uploads'),
        'store_dir': os.path.join(FIXED_DB_ROOT, 'store'),
        'doc_db': os.path.join(FIXED_DB_ROOT, 'doc_service.sqlite'),
        'parser_db': os.path.join(FIXED_DB_ROOT, 'parser.sqlite'),
    }
    os.makedirs(paths['storage_dir'], exist_ok=True)
    os.makedirs(paths['store_dir'], exist_ok=True)
    return paths


def _wait_http_ok(url: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f'http service is not ready: {url}')


def _build_store_conf(root_dir: str) -> Dict[str, Any]:
    segment_store_path = os.path.join(root_dir, 'segments.db')
    milvus_store_path = os.path.join(root_dir, 'milvus_lite.db')
    Path(segment_store_path).touch()
    Path(milvus_store_path).touch()
    return {
        'segment_store': {'type': 'map', 'kwargs': {'uri': segment_store_path}},
        'vector_store': {
            'type': 'milvus',
            'kwargs': {
                'uri': milvus_store_path,
                'index_kwargs': {'index_type': 'FLAT', 'metric_type': 'COSINE'},
            },
        },
    }


def _wait_algo_ready(parser_url: str, algo_id: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f'{parser_url}/algo/list', timeout=5)
        if response.status_code == 200:
            items = response.json().get('data', [])
            if any(item.get('algo_id') == algo_id for item in items):
                return
        time.sleep(0.2)
    raise RuntimeError(f'algorithm is not ready: {algo_id}')


def _register_demo_algorithm(parser_url: str, algo_id: str, store_dir: str):
    # Step 2: register a real Document algorithm on the parsing service.
    document = Document(
        dataset_path=None,
        name=algo_id,
        embed={'vec_dense': lambda text: [1.0, 2.0, 3.0]},
        store_conf=_build_store_conf(store_dir),
        display_name='Standalone Real Algo',
        manager=DocumentProcessor(url=parser_url),
        description='Algorithm registered by standalone doc service example',
    )
    document.create_node_group(
        name='line',
        transform=lambda text: text.split('\n'),
        parent='CoarseChunk',
        display_name='Line Chunk',
    )
    document.activate_group('CoarseChunk', embed_keys=['vec_dense'])
    document.activate_group('line', embed_keys=['vec_dense'])
    document.start()
    _wait_algo_ready(parser_url, algo_id)
    return document


def _start_local_parser(parser_port: int, parser_db: str):
    # Step 1: start a local parsing service.
    parser = DocumentProcessor(port=parser_port, db_config=_make_db_config(parser_db), num_workers=1)
    parser.start()
    parser_url = parser._impl._url.rsplit('/', 1)[0]
    _wait_http_ok(f'{parser_url}/health')
    return parser, parser_url


def main():
    parser = argparse.ArgumentParser(description='Standalone DocServer example.')
    parser.add_argument('--port', type=int, default=8848, help='DocServer listen port.')
    parser.add_argument('--parser-port', type=int, default=9966, help='DocumentProcessor listen port.')
    parser.add_argument('--parser-url', type=str, default=None, help='Existing parsing service base URL.')
    parser.add_argument('--algo-id', type=str, default=REAL_ALGO_ID, help='Algorithm ID for the local demo setup.')
    parser.add_argument('--wait', action='store_true', help='Keep the example running for manual inspection.')
    parser.add_argument(
        '--export-openapi',
        type=str,
        nargs='?',
        const=DEFAULT_OPENAPI_PATH,
        default=None,
        help=f'Export DocServer OpenAPI JSON and exit. Default path: {DEFAULT_OPENAPI_PATH}',
    )
    args = parser.parse_args()

    if args.export_openapi:
        print(f'OpenAPI exported: {DocServer.export_openapi(args.export_openapi)}', flush=True)
        return

    paths = _prepare_runtime_paths()
    parser_server = None
    server = None
    document = None
    parser_url = args.parser_url

    try:
        if parser_url:
            parser_url = parser_url.rstrip('/')
            _wait_http_ok(f'{parser_url}/health')
            _wait_algo_ready(parser_url, args.algo_id)
        else:
            parser_server, parser_url = _start_local_parser(args.parser_port, paths['parser_db'])
            document = _register_demo_algorithm(parser_url, args.algo_id, paths['store_dir'])

        # Step 3: start DocServer and point it to the parsing service.
        server = DocServer(
            storage_dir=paths['storage_dir'],
            db_config=_make_db_config(paths['doc_db']),
            parser_url=parser_url,
            port=args.port,
        )
        server.start()
        base_url = server.url.rsplit('/', 1)[0]
        _wait_http_ok(f'{base_url}/v1/health')

        print(f'DocServer URL: {base_url}', flush=True)
        print(f'DocServer Docs: {base_url}/docs', flush=True)
        print(f'Parser URL: {parser_url}', flush=True)
        print(f'Storage Dir: {paths["storage_dir"]}', flush=True)
        print(f'Doc DB: {paths["doc_db"]}', flush=True)
        print(f'Algorithm ID: {args.algo_id}', flush=True)

        if args.wait:
            # Step 4: keep the services alive for manual API testing.
            print('Services are running. Press Ctrl+C to stop.', flush=True)
            while True:
                time.sleep(1)
    finally:
        if server:
            try:
                server.stop()
            except Exception:
                pass
        if document:
            try:
                document.stop()
            except Exception:
                pass
        if parser_server:
            try:
                parser_server.stop()
            except Exception:
                pass


if __name__ == '__main__':
    main()
