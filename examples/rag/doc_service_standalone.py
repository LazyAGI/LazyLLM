'''Start standalone DocService.

Modes:
1. Full stack mode (default): starts algorithm registration, real
   DocumentProcessor, and DocServer in one process.
2. External parser mode: starts only DocServer and connects to an existing
   parsing service with ``--parser-url``.

Run:
    python examples/rag/doc_service_standalone.py --wait
    python examples/rag/doc_service_standalone.py --parser-url http://127.0.0.1:9966 --wait
'''

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, Dict

import requests

from lazyllm import Document
from lazyllm.tools.rag.doc_service import DocServer
from lazyllm.tools.rag.parsing_service import DocumentProcessor

REAL_ALGO_ID = 'real-standalone-algo'
FIXED_DB_ROOT = './tmp/db'
DEFAULT_OPENAPI_PATH = os.path.join(FIXED_DB_ROOT, 'doc_service.openapi.json')


def _make_db_config(db_name: str) -> Dict[str, Any]:
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_name,
    }


def _prepare_runtime_paths() -> Dict[str, str]:
    os.makedirs(FIXED_DB_ROOT, exist_ok=True)
    paths = {
        'root_dir': FIXED_DB_ROOT,
        'storage_dir': os.path.join(FIXED_DB_ROOT, 'uploads'),
        'store_dir': os.path.join(FIXED_DB_ROOT, 'store'),
        'parser_db': os.path.join(FIXED_DB_ROOT, 'parser.sqlite'),
        'doc_db': os.path.join(FIXED_DB_ROOT, 'doc_service.sqlite'),
    }
    os.makedirs(paths['storage_dir'], exist_ok=True)
    os.makedirs(paths['store_dir'], exist_ok=True)
    return paths


def _wait_until(predicate, timeout: float = 20.0, interval: float = 0.1):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise RuntimeError(f'condition not satisfied before timeout, last={last!r}')


def _wait_http_ok(url: str, timeout: float = 20.0):
    def _poll():
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return resp
        except Exception:
            return None
        return None

    return _wait_until(_poll, timeout=timeout)


def _build_store_conf(root_dir: str) -> Dict[str, Any]:
    segment_store_path = os.path.join(root_dir, 'segments.db')
    milvus_store_path = os.path.join(root_dir, 'milvus_lite.db')
    open(segment_store_path, 'a', encoding='utf-8').close()
    open(milvus_store_path, 'a', encoding='utf-8').close()
    return {
        'segment_store': {
            'type': 'map',
            'kwargs': {'uri': segment_store_path},
        },
        'vector_store': {
            'type': 'milvus',
            'kwargs': {
                'uri': milvus_store_path,
                'index_kwargs': {
                    'index_type': 'FLAT',
                    'metric_type': 'COSINE',
                },
            },
        },
    }


def _start_full_stack(args):
    paths = _prepare_runtime_paths()

    parser = DocumentProcessor(
        port=args.parser_port,
        db_config=_make_db_config(paths['parser_db']),
        num_workers=args.num_workers,
    )
    parser.start()
    parser_base_url = parser._impl._url.rsplit('/', 1)[0]
    _wait_http_ok(f'{parser_base_url}/health')

    store_conf = _build_store_conf(paths['store_dir'])
    document = Document(
        dataset_path=None,
        name=args.algo_id,
        embed={'vec_dense': lambda x: [1.0, 2.0, 3.0]},
        store_conf=store_conf,
        display_name='Standalone Real Algo',
        manager=DocumentProcessor(url=parser_base_url),
        description='Algorithm registered by standalone doc service example',
    )
    document.create_node_group(
        name='line',
        transform=lambda x: x.split('\n'),
        parent='CoarseChunk',
        display_name='Line Chunk',
    )
    document.activate_group('CoarseChunk', embed_keys=['vec_dense'])
    document.activate_group('line', embed_keys=['vec_dense'])
    document.start()

    _wait_until(
        lambda: any(
            item.get('algo_id') == args.algo_id
            for item in requests.get(f'{parser_base_url}/algo/list', timeout=5).json().get('data', [])
        )
    )

    server = DocServer(
        storage_dir=paths['storage_dir'],
        db_config=_make_db_config(paths['doc_db']),
        parser_url=parser_base_url,
        port=args.port,
    )
    server.start()
    base_url = server.url.rsplit('/', 1)[0]
    _wait_http_ok(f'{base_url}/v1/health')

    print(f'DocService URL: {base_url}', flush=True)
    print(f'DocService Docs: {base_url}/docs', flush=True)
    print(f'Parser URL: {parser_base_url}', flush=True)
    print(f'Parser Docs: {parser_base_url}/docs', flush=True)
    print(f'Algorithm ID: {args.algo_id}', flush=True)
    print(f'Storage Dir: {paths["storage_dir"]}', flush=True)
    print(f'Store Dir: {paths["store_dir"]}', flush=True)
    print(f'Doc DB: {paths["doc_db"]}', flush=True)
    print(f'Parser DB: {paths["parser_db"]}', flush=True)
    print(f'DB Root: {paths["root_dir"]}', flush=True)

    try:
        if args.wait:
            print('Full stack is running. Press Ctrl+C to stop...', flush=True)
            threading.Event().wait()
    finally:
        server.stop()
        try:
            parser.drop_algorithm(args.algo_id)
        except Exception:
            pass
        parser.stop()


def _start_doc_server_only(args):
    paths = _prepare_runtime_paths()
    server = DocServer(
        storage_dir=paths['storage_dir'],
        db_config=_make_db_config(paths['doc_db']),
        parser_url=args.parser_url,
        port=args.port,
    )
    server.start()
    base_url = server.url.rsplit('/', 1)[0]
    print(f'DocService URL: {base_url}', flush=True)
    print(f'DocService Docs: {base_url}/docs', flush=True)
    print(f'Parser URL: {args.parser_url}', flush=True)
    print(f'Storage Dir: {paths["storage_dir"]}', flush=True)
    print(f'Doc DB: {paths["doc_db"]}', flush=True)
    print(f'DB Root: {paths["root_dir"]}', flush=True)

    try:
        if args.wait:
            print('DocService is running. Press Ctrl+C to stop...', flush=True)
            while True:
                time.sleep(1)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description='Standalone DocService server.')
    parser.add_argument('--port', type=int, default=8848, help='DocServer listen port.')
    parser.add_argument('--parser-port', type=int, default=9966, help='DocumentProcessor listen port.')
    parser.add_argument('--parser-url', type=str, default=None, help='Existing parsing service base URL.')
    parser.add_argument('--algo-id', type=str, default=REAL_ALGO_ID, help='Algorithm id to register in full stack mode.')
    parser.add_argument('--num-workers', type=int, default=1, help='DocumentProcessor worker count.')
    parser.add_argument('--wait', action='store_true', help='Keep server alive for manual API inspection.')
    parser.add_argument(
        '--export-openapi',
        type=str,
        default=None,
        help=f'Export current DocService OpenAPI JSON before startup. Default path example: {DEFAULT_OPENAPI_PATH}',
    )
    args = parser.parse_args()

    if args.export_openapi:
        output_path = DocServer.export_openapi(args.export_openapi)
        print(f'OpenAPI exported: {output_path}', flush=True)
        return

    if args.parser_url:
        _start_doc_server_only(args)
    else:
        _start_full_stack(args)


if __name__ == '__main__':
    main()
