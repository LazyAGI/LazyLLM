'''Start standalone DocService mock server.

Run:
    python examples/rag/doc_service_standalone.py --wait
'''

from __future__ import annotations

import argparse
import os
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description='Standalone DocService mock server.')
    parser.add_argument('--port', type=int, default=None, help='DocServer listen port.')
    parser.add_argument('--wait', action='store_true', help='Keep server alive for manual API/docs inspection.')
    args = parser.parse_args()

    from lazyllm.tools.rag.doc_service import DocServer

    tmp_dir = tempfile.mkdtemp(prefix='lazyllm_doc_service_standalone_')
    storage_dir = os.path.join(tmp_dir, 'uploads')
    os.makedirs(storage_dir, exist_ok=True)
    db_config = {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': os.path.join(tmp_dir, 'doc_service.db'),
    }
    parser_db_config = {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': os.path.join(tmp_dir, 'doc_service_parser.db'),
    }

    server = DocServer(
        storage_dir=storage_dir,
        db_config=db_config,
        parser_db_config=parser_db_config,
        port=args.port,
    )
    server.start()
    base_url = server.url.rsplit('/', 1)[0]
    print(f'DocService URL: {base_url}', flush=True)
    print(f'Swagger Docs: {base_url}/docs', flush=True)
    print(f'Storage Dir: {storage_dir}', flush=True)
    print(f'Doc DB: {db_config["db_name"]}', flush=True)
    print(f'Parser DB: {parser_db_config["db_name"]}', flush=True)

    try:
        if args.wait:
            print('Server is running. Press Ctrl+C to stop...', flush=True)
            while True:
                time.sleep(1)
    finally:
        server.stop()


if __name__ == '__main__':
    main()
