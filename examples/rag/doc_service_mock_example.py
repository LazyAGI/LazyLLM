'''Connect a Document to a deployed DocServer.

Start DocServer first:
    python examples/rag/doc_service_standalone.py --wait

Run this example:
    python examples/rag/doc_service_mock_example.py --doc-server-url http://127.0.0.1:8848
'''

from __future__ import annotations

import argparse
import os
import tempfile
import time

from lazyllm import Document
from lazyllm.tools.rag.doc_service import DocServer
from lazyllm.tools.rag.doc_service.base import AddFileItem, AddRequest


def _normalize_base_url(url: str) -> str:
    url = url.rstrip('/')
    if url.endswith('/_call') or url.endswith('/generate'):
        return url.rsplit('/', 1)[0]
    return url


def _wait_task(server: DocServer, task_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = server.get_task(task_id)['data']
        if task['status'] in {'SUCCESS', 'FAILED', 'CANCELED', 'DELETED'}:
            return task
        time.sleep(0.5)
    raise TimeoutError(f'task {task_id} did not finish in time')


def main():
    parser = argparse.ArgumentParser(description='Connect a Document to an existing DocServer.')
    parser.add_argument('--doc-server-url', type=str, required=True, help='Existing DocServer base URL.')
    parser.add_argument('--algo-id', type=str, default='doc_service_demo_algo', help='Document algorithm ID.')
    parser.add_argument('--kb-id', type=str, default='__default__', help='Knowledge base ID.')
    args = parser.parse_args()

    base_url = _normalize_base_url(args.doc_server_url)
    doc_server = DocServer(url=base_url)
    with tempfile.TemporaryDirectory(prefix='lazyllm_doc_service_example_') as dataset_dir:
        file_path = os.path.join(dataset_dir, 'demo.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('hello from a real doc_service example\n')

        # Step 1: create a Document and bind it to the deployed DocServer.
        document = Document(dataset_path=dataset_dir, manager=doc_server, name=args.algo_id)
        document.start()

        try:
            print(f'DocServer URL: {base_url}')
            print(f'DocServer Docs: {base_url}/docs')

            # Step 2: add a local file through the DocServer client.
            response = doc_server.add(AddRequest(
                kb_id=args.kb_id,
                algo_id=args.algo_id,
                items=[AddFileItem(file_path=file_path)],
            ))
            item = response['data']['items'][0]
            print(f'Doc ID: {item["doc_id"]}')
            print(f'Task ID: {item["task_id"]}')

            # Step 3: wait for the asynchronous parse task to finish.
            task = _wait_task(doc_server, item['task_id'])
            print(f'Task Status: {task["status"]}')

            # Step 4: list documents from the target knowledge base.
            docs = doc_server.list_docs(
                kb_id=args.kb_id, algo_id=args.algo_id, include_deleted_or_canceled=False
            )['data']['items']
            print(f'Doc Count In {args.kb_id}: {len(docs)}')
        finally:
            document.stop()


if __name__ == '__main__':
    main()
