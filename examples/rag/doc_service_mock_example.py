'''DocService mock quickstart.

Run:
    python examples/rag/doc_service_mock_example.py
'''

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time

import requests

from lazyllm import Document


def _wait_task(base_url: str, task_id: str, targets: set[str], timeout: float = 10.0):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f'{base_url}/v1/tasks/{task_id}', timeout=5)
        resp.raise_for_status()
        task = resp.json()['data']
        if task['status'] in targets:
            return task
        time.sleep(0.2)
    raise TimeoutError(f'task {task_id} did not reach {targets}')


def main():
    parser = argparse.ArgumentParser(description='DocService mock quickstart.')
    parser.add_argument('--wait', action='store_true', help='Keep server alive for manual API/docs inspection.')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix='lazyllm_doc_service_demo_') as tmp:
        storage = os.path.join(tmp, 'uploads')
        os.makedirs(storage, exist_ok=True)
        seed_path = os.path.join(storage, 'seed.txt')
        with open(seed_path, 'w', encoding='utf-8') as f:
            f.write('seed content')

        doc = Document(
            dataset_path=storage,
            manager=True,
            name='demo_doc_service',
        )
        doc.start()

        try:
            base_url = doc.manager.url.rsplit('/', 1)[0]
            print(f'DocService URL: {base_url}')
            print(f'Swagger Docs: {base_url}/docs')

            upload_resp = requests.post(
                f'{base_url}/v1/docs/upload',
                params={'kb_id': 'kb_demo', 'algo_id': '__default__'},
                files=[('files', ('demo.txt', io.BytesIO(b'hello lazyllm rag'), 'text/plain'))],
                timeout=10,
            )
            upload_resp.raise_for_status()
            upload_item = upload_resp.json()['data']['items'][0]
            doc_id = upload_item['doc_id']
            task_id = upload_item['task_id']
            _wait_task(base_url, task_id, {'SUCCESS'})

            patch_resp = requests.post(
                f'{base_url}/v1/docs/metadata/patch',
                json={
                    'kb_id': 'kb_demo',
                    'algo_id': '__default__',
                    'items': [{'doc_id': doc_id, 'patch': {'owner': 'demo_user', 'scene': 'quickstart'}}],
                },
                timeout=10,
            )
            patch_resp.raise_for_status()
            patch_task = patch_resp.json()['data']['items'][0]['task_id']
            _wait_task(base_url, patch_task, {'SUCCESS'})

            reparse_resp = requests.post(
                f'{base_url}/v1/docs/reparse',
                json={'kb_id': 'kb_demo', 'algo_id': '__default__', 'doc_ids': [doc_id]},
                timeout=10,
            )
            reparse_resp.raise_for_status()
            reparse_task = reparse_resp.json()['data']['task_ids'][0]
            _wait_task(base_url, reparse_task, {'SUCCESS'})

            add_resp = requests.post(
                f'{base_url}/v1/docs/add',
                json={'kb_id': 'kb_demo', 'algo_id': '__default__', 'items': [{'file_path': seed_path}]},
                timeout=10,
            )
            add_resp.raise_for_status()
            add_task = add_resp.json()['data']['items'][0]['task_id']
            _wait_task(base_url, add_task, {'SUCCESS'})

            docs_resp = requests.get(
                f'{base_url}/v1/docs',
                params={'kb_id': 'kb_demo', 'include_deleted_or_canceled': False},
                timeout=10,
            )
            docs_resp.raise_for_status()
            docs = docs_resp.json()['data']['items']
            print(f'Current docs in kb_demo: {len(docs)}')

            delete_resp = requests.post(
                f'{base_url}/v1/docs/delete',
                json={'kb_id': 'kb_demo', 'algo_id': '__default__', 'doc_ids': [doc_id]},
                timeout=10,
            )
            delete_resp.raise_for_status()
            delete_task = delete_resp.json()['data']['items'][0]['task_id']
            _wait_task(base_url, delete_task, {'DELETED'})
            print('Doc lifecycle demo completed.')
            if args.wait:
                print('Server is running. Press Ctrl+C to stop...')
                while True:
                    time.sleep(1)
        finally:
            doc.stop()


if __name__ == '__main__':
    main()
