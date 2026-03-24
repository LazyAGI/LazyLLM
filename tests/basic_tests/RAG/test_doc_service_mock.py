import io
import os
import socket
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from lazyllm.tools.rag.doc_service import DocServer


@pytest.mark.skip_on_win
class TestDocServiceMock:
    @staticmethod
    def _ensure_bindable():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', 0))
        except OSError as exc:
            if 'operation not permitted' in str(exc).lower():
                pytest.skip('Socket bind is not permitted in current environment')
            raise
        finally:
            sock.close()

    @classmethod
    def setup_class(cls):
        cls._ensure_bindable()
        cls._parser_url = os.getenv('LAZYLLM_DOC_SERVICE_TEST_PARSER_URL')
        if not cls._parser_url:
            pytest.skip('LAZYLLM_DOC_SERVICE_TEST_PARSER_URL is required for real parser integration tests')
        cls._tmp_dir = tempfile.mkdtemp(prefix='lazyllm_doc_service_')
        cls._storage_dir = os.path.join(cls._tmp_dir, 'uploads')
        os.makedirs(cls._storage_dir, exist_ok=True)

        cls._seed_path = os.path.join(cls._tmp_dir, 'seed.txt')
        with open(cls._seed_path, 'w', encoding='utf-8') as f:
            f.write('seed content')

        cls._db_config = {
            'db_type': 'sqlite',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'db_name': os.path.join(cls._tmp_dir, 'doc_service.db'),
        }
        cls.server = DocServer(
            db_config=cls._db_config,
            parser_url=cls._parser_url,
            storage_dir=cls._storage_dir,
        )
        cls.server.start()
        cls.base_url = cls.server._impl._url.rsplit('/', 1)[0]
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                resp = requests.get(f'{cls.base_url}/v1/health', timeout=3)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)

    @classmethod
    def teardown_class(cls):
        cls.server.stop()
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def _wait_task(self, task_id, target_statuses, timeout=8):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            resp = requests.get(f'{self.base_url}/v1/tasks/{task_id}', timeout=5)
            assert resp.status_code == 200
            last = resp.json()['data']
            if last['status'] in target_statuses:
                return last
            time.sleep(0.1)
        raise AssertionError(f'task {task_id} not finished in time, last={last}')

    def test_p0_endpoints_and_core_flows(self):
        health = requests.get(f'{self.base_url}/v1/health', timeout=5)
        assert health.status_code == 200

        kb_create = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_a'}, timeout=5)
        assert kb_create.status_code == 200
        assert kb_create.json()['data']['kb_id'] == 'kb_a'

        kb_list = requests.get(f'{self.base_url}/v1/kbs', timeout=5)
        assert kb_list.status_code == 200
        assert any(item['kb_id'] == 'kb_a' for item in kb_list.json()['data']['items'])

        algo_list = requests.get(f'{self.base_url}/v1/algo/list', timeout=5)
        assert algo_list.status_code == 200
        assert any(item['algo_id'] == '__default__' for item in algo_list.json()['data'])

        algo_groups = requests.get(f'{self.base_url}/v1/algo/__default__/groups', timeout=5)
        assert algo_groups.status_code == 200
        assert len(algo_groups.json()['data']) > 0

        upload = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params={'kb_id': 'kb_a', 'algo_id': '__default__'},
            files=[('files', ('upload.txt', io.BytesIO(b'upload content'), 'text/plain'))],
            timeout=8,
        )
        assert upload.status_code == 200
        upload_item = upload.json()['data']['items'][0]
        doc_upload = upload_item['doc_id']
        upload_task = upload_item['task_id']
        self._wait_task(upload_task, {'SUCCESS'})

        add = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_a',
                'algo_id': '__default__',
                'items': [{'file_path': self._seed_path, 'doc_id': 'seed-doc-1', 'metadata': {'owner': 'u1'}}],
            },
            timeout=8,
        )
        assert add.status_code == 200
        add_item = add.json()['data']['items'][0]
        doc_add = add_item['doc_id']
        add_task = add_item['task_id']
        self._wait_task(add_task, {'SUCCESS'})

        meta_patch = requests.post(
            f'{self.base_url}/v1/docs/metadata/patch',
            json={
                'kb_id': 'kb_a',
                'algo_id': '__default__',
                'items': [{'doc_id': doc_add, 'patch': {'tag': 'patched'}}],
            },
            timeout=8,
        )
        assert meta_patch.status_code == 200
        meta_task = meta_patch.json()['data']['items'][0]['task_id']
        self._wait_task(meta_task, {'SUCCESS'})

        reparse = requests.post(
            f'{self.base_url}/v1/docs/reparse',
            json={'kb_id': 'kb_a', 'algo_id': '__default__', 'doc_ids': [doc_add]},
            timeout=8,
        )
        assert reparse.status_code == 200
        reparse_task = reparse.json()['data']['task_ids'][0]
        self._wait_task(reparse_task, {'SUCCESS'})

        kb_b = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_b'}, timeout=5)
        assert kb_b.status_code == 200

        transfer = requests.post(
            f'{self.base_url}/v1/docs/transfer',
            json={
                'items': [
                    {
                        'doc_id': doc_add,
                        'target_doc_id': 'copied-seed-doc-1',
                        'source_kb_id': 'kb_a',
                        'source_algo_id': '__default__',
                        'target_kb_id': 'kb_b',
                        'target_algo_id': '__default__',
                        'mode': 'copy',
                    }
                ]
            },
            timeout=8,
        )
        assert transfer.status_code == 200
        transfer_task = transfer.json()['data']['items'][0]['task_id']
        self._wait_task(transfer_task, {'SUCCESS'})

        docs = requests.get(
            f'{self.base_url}/v1/docs',
            params={'kb_id': 'kb_a', 'include_deleted_or_canceled': True, 'keyword': 'seed'},
            timeout=8,
        )
        assert docs.status_code == 200
        assert docs.json()['data']['total'] >= 1

        doc_detail = requests.get(f'{self.base_url}/v1/docs/{doc_add}', timeout=8)
        assert doc_detail.status_code == 200
        assert doc_detail.json()['data']['doc']['metadata'].get('tag') == 'patched'

        tasks = requests.get(f'{self.base_url}/v1/tasks', params={'status': ['SUCCESS', 'WAITING']}, timeout=8)
        assert tasks.status_code == 200
        assert tasks.json()['data']['total'] >= 1

        task_detail = requests.get(f'{self.base_url}/v1/tasks/{reparse_task}', timeout=8)
        assert task_detail.status_code == 200

        cancel = requests.post(f'{self.base_url}/v1/tasks/cancel', json={'task_id': reparse_task}, timeout=8)
        assert cancel.status_code == 200
        assert cancel.json()['data']['task_id'] == reparse_task

        delete = requests.post(
            f'{self.base_url}/v1/docs/delete',
            json={'kb_id': 'kb_a', 'algo_id': '__default__', 'doc_ids': [doc_upload]},
            timeout=8,
        )
        assert delete.status_code == 200
        delete_task = delete.json()['data']['items'][0]['task_id']
        self._wait_task(delete_task, {'DELETED'})

        docs_filtered = requests.get(
            f'{self.base_url}/v1/docs',
            params={'kb_id': 'kb_a', 'include_deleted_or_canceled': False},
            timeout=8,
        )
        assert docs_filtered.status_code == 200

        cb = requests.post(
            f'{self.base_url}/v1/internal/callbacks/tasks',
            json={
                'task_id': 'non-exist-task',
                'event_type': 'FINISH',
                'status': 'SUCCESS',
                'payload': {'task_type': 'DOC_ADD', 'doc_id': 'nope', 'kb_id': 'kb_a', 'algo_id': '__default__'},
            },
            timeout=8,
        )
        assert cb.status_code == 200
        assert cb.json()['data']['ack'] is True

        kb_delete = requests.delete(f'{self.base_url}/v1/kbs/kb_a', timeout=8)
        assert kb_delete.status_code == 200

    def test_document_manager_supports_doc_server_port(self):
        from lazyllm import Document

        tmp_dir = tempfile.mkdtemp(prefix='lazyllm_doc_port_')
        storage_dir = os.path.join(tmp_dir, 'uploads')
        os.makedirs(storage_dir, exist_ok=True)
        fixed_port = 18898
        doc = Document(dataset_path=storage_dir, manager=True, doc_server_port=fixed_port, name='doc_port_test')
        try:
            self._ensure_bindable()
            doc.start()
            base_url = doc.manager.url.rsplit('/', 1)[0]
            assert base_url.endswith(f':{fixed_port}')
            health = requests.get(f'{base_url}/v1/health', timeout=5)
            assert health.status_code == 200
        finally:
            doc.stop()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_missing_p0_endpoints_exist(self):
        kb_create = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_endpoints'}, timeout=5)
        assert kb_create.status_code == 200

        chunks = requests.get(f'{self.base_url}/v1/chunks', timeout=5)
        assert chunks.status_code == 200
        assert chunks.json()['data']['items'] == []

        algorithms = requests.get(f'{self.base_url}/v1/algorithms', timeout=5)
        assert algorithms.status_code == 200
        assert len(algorithms.json()['data']['items']) >= 1

        algo_info = requests.post(
            f'{self.base_url}/v1/algorithms/info', json={'algo_id': '__default__'}, timeout=5,
        )
        assert algo_info.status_code == 200
        assert algo_info.json()['data']['algo_id'] == '__default__'

        add = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_endpoints',
                'algo_id': '__default__',
                'items': [{'file_path': self._seed_path, 'doc_id': 'seed-doc-endpoints'}],
            },
            timeout=8,
        )
        assert add.status_code == 200
        task_id = add.json()['data']['items'][0]['task_id']

        task_info = requests.post(f'{self.base_url}/v1/tasks/info', json={'task_id': task_id}, timeout=5)
        assert task_info.status_code == 200
        assert task_info.json()['data']['task_id'] == task_id

        task_batch = requests.post(f'{self.base_url}/v1/tasks/batch', json={'task_ids': [task_id]}, timeout=5)
        assert task_batch.status_code == 200
        assert len(task_batch.json()['data']['items']) == 1

        kb_delete = requests.delete(f'{self.base_url}/v1/kbs', json={'kb_ids': ['kb_endpoints']}, timeout=8)
        assert kb_delete.status_code == 200
        assert len(kb_delete.json()['data']['items']) == 1

    def test_idempotency_replay_and_conflict(self):
        file_path = os.path.join(self._tmp_dir, 'idem.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('idempotent content')
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_idem'}, timeout=5)
        assert create_kb.status_code == 200

        payload = {
            'kb_id': 'kb_idem',
            'algo_id': '__default__',
            'idempotency_key': 'idem-add-key',
            'items': [{'file_path': file_path, 'doc_id': 'idem-doc-1'}],
        }
        first = requests.post(f'{self.base_url}/v1/docs/add', json=payload, timeout=8)
        second = requests.post(f'{self.base_url}/v1/docs/add', json=payload, timeout=8)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()['data']['items'][0]['task_id'] == second.json()['data']['items'][0]['task_id']

        conflict_payload = dict(payload)
        conflict_payload['items'] = [{'file_path': file_path, 'doc_id': 'idem-doc-2'}]
        conflict = requests.post(f'{self.base_url}/v1/docs/add', json=conflict_payload, timeout=8)
        assert conflict.status_code == 409
        assert conflict.json()['data']['biz_code'] == 'E_IDEMPOTENCY_CONFLICT'

    def test_upload_idempotency_replay_and_conflict(self):
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_upload_idem'}, timeout=5)
        assert create_kb.status_code == 200

        params = {
            'kb_id': 'kb_upload_idem',
            'algo_id': '__default__',
            'idempotency_key': 'idem-upload-key',
        }
        first = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params=params,
            files=[('files', ('idem-upload.txt', io.BytesIO(b'upload idem content'), 'text/plain'))],
            timeout=8,
        )
        second = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params=params,
            files=[('files', ('idem-upload.txt', io.BytesIO(b'upload idem content'), 'text/plain'))],
            timeout=8,
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()['data']['items'][0]['task_id'] == second.json()['data']['items'][0]['task_id']

        conflict = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params=params,
            files=[('files', ('idem-upload.txt', io.BytesIO(b'upload idem changed'), 'text/plain'))],
            timeout=8,
        )
        assert conflict.status_code == 409
        assert conflict.json()['data']['biz_code'] == 'E_IDEMPOTENCY_CONFLICT'

    def test_upload_same_filename_does_not_override_existing_file(self):
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_same_name'}, timeout=5)
        assert create_kb.status_code == 200

        first = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params={'kb_id': 'kb_same_name', 'algo_id': '__default__'},
            files=[('files', ('same-name.txt', io.BytesIO(b'first content'), 'text/plain'))],
            timeout=8,
        )
        second = requests.post(
            f'{self.base_url}/v1/docs/upload',
            params={'kb_id': 'kb_same_name', 'algo_id': '__default__'},
            files=[('files', ('same-name.txt', io.BytesIO(b'second content'), 'text/plain'))],
            timeout=8,
        )
        assert first.status_code == 200
        assert second.status_code == 200

        first_item = first.json()['data']['items'][0]
        second_item = second.json()['data']['items'][0]
        assert first_item['doc_id'] != second_item['doc_id']
        self._wait_task(first_item['task_id'], {'SUCCESS'})
        self._wait_task(second_item['task_id'], {'SUCCESS'})

        first_detail = requests.get(f'{self.base_url}/v1/docs/{first_item["doc_id"]}', timeout=8)
        second_detail = requests.get(f'{self.base_url}/v1/docs/{second_item["doc_id"]}', timeout=8)
        assert first_detail.status_code == 200
        assert second_detail.status_code == 200
        assert first_detail.json()['data']['doc']['path'] != second_detail.json()['data']['doc']['path']

    def test_idempotency_atomic_claim(self):
        file_path = os.path.join(self._tmp_dir, 'idem_atomic.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('idempotent atomic content')
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_idem_atomic'}, timeout=5)
        assert create_kb.status_code == 200

        payload = {
            'kb_id': 'kb_idem_atomic',
            'algo_id': '__default__',
            'idempotency_key': 'idem-atomic-key',
            'items': [{'file_path': file_path, 'doc_id': 'idem-atomic-doc'}],
        }

        def _send():
            return requests.post(f'{self.base_url}/v1/docs/add', json=payload, timeout=8)

        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: _send(), range(6)))

        statuses = [resp.status_code for resp in responses]
        assert all(status in (200, 409) for status in statuses)
        success_payloads = [resp.json()['data'] for resp in responses if resp.status_code == 200]
        unique_task_ids = {item['items'][0]['task_id'] for item in success_payloads}
        assert len(unique_task_ids) == 1
        for resp in responses:
            if resp.status_code == 409:
                assert resp.json()['data']['biz_code'] in {'E_IDEMPOTENCY_IN_PROGRESS', 'E_IDEMPOTENCY_CONFLICT'}

        replay = requests.post(f'{self.base_url}/v1/docs/add', json=payload, timeout=8)
        assert replay.status_code == 200
        assert replay.json()['data']['items'][0]['task_id'] in unique_task_ids

    def test_illegal_state_transition(self):
        file_path = os.path.join(self._tmp_dir, 'illegal.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('illegal transition content')
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_illegal'}, timeout=5)
        assert create_kb.status_code == 200

        add = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_illegal',
                'algo_id': '__default__',
                'items': [{'file_path': file_path, 'doc_id': 'illegal-doc-1'}],
            },
            timeout=8,
        )
        assert add.status_code == 200
        doc_id = add.json()['data']['items'][0]['doc_id']
        task_id = add.json()['data']['items'][0]['task_id']
        self._wait_task(task_id, {'SUCCESS'})

        delete = requests.post(
            f'{self.base_url}/v1/docs/delete',
            json={'kb_id': 'kb_illegal', 'algo_id': '__default__', 'doc_ids': [doc_id]},
            timeout=8,
        )
        assert delete.status_code == 200

        reparse_while_deleting = requests.post(
            f'{self.base_url}/v1/docs/reparse',
            json={'kb_id': 'kb_illegal', 'algo_id': '__default__', 'doc_ids': [doc_id]},
            timeout=8,
        )
        assert reparse_while_deleting.status_code == 409
        assert reparse_while_deleting.json()['data']['biz_code'] == 'E_STATE_CONFLICT'

        delete_again = requests.post(
            f'{self.base_url}/v1/docs/delete',
            json={'kb_id': 'kb_illegal', 'algo_id': '__default__', 'doc_ids': [doc_id]},
            timeout=8,
        )
        assert delete_again.status_code == 409

        add_again = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_illegal',
                'algo_id': '__default__',
                'items': [{'file_path': file_path, 'doc_id': doc_id}],
            },
            timeout=8,
        )
        assert add_again.status_code == 409

    def test_kb_algo_binding_and_transfer_validation(self):
        create_kb = requests.post(
            f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_bind', 'algo_id': '__default__'}, timeout=5,
        )
        assert create_kb.status_code == 200

        rebind = requests.post(
            f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_bind', 'algo_id': 'another_algo'}, timeout=5,
        )
        assert rebind.status_code == 409
        assert rebind.json()['data']['biz_code'] == 'E_STATE_CONFLICT'

        file_path = os.path.join(self._tmp_dir, 'binding.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('binding content')

        mismatch = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_bind',
                'algo_id': 'another_algo',
                'items': [{'file_path': file_path, 'doc_id': 'bind-doc'}],
            },
            timeout=8,
        )
        assert mismatch.status_code == 400
        assert mismatch.json()['data']['biz_code'] == 'E_INVALID_PARAM'

        add = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_bind',
                'algo_id': '__default__',
                'items': [{'file_path': file_path, 'doc_id': 'bind-doc'}],
            },
            timeout=8,
        )
        assert add.status_code == 200
        doc_id = add.json()['data']['items'][0]['doc_id']
        self._wait_task(add.json()['data']['items'][0]['task_id'], {'SUCCESS'})

        invalid_transfer = requests.post(
            f'{self.base_url}/v1/docs/transfer',
            json={
                'items': [{
                    'doc_id': doc_id,
                    'target_doc_id': 'invalid-transfer-doc',
                    'source_kb_id': 'kb_bind',
                    'source_algo_id': '__default__',
                    'target_kb_id': 'kb_bind',
                    'target_algo_id': '__default__',
                    'mode': 'invalid',
                }]
            },
            timeout=8,
        )
        assert invalid_transfer.status_code == 400
        assert invalid_transfer.json()['data']['biz_code'] == 'E_INVALID_PARAM'

    def test_stale_callback_ignored(self):
        file_path = os.path.join(self._tmp_dir, 'stale.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('stale callback content')
        create_kb = requests.post(f'{self.base_url}/v1/kbs', json={'kb_id': 'kb_stale'}, timeout=5)
        assert create_kb.status_code == 200

        add = requests.post(
            f'{self.base_url}/v1/docs/add',
            json={
                'kb_id': 'kb_stale',
                'algo_id': '__default__',
                'items': [{'file_path': file_path, 'doc_id': 'stale-doc-1'}],
            },
            timeout=8,
        )
        assert add.status_code == 200
        doc_id = add.json()['data']['items'][0]['doc_id']
        self._wait_task(add.json()['data']['items'][0]['task_id'], {'SUCCESS'})

        first = requests.post(
            f'{self.base_url}/v1/docs/reparse',
            json={'kb_id': 'kb_stale', 'algo_id': '__default__', 'doc_ids': [doc_id]},
            timeout=8,
        )
        assert first.status_code == 200
        first_task_id = first.json()['data']['task_ids'][0]

        second = requests.post(
            f'{self.base_url}/v1/docs/reparse',
            json={'kb_id': 'kb_stale', 'algo_id': '__default__', 'doc_ids': [doc_id]},
            timeout=8,
        )
        assert second.status_code == 200
        second_task_id = second.json()['data']['task_ids'][0]
        assert first_task_id != second_task_id

        stale = requests.post(
            f'{self.base_url}/v1/internal/callbacks/tasks',
            json={
                'callback_id': 'stale-callback-1',
                'task_id': first_task_id,
                'event_type': 'FINISH',
                'status': 'SUCCESS',
            },
            timeout=8,
        )
        assert stale.status_code == 200
        assert stale.json()['data']['ignored_reason'] == 'stale_task_callback'

        duplicate = requests.post(
            f'{self.base_url}/v1/internal/callbacks/tasks',
            json={
                'callback_id': 'stale-callback-1',
                'task_id': first_task_id,
                'event_type': 'FINISH',
                'status': 'SUCCESS',
            },
            timeout=8,
        )
        assert duplicate.status_code == 200
        assert duplicate.json()['data']['deduped'] is True

    def test_get_doc_404_is_wrapped(self):
        resp = requests.get(f'{self.base_url}/v1/docs/not-exists-doc', timeout=5)
        assert resp.status_code == 404
        body = resp.json()
        assert body['code'] == 404
        assert body['data']['biz_code'] == 'E_NOT_FOUND'

    def test_delete_kbs_empty_payload_returns_400(self):
        resp = requests.delete(f'{self.base_url}/v1/kbs', json={'kb_ids': []}, timeout=5)
        assert resp.status_code == 400
        assert resp.json()['data']['biz_code'] == 'E_INVALID_PARAM'

    def test_kb_update_pagination_and_batch_query(self):
        first = requests.post(
            f'{self.base_url}/v1/kbs',
            json={'kb_id': 'kb_page_1', 'display_name': 'Page 1', 'algo_id': '__default__'},
            timeout=5,
        )
        second = requests.post(
            f'{self.base_url}/v1/kbs',
            json={'kb_id': 'kb_page_2', 'display_name': 'Page 2', 'algo_id': '__default__'},
            timeout=5,
        )
        assert first.status_code == 200
        assert second.status_code == 200

        paged = requests.get(f'{self.base_url}/v1/kbs', params={'page': 1, 'page_size': 1}, timeout=5)
        assert paged.status_code == 200
        paged_data = paged.json()['data']
        assert paged_data['page'] == 1
        assert paged_data['page_size'] == 1
        assert paged_data['total'] >= 2
        assert len(paged_data['items']) == 1

        detail = requests.get(f'{self.base_url}/v1/kbs/kb_page_1', timeout=5)
        assert detail.status_code == 200
        assert detail.json()['data']['algo_id'] == '__default__'

        updated = requests.post(
            f'{self.base_url}/v1/kbs/kb_page_1/update',
            json={
                'display_name': 'Page 1 Updated',
                'description': 'updated description',
                'owner_id': 'owner-a',
                'meta': {'scene': 'pagination-test'},
            },
            timeout=5,
        )
        assert updated.status_code == 200
        updated_data = updated.json()['data']
        assert updated_data['display_name'] == 'Page 1 Updated'
        assert updated_data['meta']['scene'] == 'pagination-test'

        batch = requests.post(
            f'{self.base_url}/v1/kbs/batch',
            json={'kb_ids': ['kb_page_1', 'kb_missing']},
            timeout=5,
        )
        assert batch.status_code == 200
        batch_data = batch.json()['data']
        assert len(batch_data['items']) == 1
        assert batch_data['items'][0]['kb_id'] == 'kb_page_1'
        assert batch_data['missing_kb_ids'] == ['kb_missing']
