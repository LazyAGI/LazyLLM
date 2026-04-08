import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from uuid import uuid4

import pytest

from lazyllm.tools.rag.doc_service.base import (
    AddFileItem,
    CallbackEventType,
    DeleteRequest,
    DocServiceError,
    DocStatus,
    ReparseRequest,
    SourceType,
    TaskCallbackRequest,
    TransferItem,
    TransferRequest,
    UploadRequest,
)
from lazyllm.tools.rag.doc_service.doc_manager import DocManager
from lazyllm.tools.rag.doc_service.parser_client import ParserClient
from lazyllm.tools.rag.parsing_service.base import TaskType
from lazyllm.tools.rag.utils import BaseResponse


class _ManagerHarness:
    def __init__(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix='lazyllm_doc_service_manager_')
        self.tmp_dir = self._tmp_dir.name
        self.seed_path = self.make_file('seed.txt', 'seed content')
        self.db_config = {
            'db_type': 'sqlite',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'db_name': os.path.join(self.tmp_dir, 'doc_service_local.db'),
        }
        original_health = ParserClient.health
        ParserClient.health = lambda self: BaseResponse(code=200, msg='success', data={'ok': True})
        try:
            self.manager = DocManager(db_config=self.db_config, parser_url='http://parser.test')
        finally:
            ParserClient.health = original_health
        self.pending_task_status = {}
        self.cancel_calls = []
        self.delete_calls = []
        self.chunk_calls = []
        self.add_doc_calls = []
        self.chunk_response = BaseResponse(code=200, msg='success', data={'items': [], 'total': 0})
        self._patch_parser_client()

    def close(self):
        self._tmp_dir.cleanup()

    def make_file(self, name: str, content: str):
        file_path = os.path.join(self.tmp_dir, name)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return file_path

    def finish_task(self, task_id: str, status: DocStatus = DocStatus.SUCCESS, callback_id: str = None):
        return self.manager.on_task_callback(TaskCallbackRequest(
            callback_id=callback_id or str(uuid4()),
            task_id=task_id,
            event_type=CallbackEventType.FINISH,
            status=status,
        ))

    def start_task(self, task_id: str, callback_id: str = None):
        return self.manager.on_task_callback(TaskCallbackRequest(
            callback_id=callback_id or str(uuid4()),
            task_id=task_id,
            event_type=CallbackEventType.START,
            status=DocStatus.WORKING,
        ))

    def _queue_task(self, task_id: str, final_status: DocStatus):
        self.pending_task_status[task_id] = final_status

    def _patch_parser_client(self):
        def add_doc(task_id, algo_id, kb_id, doc_id, file_path, metadata=None, reparse_group=None,
                    callback_url=None, transfer_params=None):
            self.add_doc_calls.append({
                'task_id': task_id,
                'algo_id': algo_id,
                'kb_id': kb_id,
                'doc_id': doc_id,
                'file_path': file_path,
                'metadata': metadata,
                'reparse_group': reparse_group,
                'callback_url': callback_url,
                'transfer_params': transfer_params,
            })
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'algo_id': algo_id, 'kb_id': kb_id})

        def update_meta(task_id, algo_id, kb_id, doc_id, metadata=None, file_path=None, callback_url=None):
            del doc_id, metadata, file_path, callback_url
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'algo_id': algo_id, 'kb_id': kb_id})

        def delete_doc(task_id, algo_id, kb_id, doc_id, callback_url=None):
            del callback_url
            self.delete_calls.append({'task_id': task_id, 'algo_id': algo_id, 'kb_id': kb_id, 'doc_id': doc_id})
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'algo_id': algo_id, 'kb_id': kb_id})

        def cancel_task(task_id):
            self.cancel_calls.append(task_id)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'cancel_status': True})

        def list_doc_chunks(algo_id, kb_id, doc_id, group, offset, page_size):
            self.chunk_calls.append({
                'algo_id': algo_id,
                'kb_id': kb_id,
                'doc_id': doc_id,
                'group': group,
                'offset': offset,
                'page_size': page_size,
            })
            return self.chunk_response

        self.manager._parser_client.add_doc = add_doc
        self.manager._parser_client.update_meta = update_meta
        self.manager._parser_client.delete_doc = delete_doc
        self.manager._parser_client.cancel_task = cancel_task
        self.manager._parser_client.list_doc_chunks = list_doc_chunks
        self.manager._parser_client.list_algorithms = lambda: BaseResponse(
            code=200,
            msg='success',
            data=[{'algo_id': '__default__', 'display_name': 'Default', 'description': 'desc'}],
        )
        self.manager._parser_client.get_algorithm_groups = lambda algo_id: BaseResponse(
            code=200,
            msg='success',
            data=[{'name': 'line', 'type': 'chunk', 'display_name': 'Line'}] if algo_id == '__default__' else None,
        )


@pytest.fixture
def manager_harness():
    harness = _ManagerHarness()
    try:
        yield harness
    finally:
        harness.close()


def test_manager_run_idempotent_atomic(manager_harness):
    started = []

    def handler():
        started.append(time.time())
        time.sleep(0.2)
        return {'task_id': str(uuid4())}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(manager_harness.manager.run_idempotent, '/local/atomic', 'same-key', {'k': 1}, handler)
        time.sleep(0.05)
        with pytest.raises(DocServiceError) as exc:
            manager_harness.manager.run_idempotent('/local/atomic', 'same-key', {'k': 1}, handler)
        result = future.result(timeout=2)

    assert exc.value.biz_code == 'E_IDEMPOTENCY_IN_PROGRESS'
    replay = manager_harness.manager.run_idempotent('/local/atomic', 'same-key', {'k': 1}, handler)
    assert len(started) == 1
    assert replay == result


def test_manager_upload_callback_and_doc_detail(manager_harness):
    manager_harness.manager.create_kb('kb_upload', algo_id='__default__')
    file_path = manager_harness.make_file('upload.txt', 'upload content')

    items = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_upload',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='upload-doc')],
    ))

    task_id = items[0]['task_id']
    assert items[0]['accepted'] is True
    assert items[0]['parse_status'] == DocStatus.WAITING.value

    manager_harness.finish_task(task_id)

    task = manager_harness.manager.get_task(task_id)
    detail = manager_harness.manager.get_doc_detail('upload-doc')

    assert task.code == 200
    assert task.data['status'] == DocStatus.SUCCESS.value
    assert detail['doc']['upload_status'] == DocStatus.SUCCESS.value
    assert detail['snapshot']['status'] == DocStatus.SUCCESS.value
    assert detail['latest_task']['task_id'] == task_id


def test_manager_cancel_waiting_add_updates_all_states(manager_harness):
    manager_harness.manager.create_kb('kb_cancel', algo_id='__default__')
    file_path = manager_harness.make_file('cancel.txt', 'cancel content')

    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_cancel',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='cancel-doc')],
    ))
    task_id = upload[0]['task_id']

    resp = manager_harness.manager.cancel_task(task_id)
    snapshot = manager_harness.manager._get_parse_snapshot('cancel-doc', 'kb_cancel', '__default__')
    doc = manager_harness.manager._get_doc('cancel-doc')
    task = manager_harness.manager.get_task(task_id)

    assert resp.code == 200
    assert resp.data['status'] == DocStatus.CANCELED.value
    assert task.data['status'] == DocStatus.CANCELED.value
    assert snapshot['status'] == DocStatus.CANCELED.value
    assert doc['upload_status'] == DocStatus.CANCELED.value


def test_manager_cancel_working_task_rejected(manager_harness):
    manager_harness.manager.create_kb('kb_working', algo_id='__default__')
    file_path = manager_harness.make_file('working.txt', 'working content')

    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_working',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='working-doc')],
    ))
    task_id = upload[0]['task_id']
    manager_harness.start_task(task_id)

    resp = manager_harness.manager.cancel_task(task_id)

    assert resp.code == 409
    assert resp.data['status'] == DocStatus.WORKING.value


def test_manager_delete_waiting_add_uses_cancel_path(manager_harness):
    manager_harness.manager.create_kb('kb_delete_waiting', algo_id='__default__')
    file_path = manager_harness.make_file('delete_waiting.txt', 'delete waiting content')

    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_delete_waiting',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='delete-waiting-doc')],
    ))
    original_task_id = upload[0]['task_id']

    items = manager_harness.manager.delete(DeleteRequest(
        kb_id='kb_delete_waiting',
        algo_id='__default__',
        doc_ids=['delete-waiting-doc'],
    ))
    snapshot = manager_harness.manager._get_parse_snapshot('delete-waiting-doc', 'kb_delete_waiting', '__default__')
    doc = manager_harness.manager._get_doc('delete-waiting-doc')

    assert items[0]['task_id'] == original_task_id
    assert items[0]['status'] == DocStatus.CANCELED.value
    assert manager_harness.cancel_calls == [original_task_id]
    assert manager_harness.delete_calls == []
    assert snapshot['status'] == DocStatus.CANCELED.value
    assert doc['upload_status'] == DocStatus.CANCELED.value


def test_manager_stale_callback_ignored_after_reparse(manager_harness):
    manager_harness.manager.create_kb('kb_stale', algo_id='__default__')
    file_path = manager_harness.make_file('stale.txt', 'stale content')

    uploaded = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_stale',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='stale-doc')],
    ))
    manager_harness.finish_task(uploaded[0]['task_id'])

    first_task_id = manager_harness.manager.reparse(ReparseRequest(
        kb_id='kb_stale',
        algo_id='__default__',
        doc_ids=['stale-doc'],
    ))[0]
    second_task_id = manager_harness.manager.reparse(ReparseRequest(
        kb_id='kb_stale',
        algo_id='__default__',
        doc_ids=['stale-doc'],
    ))[0]

    stale_resp = manager_harness.manager.on_task_callback(TaskCallbackRequest(
        callback_id='stale-callback',
        task_id=first_task_id,
        event_type=CallbackEventType.FINISH,
        status=DocStatus.SUCCESS,
    ))

    assert first_task_id != second_task_id
    assert stale_resp['ignored_reason'] == 'stale_task_callback'


def test_manager_transfer_uses_target_doc_id_for_target_records(manager_harness):
    manager_harness.manager.create_kb('kb_transfer_source', algo_id='__default__')
    manager_harness.manager.create_kb('kb_transfer_target', algo_id='__default__')
    file_path = manager_harness.make_file('transfer.txt', 'transfer content')
    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_transfer_source',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='source-doc')],
    ))
    manager_harness.finish_task(upload[0]['task_id'])

    items = manager_harness.manager.transfer(TransferRequest(items=[TransferItem(
        doc_id='source-doc',
        target_doc_id='target-doc-copy',
        source_kb_id='kb_transfer_source',
        source_algo_id='__default__',
        target_kb_id='kb_transfer_target',
        target_algo_id='__default__',
        mode='copy',
    )]))

    task_id = items[0]['task_id']
    manager_harness.finish_task(task_id)
    target_doc = manager_harness.manager._get_doc('target-doc-copy')

    assert items[0]['doc_id'] == 'source-doc'
    assert items[0]['target_doc_id'] == 'target-doc-copy'
    assert manager_harness.add_doc_calls[-1]['doc_id'] == 'source-doc'
    assert manager_harness.add_doc_calls[-1]['transfer_params']['target_doc_id'] == 'target-doc-copy'
    assert manager_harness.add_doc_calls[-1]['file_path'] == file_path
    assert manager_harness.manager._has_kb_document('kb_transfer_target', 'target-doc-copy') is True
    assert manager_harness.manager._has_kb_document('kb_transfer_target', 'source-doc') is False
    assert target_doc['upload_status'] == DocStatus.SUCCESS.value
    assert target_doc['filename'] == 'transfer.txt'
    assert target_doc['meta'] == '{}'
    assert target_doc['path'] == file_path
    assert manager_harness.manager._has_kb_document('kb_transfer_source', 'source-doc') is True


def test_manager_transfer_same_kb_copy_reuses_source_path(manager_harness):
    manager_harness.manager.create_kb('kb_same_transfer', algo_id='__default__')
    file_path = manager_harness.make_file('same-transfer.txt', 'same transfer content')
    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_same_transfer',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='same-source-doc')],
    ))
    manager_harness.finish_task(upload[0]['task_id'])

    items = manager_harness.manager.transfer(TransferRequest(items=[TransferItem(
        doc_id='same-source-doc',
        target_doc_id='same-target-doc',
        source_kb_id='kb_same_transfer',
        source_algo_id='__default__',
        target_kb_id='kb_same_transfer',
        target_algo_id='__default__',
        mode='copy',
    )]))

    assert items[0]['accepted'] is True
    assert items[0]['status'] == DocStatus.WAITING.value
    assert items[0]['target_file_path'] == file_path
    assert manager_harness.add_doc_calls[-1]['doc_id'] == 'same-source-doc'
    assert manager_harness.add_doc_calls[-1]['transfer_params']['target_doc_id'] == 'same-target-doc'
    assert manager_harness.manager._has_kb_document('kb_same_transfer', 'same-source-doc') is True
    assert manager_harness.manager._has_kb_document('kb_same_transfer', 'same-target-doc') is True


def test_manager_transfer_move_cleans_source_doc_with_target_doc_id(manager_harness):
    manager_harness.manager.create_kb('kb_move_source', algo_id='__default__')
    manager_harness.manager.create_kb('kb_move_target', algo_id='__default__')
    file_path = manager_harness.make_file('move.txt', 'move content')
    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_move_source',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='source-doc-move')],
    ))
    manager_harness.finish_task(upload[0]['task_id'])

    items = manager_harness.manager.transfer(TransferRequest(items=[TransferItem(
        doc_id='source-doc-move',
        target_doc_id='target-doc-move',
        source_kb_id='kb_move_source',
        source_algo_id='__default__',
        target_kb_id='kb_move_target',
        target_algo_id='__default__',
        mode='move',
    )]))

    task_id = items[0]['task_id']
    manager_harness.finish_task(task_id)

    assert manager_harness.manager._has_kb_document('kb_move_source', 'source-doc-move') is False
    assert manager_harness.manager._has_kb_document('kb_move_target', 'target-doc-move') is True
    assert manager_harness.manager._get_doc('source-doc-move')['upload_status'] == DocStatus.DELETED.value
    assert manager_harness.manager._get_doc('target-doc-move')['upload_status'] == DocStatus.SUCCESS.value
    assert manager_harness.manager._get_parse_snapshot('source-doc-move', 'kb_move_source', '__default__') is None


def test_manager_transfer_target_fields_override_source_defaults(manager_harness):
    manager_harness.manager.create_kb('kb_override_source', algo_id='__default__')
    manager_harness.manager.create_kb('kb_override_target', algo_id='__default__')
    file_path = manager_harness.make_file('override.txt', 'override content')
    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_override_source',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='source-doc-override', metadata={'a': 1, 'b': 'keep'})],
    ))
    manager_harness.finish_task(upload[0]['task_id'])

    items = manager_harness.manager.transfer(TransferRequest(items=[TransferItem(
        doc_id='source-doc-override',
        target_doc_id='target-doc-override',
        source_kb_id='kb_override_source',
        source_algo_id='__default__',
        target_kb_id='kb_override_target',
        target_algo_id='__default__',
        target_metadata={'a': 2, 'c': 'new'},
        target_filename='renamed.txt',
        mode='copy',
    )]))

    manager_harness.finish_task(items[0]['task_id'])
    target_doc = manager_harness.manager._get_doc('target-doc-override')

    assert target_doc['filename'] == 'renamed.txt'
    assert target_doc['meta'] == '{"a": 2, "b": "keep", "c": "new"}'
    assert os.path.basename(target_doc['path']) == 'renamed.txt'
    assert manager_harness.add_doc_calls[-1]['metadata'] == {'a': 2, 'b': 'keep', 'c': 'new'}
    assert manager_harness.add_doc_calls[-1]['file_path'] == file_path


def test_manager_transfer_target_file_path_overrides_target_file_info(manager_harness):
    manager_harness.manager.create_kb('kb_path_source', algo_id='__default__')
    manager_harness.manager.create_kb('kb_path_target', algo_id='__default__')
    file_path = manager_harness.make_file('path-source.txt', 'path source content')
    upload = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_path_source',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='source-doc-path')],
    ))
    manager_harness.finish_task(upload[0]['task_id'])

    items = manager_harness.manager.transfer(TransferRequest(items=[TransferItem(
        doc_id='source-doc-path',
        target_doc_id='target-doc-path',
        source_kb_id='kb_path_source',
        source_algo_id='__default__',
        target_kb_id='kb_path_target',
        target_algo_id='__default__',
        target_file_path='/virtual/target/renamed-from-path.txt',
        mode='copy',
    )]))

    manager_harness.finish_task(items[0]['task_id'])
    target_doc = manager_harness.manager._get_doc('target-doc-path')

    assert items[0]['target_file_path'] == '/virtual/target/renamed-from-path.txt'
    assert target_doc['filename'] == 'renamed-from-path.txt'
    assert target_doc['path'] == '/virtual/target/renamed-from-path.txt'
    assert manager_harness.add_doc_calls[-1]['file_path'] == file_path


def test_manager_list_chunks_forwards_true_pagination(manager_harness):
    manager_harness.manager.create_kb('kb_chunks', algo_id='__default__')
    file_path = manager_harness.make_file('chunks.txt', 'chunks content')
    uploaded = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_chunks',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='chunks-doc')],
    ))
    manager_harness.finish_task(uploaded[0]['task_id'])
    manager_harness.chunk_response = BaseResponse(code=200, msg='success', data={
        'items': [{'uid': 'chunk-2'}, {'uid': 'chunk-3'}],
        'total': 7,
    })

    data = manager_harness.manager.list_chunks(
        kb_id='kb_chunks',
        doc_id='chunks-doc',
        group='line',
        algo_id='__default__',
        page=2,
        page_size=2,
    )

    assert manager_harness.chunk_calls == [{
        'algo_id': '__default__',
        'kb_id': 'kb_chunks',
        'doc_id': 'chunks-doc',
        'group': 'line',
        'offset': 2,
        'page_size': 2,
    }]
    assert data['items'] == [{'uid': 'chunk-2'}, {'uid': 'chunk-3'}]
    assert data['total'] == 7
    assert data['page'] == 2
    assert data['offset'] == 2


def test_manager_callback_payload_fallback_and_delete_transition(manager_harness):
    manager_harness.manager.create_kb('kb_callback', algo_id='__default__')
    file_path = manager_harness.make_file('callback.txt', 'callback content')
    manager_harness.manager._upsert_doc(
        doc_id='callback-doc',
        filename='callback.txt',
        path=file_path,
        metadata={'case': 'callback'},
        source_type=SourceType.EXTERNAL,
    )
    manager_harness.manager._ensure_kb_document('kb_callback', 'callback-doc')
    queued_at = manager_harness.manager._upsert_parse_snapshot(
        doc_id='callback-doc',
        kb_id='kb_callback',
        algo_id='__default__',
        status=DocStatus.DELETING,
        task_type=TaskType.DOC_DELETE,
        current_task_id='delete-task',
        queued_at=datetime.now(),
    )['queued_at']

    start_resp = manager_harness.manager.on_task_callback(TaskCallbackRequest(
        callback_id='delete-start',
        task_id='delete-task',
        event_type=CallbackEventType.START,
        status=DocStatus.WORKING,
        payload={
            'task_type': TaskType.DOC_DELETE.value,
            'doc_id': 'callback-doc',
            'kb_id': 'kb_callback',
            'algo_id': '__default__',
        },
    ))
    start_snapshot = manager_harness.manager._get_parse_snapshot('callback-doc', 'kb_callback', '__default__')
    finish_resp = manager_harness.manager.on_task_callback(TaskCallbackRequest(
        callback_id='delete-finish',
        task_id='delete-task',
        event_type=CallbackEventType.FINISH,
        status=DocStatus.SUCCESS,
        payload={
            'task_type': TaskType.DOC_DELETE.value,
            'doc_id': 'callback-doc',
            'kb_id': 'kb_callback',
            'algo_id': '__default__',
        },
    ))
    snapshot = manager_harness.manager._get_parse_snapshot('callback-doc', 'kb_callback', '__default__')

    assert start_resp['ack'] is True
    assert finish_resp['ack'] is True
    assert start_snapshot['queued_at'] == queued_at
    assert snapshot['status'] == DocStatus.DELETED.value
    assert manager_harness.manager._has_kb_document('kb_callback', 'callback-doc') is False
    assert manager_harness.manager._get_doc('callback-doc')['upload_status'] == DocStatus.DELETED.value


def test_parser_client_algo_endpoint_fallback():
    client = ParserClient(parser_url='http://parser.test')
    calls = []

    def fake_request(method, path, params=None, **kwargs):
        assert method == 'GET'
        assert kwargs == {}
        del params
        calls.append(path)
        if path == '/v1/algo/list':
            raise RuntimeError('parser http error: 404 missing route')
        if path == '/algo/list':
            return {
                'code': 200,
                'msg': 'success',
                'data': [{'algo_id': '__default__', 'display_name': 'Default', 'description': 'desc'}],
            }
        if path == '/v1/algo/__default__/groups':
            raise RuntimeError('parser http error: 404 missing route')
        if path == '/algo/__default__/group/info':
            return {
                'code': 200,
                'msg': 'success',
                'data': [{'name': 'line', 'type': 'chunk', 'display_name': 'Line'}],
            }
        raise AssertionError(path)

    client._request = fake_request

    algo_resp = client.list_algorithms()
    group_resp = client.get_algorithm_groups('__default__')

    assert algo_resp.code == 200
    assert group_resp.code == 200
    assert calls == [
        '/v1/algo/list',
        '/algo/list',
        '/v1/algo/__default__/groups',
        '/algo/__default__/group/info',
    ]
