import os
import tempfile
import threading
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
        # Dispose SQLAlchemy engine before removing temp dir; on Windows the
        # open DB connections lock the .db / -wal / -shm files and prevent
        # TemporaryDirectory.cleanup() from succeeding.
        if hasattr(self.manager, '_db_manager') and hasattr(self.manager._db_manager, '_engine'):
            engine = self.manager._db_manager._engine
            if engine is not None:
                engine.dispose()
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


def test_manager_run_idempotent_completion_failure_releases_claim(manager_harness):
    original_complete = manager_harness.manager._complete_idempotency_record
    manager_harness.manager._complete_idempotency_record = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError('complete failed')
    )

    with pytest.raises(RuntimeError, match='complete failed'):
        manager_harness.manager.run_idempotent('/local/complete-fail', 'same-key', {'k': 1}, lambda: {'ok': True})

    manager_harness.manager._complete_idempotency_record = original_complete
    result = manager_harness.manager.run_idempotent('/local/complete-fail', 'same-key', {'k': 1}, lambda: {'ok': True})
    assert result == {'ok': True}


def test_manager_upsert_same_path_is_serialized(manager_harness):
    file_path = manager_harness.make_file('serialized.txt', 'serialized content')
    barrier = threading.Barrier(2)

    def worker(doc_id: str):
        barrier.wait()
        return manager_harness.manager._upsert_doc(
            doc_id=doc_id,
            filename='serialized.txt',
            path=file_path,
            metadata={},
            source_type=SourceType.API,
            upload_status=DocStatus.SUCCESS,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, doc_id) for doc_id in ('serialized-a', 'serialized-b')]

    results = []
    errors = []
    for future in futures:
        try:
            results.append(future.result(timeout=2))
        except Exception as exc:
            errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DocServiceError)
    assert errors[0].biz_code == 'E_STATE_CONFLICT'

    with manager_harness.manager._db_manager.get_session() as session:
        Doc = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_documents')
        assert session.query(Doc).filter(Doc.path == file_path).count() == 1


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
    manager_harness.manager.list_docs = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('get_doc_detail should not scan list_docs')
    )

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
    assert snapshot is None
    assert manager_harness.manager._has_kb_document('kb_delete_waiting', 'delete-waiting-doc') is False
    assert doc is None


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


def test_manager_purge_local_and_rebind_keep_doc_consistent(manager_harness):
    manager_harness.manager.create_kb('kb_purge_source', algo_id='__default__')
    manager_harness.manager.create_kb('kb_purge_target', algo_id='__default__')
    file_path = manager_harness.make_file('purge-rebind.txt', 'purge rebind content')
    uploaded = manager_harness.manager.upload(UploadRequest(
        kb_id='kb_purge_source',
        algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='shared-doc')],
    ))
    manager_harness.finish_task(uploaded[0]['task_id'])

    barrier = threading.Barrier(2)

    def purge():
        barrier.wait()
        manager_harness.manager._purge_deleted_kb_doc_data('kb_purge_source', 'shared-doc', remove_relation=True)

    def rebind():
        barrier.wait()
        return manager_harness.manager.upload(UploadRequest(
            kb_id='kb_purge_target',
            algo_id='__default__',
            items=[AddFileItem(file_path=file_path, doc_id='shared-doc')],
        ))

    with ThreadPoolExecutor(max_workers=2) as pool:
        purge_future = pool.submit(purge)
        rebind_future = pool.submit(rebind)
        purge_future.result(timeout=2)
        rebound = rebind_future.result(timeout=2)

    manager_harness.finish_task(rebound[0]['task_id'])

    assert manager_harness.manager._has_kb_document('kb_purge_source', 'shared-doc') is False
    assert manager_harness.manager._has_kb_document('kb_purge_target', 'shared-doc') is True
    doc = manager_harness.manager._get_doc('shared-doc')
    assert doc is not None
    assert doc['path'] == file_path


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


# ---------------------------------------------------------------------------
# Regression tests for scan logic (FAILED/CANCELED retry, stale cleanup,
# multi-KB scan iteration)
# ---------------------------------------------------------------------------

def test_list_kb_docs_excludes_failed_for_retry(manager_harness):
    '''FAILED/CANCELED docs must be excluded from the synced view so scan retries them.'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_retry', algo_id='__default__')
    file_ok = manager_harness.make_file('ok.txt', 'ok')
    file_fail = manager_harness.make_file('fail.txt', 'fail')

    items = mgr.upload(UploadRequest(
        kb_id='kb_retry', algo_id='__default__',
        items=[
            AddFileItem(file_path=file_ok, doc_id='doc-ok'),
            AddFileItem(file_path=file_fail, doc_id='doc-fail'),
        ],
    ))
    # Complete one as SUCCESS, the other as FAILED
    manager_harness.finish_task(items[0]['task_id'], status=DocStatus.SUCCESS)
    manager_harness.finish_task(items[1]['task_id'], status=DocStatus.FAILED)

    # exclude_failed=True (default): FAILED doc should NOT appear → scan will retry it
    synced = mgr._list_kb_docs_by_path('kb_retry', exclude_failed=True)
    assert file_ok in synced
    assert file_fail not in synced

    # exclude_failed=False: FAILED doc SHOULD appear → stale cleanup can see it
    all_known = mgr._list_kb_docs_by_path('kb_retry', exclude_failed=False)
    assert file_ok in all_known
    assert file_fail in all_known


def test_stale_cleanup_sees_failed_docs(manager_harness):
    '''When a FAILED doc source file is removed from disk, stale cleanup must still find it.'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_stale', algo_id='__default__')
    file_path = manager_harness.make_file('stale.txt', 'stale')

    items = mgr.upload(UploadRequest(
        kb_id='kb_stale', algo_id='__default__',
        items=[AddFileItem(file_path=file_path, doc_id='doc-stale')],
    ))
    manager_harness.finish_task(items[0]['task_id'], status=DocStatus.FAILED)

    # File removed from disk
    os.remove(file_path)
    disk_set = set()

    # exclude_failed=False should still contain the failed doc for stale cleanup
    all_known = mgr._list_kb_docs_by_path('kb_stale', exclude_failed=False)
    stale_ids = [did for path, did in all_known.items() if path not in disk_set]
    assert 'doc-stale' in stale_ids


def test_list_active_kb_algo_pairs(manager_harness):
    '''_list_active_kb_algo_pairs should return all active KB+algo bindings for multi-KB scan.'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_a', algo_id='__default__')
    mgr.create_kb('kb_b', algo_id='__default__')

    pairs = mgr._list_active_kb_algo_pairs()
    kb_ids = {kb_id for kb_id, _ in pairs}
    # Both KBs plus the auto-created __default__ should be present
    assert 'kb_a' in kb_ids
    assert 'kb_b' in kb_ids


def test_ensure_kb_registers_in_active_pairs(manager_harness):
    '''Lightweight ensure_kb + ensure_kb_algorithm must make the KB visible to scan.'''
    mgr = manager_harness.manager
    # Use the lightweight registration path (no algorithm validation)
    mgr._ensure_kb('custom_group', display_name='custom_group')
    mgr._ensure_kb_algorithm('custom_group', 'custom_group')

    pairs = mgr._list_active_kb_algo_pairs()
    pair_set = {(kb_id, algo_id) for kb_id, algo_id in pairs}
    assert ('custom_group', 'custom_group') in pair_set, (
        'KB registered via _ensure_kb + _ensure_kb_algorithm must appear in active pairs for scan'
    )


def test_scan_syncs_non_default_kb(manager_harness):
    '''End-to-end: after registering a non-default KB, scan should upload files to it.'''
    mgr = manager_harness.manager
    # Register a non-default KB
    mgr._ensure_kb('my_group', display_name='my_group')
    mgr._ensure_kb_algorithm('my_group', 'my_group')
    file_path = manager_harness.make_file('scan_target.txt', 'content')

    # Simulate what _sync_dataset_for_kb does: check new paths
    synced = mgr._list_kb_docs_by_path('my_group', exclude_failed=True)
    assert file_path not in synced, 'File should not be synced yet'

    # Upload via the scan path
    items = mgr.upload(UploadRequest(
        kb_id='my_group', algo_id='my_group',
        items=[AddFileItem(file_path=file_path)],
        source_type=SourceType.SCAN,
    ))
    manager_harness.finish_task(items[0]['task_id'], status=DocStatus.SUCCESS)

    # Now scan should see the file as synced
    synced = mgr._list_kb_docs_by_path('my_group', exclude_failed=True)
    assert file_path in synced, 'After scan upload, file must appear in synced docs'
