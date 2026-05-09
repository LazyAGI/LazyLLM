import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from lazyllm.tools.rag.doc_service.base import (
    AddFileItem,
    CallbackEventType,
    DeleteRequest,
    DocServiceError,
    DocStatus,
    KBStatus,
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
        # ng_id → algo_id mapping, populated by _patch_multi_algo
        self._ng_to_algo: dict = {}
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

    def set_ng_status(self, kb_id: str, doc_id: str, ng_ids, status: str = 'SUCCESS'):
        '''Simulate parser worker writing ng_status after successful parse.'''
        NgStatus = self.manager._db_manager.get_table_orm_class('lazyllm_doc_node_group_status')
        from datetime import datetime as _dt
        with self.manager._db_manager.get_session() as session:
            for ng_id in ng_ids:
                row = session.query(NgStatus).filter(
                    NgStatus.kb_id == kb_id, NgStatus.doc_id == doc_id,
                    NgStatus.node_group_id == ng_id,
                ).first()
                if row:
                    row.status = status
                    row.updated_at = _dt.now()

    def _queue_task(self, task_id: str, final_status: DocStatus):
        self.pending_task_status[task_id] = final_status

    def _patch_parser_client(self):
        def add_doc(task_id, kb_id, doc_id, file_path, metadata=None, ng_names=None,
                    extractor_names=None, task_type=None, callback_url=None, transfer_params=None):
            # Infer algo_id from ng_names using the ng→algo mapping populated by _patch_multi_algo.
            # Only exclusive ng_names (not shared across algos) are in the mapping.
            # Falls back to '__default__' when the mapping is empty (single-algo tests).
            algo_id = '__default__'
            if ng_names and self._ng_to_algo:
                for ng_name in ng_names:
                    if ng_name in self._ng_to_algo:
                        algo_id = self._ng_to_algo[ng_name]
                        break
            self.add_doc_calls.append({
                'task_id': task_id,
                'algo_id': algo_id,
                'kb_id': kb_id,
                'doc_id': doc_id,
                'file_path': file_path,
                'metadata': metadata,
                'ng_names': ng_names,
                'task_type': task_type,
                'callback_url': callback_url,
                'transfer_params': transfer_params,
            })
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'kb_id': kb_id})

        def update_meta(task_id, kb_id, doc_id, metadata=None, file_path=None, callback_url=None):
            del doc_id, metadata, file_path, callback_url
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'kb_id': kb_id})

        def delete_doc(task_id, kb_id, doc_id, callback_url=None, node_group_ids_to_delete=None):
            del callback_url
            self.delete_calls.append({
                'task_id': task_id, 'kb_id': kb_id, 'doc_id': doc_id,
                'node_group_ids_to_delete': node_group_ids_to_delete,
            })
            self._queue_task(task_id, DocStatus.SUCCESS)
            return BaseResponse(code=200, msg='success', data={'task_id': task_id, 'kb_id': kb_id})

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
            data=[{'id': 'line', 'name': 'line'}] if algo_id == '__default__' else None,
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
    original_finish = manager_harness.manager._finish_idempotency
    manager_harness.manager._finish_idempotency = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError('complete failed')
    ) if not kwargs.get('drop', False) else original_finish(*args, **kwargs)

    with pytest.raises(RuntimeError, match='complete failed'):
        manager_harness.manager.run_idempotent('/local/complete-fail', 'same-key', {'k': 1}, lambda: {'ok': True})

    manager_harness.manager._finish_idempotency = original_finish
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
    snapshot = manager_harness.manager._get_parse_snapshot('cancel-doc', 'kb_cancel')
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
        items=[AddFileItem(file_path=file_path, doc_id='delete-waiting-doc')],
    ))
    original_task_id = upload[0]['task_id']

    items = manager_harness.manager.delete(DeleteRequest(
        kb_id='kb_delete_waiting',
        doc_ids=['delete-waiting-doc'],
    ))
    snapshot = manager_harness.manager._get_parse_snapshot('delete-waiting-doc', 'kb_delete_waiting')
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
        algo_ids=['__default__'],
        doc_ids=['stale-doc'],
    ))[0]
    second_task_id = manager_harness.manager.reparse(ReparseRequest(
        kb_id='kb_stale',
        algo_ids=['__default__'],
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
    assert manager_harness.manager._get_parse_snapshot('source-doc-move', 'kb_move_source') is None


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
        },
    ))
    start_snapshot = manager_harness.manager._get_parse_snapshot('callback-doc', 'kb_callback')
    finish_resp = manager_harness.manager.on_task_callback(TaskCallbackRequest(
        callback_id='delete-finish',
        task_id='delete-task',
        event_type=CallbackEventType.FINISH,
        status=DocStatus.SUCCESS,
        payload={
            'task_type': TaskType.DOC_DELETE.value,
            'doc_id': 'callback-doc',
            'kb_id': 'kb_callback',
        },
    ))
    snapshot = manager_harness.manager._get_parse_snapshot('callback-doc', 'kb_callback')

    assert start_resp['ack'] is True
    assert finish_resp['ack'] is True
    assert start_snapshot['queued_at'] == queued_at
    assert snapshot['status'] == DocStatus.DELETED.value
    assert manager_harness.manager._has_kb_document('kb_callback', 'callback-doc') is False
    assert manager_harness.manager._get_doc('callback-doc')['upload_status'] == DocStatus.DELETED.value


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


def test_ensure_kb_handles_flush_conflict(manager_harness, monkeypatch):
    mgr = manager_harness.manager
    Kb = mgr._db_manager.get_table_orm_class('lazyllm_knowledge_bases')
    original_get_session = mgr._db_manager.get_session
    state = {'done': False}

    @contextmanager
    def conflict_session():
        with original_get_session() as session:
            original_flush = session.flush

            def flush(*args, **kwargs):
                if (
                    not state['done']
                    and any(isinstance(obj, Kb) and obj.kb_id == 'conflict_group' for obj in session.new)
                ):
                    state['done'] = True
                    with original_get_session() as other:
                        other.add(Kb(
                            kb_id='conflict_group',
                            display_name='conflict_group',
                            description=None,
                            doc_count=0,
                            status=KBStatus.ACTIVE.value,
                            owner_id=None,
                            meta=None,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                        ))
                    raise IntegrityError('INSERT', None, Exception('duplicate kb'))
                return original_flush(*args, **kwargs)

            session.flush = flush
            yield session

    monkeypatch.setattr(mgr._db_manager, 'get_session', conflict_session)

    mgr._ensure_kb('conflict_group', display_name='conflict_group')

    kb = mgr._get_kb('conflict_group')
    assert kb is not None
    assert kb['kb_id'] == 'conflict_group'


def test_ensure_kb_algorithm_handles_flush_conflict(manager_harness, monkeypatch):
    mgr = manager_harness.manager
    Rel = mgr._db_manager.get_table_orm_class('lazyllm_kb_algorithm')
    original_get_session = mgr._db_manager.get_session
    state = {'done': False}

    mgr._ensure_kb('algo_conflict_group', display_name='algo_conflict_group')

    @contextmanager
    def conflict_session():
        with original_get_session() as session:
            original_flush = session.flush

            def flush(*args, **kwargs):
                if (
                    not state['done']
                    and any(isinstance(obj, Rel) and obj.kb_id == 'algo_conflict_group' for obj in session.new)
                ):
                    state['done'] = True
                    with original_get_session() as other:
                        other.add(Rel(
                            kb_id='algo_conflict_group',
                            algo_id='__default__',
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                        ))
                    raise IntegrityError('INSERT', None, Exception('duplicate kb algo'))
                return original_flush(*args, **kwargs)

            session.flush = flush
            yield session

    monkeypatch.setattr(mgr._db_manager, 'get_session', conflict_session)

    mgr._ensure_kb_algorithm('algo_conflict_group', '__default__')

    binding = mgr._get_kb_algorithms('algo_conflict_group')
    assert binding is not None
    assert '__default__' in binding


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


# ======================================================================
# list_docs large-dataset + pagination regression suite
#
# These tests are a baseline for the list_docs SQL rewrite: they encode
# the observable contract so the old Python-loop implementation and the
# new SQL-side JOIN + window-function implementation must both satisfy
# them. They deliberately avoid touching file system / upload pipeline
# to keep the large-dataset setups fast — rows are inserted directly
# via ORM.
# ======================================================================


def _bulk_insert_docs(manager, *, kb_id, doc_ids, snapshots=None, upload_status=DocStatus.SUCCESS,
                      filename_prefix='bulk', base_time=None):
    '''Directly insert Doc / Rel / optional State rows via ORM.

    ``snapshots`` is a list of ``(status_value, updated_offset_seconds)`` tuples
    applied to every doc. Only the last entry is used (parse_state is keyed by
    (doc_id, kb_id) — one row per doc per kb).
    '''
    base_time = base_time or datetime(2025, 1, 1, 12, 0, 0)
    with manager._db_manager.get_session() as session:
        Doc = manager._db_manager.get_table_orm_class('lazyllm_documents')
        Rel = manager._db_manager.get_table_orm_class('lazyllm_kb_documents')
        State = manager._db_manager.get_table_orm_class('lazyllm_doc_parse_state')
        for i, doc_id in enumerate(doc_ids):
            doc_ts = base_time + timedelta(seconds=i)
            session.add(Doc(
                doc_id=doc_id, filename=f'{filename_prefix}_{i}.txt',
                path=f'/tmp/{filename_prefix}/{doc_id}.txt',
                meta='{}', upload_status=upload_status.value if isinstance(upload_status, DocStatus)
                else upload_status,
                source_type=SourceType.API.value, file_type='txt', size_bytes=10,
                created_at=doc_ts, updated_at=doc_ts,
            ))
            session.add(Rel(kb_id=kb_id, doc_id=doc_id, created_at=doc_ts, updated_at=doc_ts))
            if snapshots:
                # Use the last snapshot entry as the single parse_state row.
                snap = snapshots[-1]
                status_val, offset_sec = snap
                snap_ts = doc_ts + timedelta(seconds=offset_sec)
                session.add(State(
                    doc_id=doc_id, kb_id=kb_id,
                    status=status_val, priority=0, retry_count=0, max_retry=3,
                    created_at=snap_ts, updated_at=snap_ts,
                ))


def _collect_all_pages(manager, *, page_size, max_pages=200, **list_kwargs):
    '''Iterate every page; return (flat_items, unique_totals_seen).'''
    all_items = []
    totals_seen = set()
    for page in range(1, max_pages + 1):
        resp = manager.list_docs(page=page, page_size=page_size, **list_kwargs)
        totals_seen.add(resp['total'])
        all_items.extend(resp['items'])
        if len(resp['items']) < page_size:
            break
    else:
        raise AssertionError(f'exceeded max_pages={max_pages}, something is wrong')
    return all_items, totals_seen


def test_list_docs_large_dataset_total_matches_pagination_sum(manager_harness):
    '''Across every page, total must be stable and sum-of-items must equal total.'''
    kb_id = 'kb_bulk_sum'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    doc_ids = [f'bulk-{i:04d}' for i in range(250)]
    _bulk_insert_docs(manager_harness.manager, kb_id=kb_id, doc_ids=doc_ids,
                      snapshots=[(DocStatus.SUCCESS.value, 0)])

    items, totals = _collect_all_pages(manager_harness.manager, page_size=30, kb_id=kb_id)
    assert totals == {250}, f'inconsistent totals across pages: {totals}'
    assert len(items) == 250
    assert len({it['doc']['doc_id'] for it in items}) == 250, 'duplicate doc_ids across pages'


def test_list_docs_pagination_boundary_last_partial_and_beyond(manager_harness):
    '''Last page can be partial; pages beyond the last return empty items but same total.'''
    kb_id = 'kb_bnd'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(manager_harness.manager, kb_id=kb_id,
                      doc_ids=[f'bnd-{i:04d}' for i in range(105)])

    last_page = manager_harness.manager.list_docs(kb_id=kb_id, page=6, page_size=20)
    assert last_page['total'] == 105
    assert len(last_page['items']) == 5

    beyond = manager_harness.manager.list_docs(kb_id=kb_id, page=10, page_size=20)
    assert beyond['total'] == 105
    assert beyond['items'] == []


def test_list_docs_status_filter_is_consistent_across_pages(manager_harness):
    '''total must reflect post-status-filter count and be identical on every page.

    This is the explicit regression for the review comment:
        "status filter in Python => total counts and DB pagination inconsistent"
    '''
    kb_id = 'kb_status_filter'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    status_cycle = [
        DocStatus.SUCCESS.value,
        DocStatus.FAILED.value,
        DocStatus.WORKING.value,
        DocStatus.DELETED.value,
    ]
    with manager_harness.manager._db_manager.get_session() as session:
        Doc = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_documents')
        Rel = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_kb_documents')
        State = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_doc_parse_state')
        for i in range(200):
            doc_id = f'stat-{i:04d}'
            ts = base_time + timedelta(seconds=i)
            session.add(Doc(
                doc_id=doc_id, filename=f'stat_{i}.txt', path=f'/tmp/stat/{doc_id}.txt',
                meta='{}', upload_status=DocStatus.SUCCESS.value,
                source_type=SourceType.API.value, file_type='txt', size_bytes=10,
                created_at=ts, updated_at=ts,
            ))
            session.add(Rel(kb_id=kb_id, doc_id=doc_id, created_at=ts, updated_at=ts))
            session.add(State(
                doc_id=doc_id, kb_id=kb_id,
                status=status_cycle[i % 4], priority=0, retry_count=0, max_retry=3,
                created_at=ts, updated_at=ts,
            ))

    items, totals = _collect_all_pages(manager_harness.manager, page_size=15, kb_id=kb_id,
                                       status=[DocStatus.SUCCESS.value])
    assert totals == {50}, f'total drifted across pages: {totals}'
    assert len(items) == 50
    assert all(it['snapshot']['status'] == DocStatus.SUCCESS.value for it in items)


def test_list_docs_status_filter_excludes_rows_without_snapshot(manager_harness):
    '''If a row has no parse_state at all, any status filter should exclude it.

    Also pins down the LEFT JOIN contract: with no status filter, rows without
    a snapshot must still appear, carrying ``snapshot = None``.
    '''
    kb_id = 'kb_no_snap'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'with-snap-{i:03d}' for i in range(50)],
        snapshots=[(DocStatus.SUCCESS.value, 0)],
        base_time=datetime(2025, 1, 1, 10, 0, 0),
        filename_prefix='withsnap',
    )
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'no-snap-{i:03d}' for i in range(50)],
        snapshots=None,
        base_time=datetime(2025, 1, 1, 14, 0, 0),
        filename_prefix='nosnap',
    )

    # No status filter => all 100 docs show up (LEFT JOIN).
    resp_all = manager_harness.manager.list_docs(kb_id=kb_id, page=1, page_size=500)
    assert resp_all['total'] == 100
    null_snap = [it for it in resp_all['items'] if it['snapshot'] is None]
    with_snap = [it for it in resp_all['items'] if it['snapshot'] is not None]
    assert len(null_snap) == 50
    assert len(with_snap) == 50
    assert all(it['doc']['doc_id'].startswith('no-snap-') for it in null_snap)
    assert all(it['doc']['doc_id'].startswith('with-snap-') for it in with_snap)

    # Status filter => only rows with a matching snapshot (50).
    resp_filtered = manager_harness.manager.list_docs(kb_id=kb_id, status=[DocStatus.SUCCESS.value],
                                                      page=1, page_size=500)
    assert resp_filtered['total'] == 50
    assert all(it['doc']['doc_id'].startswith('with-snap-') for it in resp_filtered['items'])


def test_list_docs_latest_snapshot_when_algo_id_not_specified(manager_harness):
    '''Snapshot is the single parse_state row per (doc_id, kb_id); status is returned correctly.'''
    kb_id = 'kb_latest'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'snap-{i:03d}' for i in range(10)],
        snapshots=[(DocStatus.SUCCESS.value, 0)],
    )

    resp_default = manager_harness.manager.list_docs(kb_id=kb_id, page=1, page_size=50)
    assert resp_default['total'] == 10
    for item in resp_default['items']:
        assert item['snapshot']['status'] == DocStatus.SUCCESS.value


def test_list_docs_kb_id_filter_isolates_results(manager_harness):
    for kb_id in ('kb-a', 'kb-b', 'kb-c'):
        manager_harness.manager.create_kb(kb_id, algo_id='__default__')
        _bulk_insert_docs(manager_harness.manager, kb_id=kb_id,
                          doc_ids=[f'{kb_id}-{i:03d}' for i in range(40)], filename_prefix=kb_id)

    resp = manager_harness.manager.list_docs(kb_id='kb-b', page=1, page_size=500)
    assert resp['total'] == 40
    assert all(it['relation']['kb_id'] == 'kb-b' for it in resp['items'])
    assert all(it['doc']['doc_id'].startswith('kb-b-') for it in resp['items'])


def test_list_docs_keyword_filter_matches_filename(manager_harness):
    kb_id = 'kb_keyword'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'report-{i:03d}' for i in range(50)],
        filename_prefix='report',
        base_time=datetime(2025, 1, 1, 10, 0, 0),
    )
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'invoice-{i:03d}' for i in range(50)],
        filename_prefix='invoice',
        base_time=datetime(2025, 1, 1, 12, 0, 0),
    )

    resp = manager_harness.manager.list_docs(kb_id=kb_id, keyword='invoice', page=1, page_size=500)
    assert resp['total'] == 50
    assert all('invoice' in it['doc']['filename'] for it in resp['items'])


def test_list_docs_include_deleted_or_canceled_flag(manager_harness):
    kb_id = 'kb_upload_status'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    base_time = datetime(2025, 1, 1, 10, 0, 0)
    upload_statuses = [
        DocStatus.SUCCESS.value,
        DocStatus.DELETED.value,
        DocStatus.CANCELED.value,
        DocStatus.WORKING.value,
    ]
    with manager_harness.manager._db_manager.get_session() as session:
        Doc = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_documents')
        Rel = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_kb_documents')
        for i in range(200):
            doc_id = f'ups-{i:04d}'
            ts = base_time + timedelta(seconds=i)
            session.add(Doc(
                doc_id=doc_id, filename=f'ups_{i}.txt', path=f'/tmp/ups/{doc_id}.txt',
                meta='{}', upload_status=upload_statuses[i % 4],
                source_type=SourceType.API.value, file_type='txt', size_bytes=10,
                created_at=ts, updated_at=ts,
            ))
            session.add(Rel(kb_id=kb_id, doc_id=doc_id, created_at=ts, updated_at=ts))

    resp_all = manager_harness.manager.list_docs(kb_id=kb_id, include_deleted_or_canceled=True,
                                                 page=1, page_size=500)
    assert resp_all['total'] == 200

    resp_filtered = manager_harness.manager.list_docs(kb_id=kb_id, include_deleted_or_canceled=False,
                                                      page=1, page_size=500)
    assert resp_filtered['total'] == 100
    excluded = {DocStatus.DELETED.value, DocStatus.CANCELED.value}
    assert all(it['doc']['upload_status'] not in excluded for it in resp_filtered['items'])


def test_list_docs_ordering_by_relation_updated_at_desc(manager_harness):
    '''Order contract: Rel.updated_at DESC primary, Doc.updated_at DESC as tiebreak.

    Asserts on the observable timestamps rather than doc_id lex-order so a
    rewrite that swaps storage/partitioning but preserves the ordering contract
    still passes.
    '''
    kb_id = 'kb_order'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(manager_harness.manager, kb_id=kb_id,
                      doc_ids=[f'order-{i:04d}' for i in range(100)],
                      base_time=datetime(2025, 1, 1, 10, 0, 0))
    resp = manager_harness.manager.list_docs(kb_id=kb_id, page=1, page_size=100)
    rel_times = [it['relation']['updated_at'] for it in resp['items']]
    assert rel_times == sorted(rel_times, reverse=True), \
        'Rel.updated_at must be non-increasing across returned rows'
    # Secondary tie-break: within equal Rel.updated_at, Doc.updated_at DESC.
    # The helper writes Rel.updated_at == Doc.updated_at for every row, so
    # Doc.updated_at must also be non-increasing on the returned projection.
    doc_times = [it['doc']['updated_at'] for it in resp['items']]
    assert doc_times == sorted(doc_times, reverse=True), \
        'Doc.updated_at must be non-increasing (secondary tie-break)'


def test_list_docs_latest_snapshot_tie_break_uses_created_at(manager_harness):
    '''parse_state is keyed by (doc_id, kb_id) — one row per doc. Snapshot status is returned correctly.'''
    kb_id = 'kb_tie'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    doc_ts = datetime(2025, 1, 1, 12, 0, 0)
    with manager_harness.manager._db_manager.get_session() as session:
        Doc = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_documents')
        Rel = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_kb_documents')
        State = manager_harness.manager._db_manager.get_table_orm_class('lazyllm_doc_parse_state')
        session.add(Doc(
            doc_id='tie-doc', filename='tie.txt', path='/tmp/tie.txt', meta='{}',
            upload_status=DocStatus.SUCCESS.value, source_type=SourceType.API.value,
            file_type='txt', size_bytes=10, created_at=doc_ts, updated_at=doc_ts,
        ))
        session.add(Rel(kb_id=kb_id, doc_id='tie-doc', created_at=doc_ts, updated_at=doc_ts))
        session.add(State(
            doc_id='tie-doc', kb_id=kb_id,
            status=DocStatus.SUCCESS.value, priority=0, retry_count=0, max_retry=3,
            created_at=doc_ts, updated_at=doc_ts,
        ))

    resp = manager_harness.manager.list_docs(kb_id=kb_id, page=1, page_size=10)
    assert resp['total'] == 1
    snap = resp['items'][0]['snapshot']
    assert snap is not None
    assert snap['status'] == DocStatus.SUCCESS.value


def test_list_docs_status_filter_uses_latest_snapshot_not_any(manager_harness):
    '''status filter matches the single parse_state row per (doc_id, kb_id).

    Docs with FAILED snapshot are excluded from status=SUCCESS query,
    and included in status=FAILED query.
    '''
    kb_id = 'kb_filter_latest'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'latest-{i:03d}' for i in range(20)],
        snapshots=[(DocStatus.FAILED.value, 0)],
    )

    resp_success = manager_harness.manager.list_docs(kb_id=kb_id, status=[DocStatus.SUCCESS.value],
                                                     page=1, page_size=100)
    assert resp_success['total'] == 0
    assert resp_success['items'] == []

    resp_failed = manager_harness.manager.list_docs(kb_id=kb_id, status=[DocStatus.FAILED.value],
                                                    page=1, page_size=100)
    assert resp_failed['total'] == 20
    for item in resp_failed['items']:
        assert item['snapshot']['status'] == DocStatus.FAILED.value


def test_list_docs_status_and_algo_id_combined(manager_harness):
    '''status filter matches the single parse_state row per (doc_id, kb_id).
    FAILED docs are returned by status=FAILED, excluded by status=SUCCESS.
    '''
    kb_id = 'kb_combo'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(
        manager_harness.manager, kb_id=kb_id,
        doc_ids=[f'combo-{i:03d}' for i in range(30)],
        snapshots=[(DocStatus.FAILED.value, 0)],
    )

    resp = manager_harness.manager.list_docs(kb_id=kb_id,
                                             status=[DocStatus.FAILED.value], page=1, page_size=100)
    assert resp['total'] == 30

    resp = manager_harness.manager.list_docs(kb_id=kb_id,
                                             status=[DocStatus.SUCCESS.value], page=1, page_size=100)
    assert resp['total'] == 0
    assert resp['items'] == []


def test_list_docs_empty_result_returns_well_formed_page(manager_harness):
    '''Query with no matches must return items=[], total=0, with page/page_size echoed.'''
    kb_id = 'kb_empty'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(manager_harness.manager, kb_id=kb_id,
                      doc_ids=[f'empty-{i:03d}' for i in range(5)],
                      snapshots=[(DocStatus.SUCCESS.value, 0)])

    # Nonexistent kb_id
    resp_nokb = manager_harness.manager.list_docs(kb_id='kb-does-not-exist', page=1, page_size=20)
    assert resp_nokb == {'items': [], 'total': 0, 'page': 1, 'page_size': 20}

    # Status that no snapshot has
    resp_nostatus = manager_harness.manager.list_docs(kb_id=kb_id, status=[DocStatus.CANCELED.value],
                                                      page=1, page_size=20)
    assert resp_nostatus == {'items': [], 'total': 0, 'page': 1, 'page_size': 20}


def test_list_docs_page_and_size_are_clamped(manager_harness):
    '''page<1 → clamped to 1; page_size<1 → clamped to 1. Behavior contract.'''
    kb_id = 'kb_clamp'
    manager_harness.manager.create_kb(kb_id, algo_id='__default__')
    _bulk_insert_docs(manager_harness.manager, kb_id=kb_id,
                      doc_ids=[f'clamp-{i:03d}' for i in range(5)])
    resp = manager_harness.manager.list_docs(kb_id=kb_id, page=0, page_size=0)
    assert resp['page'] == 1
    assert resp['page_size'] == 1
    assert resp['total'] == 5
    assert len(resp['items']) == 1

    resp_neg = manager_harness.manager.list_docs(kb_id=kb_id, page=-5, page_size=-1)
    assert resp_neg['page'] == 1
    assert resp_neg['page_size'] == 1


# ======================================================================
# Multi-algo node-group reuse tests
#
# Background:
#   algo1 owns ng_ids [ng1, ng2, ng3]
#   algo2 owns ng_ids [ng1, ng4, ng5]
#   algo3 owns ng_ids [ng1, ng6, ng7]
#   ng1 is shared across all three algos within the same kb.
#
# These tests encode the contract that shared node groups are never
# double-reset, double-deleted, or left as stale records when an algo
# is unbound.
# ======================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALGO_NG_MAP = {
    'algo1': ['ng1', 'ng2', 'ng3'],
    'algo2': ['ng1', 'ng4', 'ng5'],
    'algo3': ['ng1', 'ng6', 'ng7'],
}


def _patch_multi_algo(harness):
    '''Replace get_algorithm_groups and list_algorithms with versions that know all three algos.'''
    def get_algorithm_groups(algo_id):
        ng_ids = _ALGO_NG_MAP.get(algo_id)
        if ng_ids is None:
            return BaseResponse(code=404, msg='algo not found', data=None)
        # Return list-of-dict format: [{id, name}, ...] matching real parser response
        data = [{'id': ng_id, 'name': ng_id} for ng_id in ng_ids]
        return BaseResponse(code=200, msg='success', data=data)

    all_algo_ids = list(_ALGO_NG_MAP.keys())
    harness.manager._parser_client.list_algorithms = lambda: BaseResponse(
        code=200, msg='success',
        data=[{'algo_id': a, 'display_name': a, 'description': ''} for a in all_algo_ids],
    )
    harness.manager._parser_client.get_algorithm_groups = get_algorithm_groups

    # Populate ng_name → algo_id mapping for add_doc inference (shared ngs are excluded)
    ng_to_algo = {}
    ng_count = {}
    for _algo_id, ng_ids in _ALGO_NG_MAP.items():
        for ng_id in ng_ids:
            ng_count[ng_id] = ng_count.get(ng_id, 0) + 1
    for algo_id, ng_ids in _ALGO_NG_MAP.items():
        for ng_id in ng_ids:
            if ng_count[ng_id] == 1:  # exclusive ng (name == id in test fixtures)
                ng_to_algo[ng_id] = algo_id
    harness._ng_to_algo = ng_to_algo


def _ng_statuses(harness, kb_id, doc_id):
    '''Return {node_group_id: status} for a given (kb_id, doc_id).'''
    NgStatus = harness.manager._db_manager.get_table_orm_class('lazyllm_doc_node_group_status')
    with harness.manager._db_manager.get_session() as session:
        rows = session.query(NgStatus).filter(
            NgStatus.kb_id == kb_id, NgStatus.doc_id == doc_id,
        ).all()
        return {r.node_group_id: r.status for r in rows}


def _upload_doc(harness, kb_id, doc_id, filename='doc.txt'):
    fp = harness.make_file(filename, f'content of {doc_id}')
    items = harness.manager.upload(UploadRequest(
        kb_id=kb_id,
        items=[AddFileItem(file_path=fp, doc_id=doc_id)],
    ))
    return fp, items


# ---------------------------------------------------------------------------
# Scenario 1: single kb + single algo — baseline
# ---------------------------------------------------------------------------

def test_multi_algo_scenario1_single_kb_single_algo(manager_harness):
    '''Scenario 1: create kb1 with algo1, add docs → ng1/ng2/ng3 all get PENDING then SUCCESS.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1', algo_id='algo1')

    fp, items = _upload_doc(manager_harness, 'kb1', 'doc1', 'doc1.txt')
    assert items[0]['accepted'] is True
    task_id = items[0]['task_id']

    # Before callback: ng1/ng2/ng3 should be PENDING
    statuses = _ng_statuses(manager_harness, 'kb1', 'doc1')
    assert set(statuses.keys()) == {'ng1', 'ng2', 'ng3'}
    assert all(v == 'PENDING' for v in statuses.values())

    # Simulate parser writing SUCCESS for all ngs, then finish the task
    manager_harness.set_ng_status('kb1', 'doc1', ['ng1', 'ng2', 'ng3'], 'SUCCESS')
    manager_harness.finish_task(task_id)

    statuses = _ng_statuses(manager_harness, 'kb1', 'doc1')
    assert all(v == 'SUCCESS' for v in statuses.values())


# ---------------------------------------------------------------------------
# Scenario 2: single kb + two algos — shared ng1 has only one ng_status row
# ---------------------------------------------------------------------------

def test_multi_algo_scenario2_two_algos_shared_ng_single_row(manager_harness):
    '''Scenario 2: kb2 with algo1+algo3, add doc → single task covers all algos' ng_names.
    ng1 (shared) must have exactly one ng_status row.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb2', algo_id='algo1')
    mgr.update_kb('kb2', algo_id='algo3')

    fp, items = _upload_doc(manager_harness, 'kb2', 'doc2', 'doc2.txt')
    # Single task covers all algos
    assert len(items) == 1
    task_id = items[0]['task_id']

    # Exactly one add_doc call with all ng_names merged
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc2']
    assert len(add_calls) == 1
    sent_ngs = set(add_calls[0]['ng_names'] or [])
    assert sent_ngs == {'ng1', 'ng2', 'ng3', 'ng6', 'ng7'}

    manager_harness.finish_task(task_id)

    statuses = _ng_statuses(manager_harness, 'kb2', 'doc2')
    # ng1 shared → exactly one row; ng2/ng3 from algo1; ng6/ng7 from algo3
    assert set(statuses.keys()) == {'ng1', 'ng2', 'ng3', 'ng6', 'ng7'}
    # ng1 must appear only once (shared, not duplicated)
    NgStatus = mgr._db_manager.get_table_orm_class('lazyllm_doc_node_group_status')
    with mgr._db_manager.get_session() as session:
        count = session.query(NgStatus).filter(
            NgStatus.kb_id == 'kb2', NgStatus.doc_id == 'doc2',
            NgStatus.node_group_id == 'ng1',
        ).count()
    assert count == 1, 'shared ng1 must have exactly one ng_status row'


# ---------------------------------------------------------------------------
# Scenario 3a: add algo2 to kb1 (which already has algo1 with parsed docs)
#              reparse for algo2 must NOT reset ng1 (shared with algo1)
# ---------------------------------------------------------------------------

def test_multi_algo_scenario3a_reparse_skips_shared_ng(manager_harness):
    '''Scenario 3a: kb1 has algo1 (ng1/ng2/ng3 SUCCESS). Add algo2 and reparse all.
    Reparse must only touch ng4/ng5 (algo2-exclusive); ng1 must stay SUCCESS.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1s3', algo_id='algo1')

    fp, items = _upload_doc(manager_harness, 'kb1s3', 'doc1s3', 'doc1s3.txt')
    manager_harness.finish_task(items[0]['task_id'])

    # Simulate parser writing SUCCESS for ng1/ng2/ng3
    manager_harness.set_ng_status('kb1s3', 'doc1s3', ['ng1', 'ng2', 'ng3'], 'SUCCESS')

    # Verify ng1/ng2/ng3 are SUCCESS
    statuses = _ng_statuses(manager_harness, 'kb1s3', 'doc1s3')
    assert statuses == {'ng1': 'SUCCESS', 'ng2': 'SUCCESS', 'ng3': 'SUCCESS'}

    # Add algo2 to kb1s3
    mgr.update_kb('kb1s3', algo_id='algo2')

    # Reparse all for algo2 (ng1 is shared with algo1 → must be skipped)
    task_ids = mgr.reparse(ReparseRequest(
        kb_id='kb1s3', algo_ids=['algo2'], doc_ids=['doc1s3'],
    ))
    assert len(task_ids) == 1

    # ng1 must still be SUCCESS (not reset to PENDING)
    statuses_after = _ng_statuses(manager_harness, 'kb1s3', 'doc1s3')
    assert statuses_after['ng1'] == 'SUCCESS', 'shared ng1 must not be reset by algo2 reparse'
    # ng4/ng5 must be PENDING (algo2-exclusive, being reparsed)
    assert statuses_after['ng4'] == 'PENDING'
    assert statuses_after['ng5'] == 'PENDING'

    # The reparse task sent to parser must NOT include ng1.
    # New code sends ng_names=None (meaning "all non-shared ngs") when reparsing all ngs of an algo,
    # so we verify via ng_status: ng4/ng5 are PENDING (reparsed) and ng1 is still SUCCESS.
    reparse_calls = [c for c in manager_harness.add_doc_calls
                     if c['task_type'] == TaskType.DOC_REPARSE.value]
    assert len(reparse_calls) == 1


# ---------------------------------------------------------------------------
# Scenario 3b: explicit single-ng reparse is always allowed even if shared
# ---------------------------------------------------------------------------

def test_multi_algo_scenario3b_explicit_single_ng_reparse_allowed(manager_harness):
    '''Scenario 3b: user explicitly requests reparse of ng1 (shared). Must be allowed.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1s3b', algo_id='algo1')
    mgr.update_kb('kb1s3b', algo_id='algo2')

    fp, items = _upload_doc(manager_harness, 'kb1s3b', 'doc1s3b', 'doc1s3b.txt')
    # New code sends a single add_doc covering all algos; finish that one task.
    manager_harness.finish_task(items[0]['task_id'])
    # Simulate parser writing SUCCESS for all ngs
    manager_harness.set_ng_status('kb1s3b', 'doc1s3b', ['ng1', 'ng2', 'ng3', 'ng4', 'ng5'], 'SUCCESS')

    # Explicit reparse of ng1 via algo1 — must succeed and reset ng1 to PENDING
    task_ids = mgr.reparse(ReparseRequest(
        kb_id='kb1s3b', doc_ids=['doc1s3b'], ng_names=['ng1'],
    ))
    assert len(task_ids) == 1
    statuses = _ng_statuses(manager_harness, 'kb1s3b', 'doc1s3b')
    assert statuses['ng1'] == 'PENDING', 'explicit single-ng reparse must reset ng1 to PENDING'


# ---------------------------------------------------------------------------
# Scenario 3c: new doc added after algo2 is bound → all ngs (ng1..ng5) parsed
# ---------------------------------------------------------------------------

def test_multi_algo_scenario3c_new_doc_after_second_algo_bound(manager_harness):
    '''Scenario 3c: kb1 has algo1+algo2. New doc → tasks for both algos, ng1 has one row.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1s3c', algo_id='algo1')
    mgr.update_kb('kb1s3c', algo_id='algo2')

    fp, items = _upload_doc(manager_harness, 'kb1s3c', 'doc_new', 'doc_new.txt')

    # Single task covers all algos' ng_names merged
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc_new']
    assert len(add_calls) == 1
    sent_ngs = set(add_calls[0]['ng_names'] or [])
    assert sent_ngs == {'ng1', 'ng2', 'ng3', 'ng4', 'ng5'}

    manager_harness.finish_task(add_calls[0]['task_id'])

    statuses = _ng_statuses(manager_harness, 'kb1s3c', 'doc_new')
    assert set(statuses.keys()) == {'ng1', 'ng2', 'ng3', 'ng4', 'ng5'}

    # ng1 must appear exactly once
    NgStatus = mgr._db_manager.get_table_orm_class('lazyllm_doc_node_group_status')
    with mgr._db_manager.get_session() as session:
        count = session.query(NgStatus).filter(
            NgStatus.kb_id == 'kb1s3c', NgStatus.doc_id == 'doc_new',
            NgStatus.node_group_id == 'ng1',
        ).count()
    assert count == 1


# ---------------------------------------------------------------------------
# Scenario 4: delete docs from kb2 (algo1+algo3) → all ng_status rows removed
# ---------------------------------------------------------------------------

def test_multi_algo_scenario4_delete_doc_cleans_all_ng_status(manager_harness):
    '''Scenario 4: delete a doc from kb2 (algo1+algo3) → ng_status rows for all ngs removed.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb2s4', algo_id='algo1')
    mgr.update_kb('kb2s4', algo_id='algo3')

    fp, items = _upload_doc(manager_harness, 'kb2s4', 'doc_del', 'doc_del.txt')
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc_del']
    assert len(add_calls) == 1
    manager_harness.finish_task(add_calls[0]['task_id'])

    # Verify ng_status rows exist before delete
    statuses_before = _ng_statuses(manager_harness, 'kb2s4', 'doc_del')
    assert set(statuses_before.keys()) == {'ng1', 'ng2', 'ng3', 'ng6', 'ng7'}

    # Delete the doc
    del_items = mgr.delete(DeleteRequest(kb_id='kb2s4', doc_ids=['doc_del']))
    assert len(del_items) == 1

    # Finish delete tasks for both algos
    for call in manager_harness.delete_calls:
        manager_harness.finish_task(call['task_id'])

    # All ng_status rows must be gone
    statuses_after = _ng_statuses(manager_harness, 'kb2s4', 'doc_del')
    assert statuses_after == {}, 'all ng_status rows must be removed after full doc delete'


# ---------------------------------------------------------------------------
# Scenario 5: unbind algo1 from kb2 (algo1+algo3) → ng2/ng3 deleted, ng1 preserved
# ---------------------------------------------------------------------------

def test_multi_algo_scenario5_unbind_algo_preserves_shared_ng(manager_harness):
    '''Scenario 5: kb2 has algo1(ng1/ng2/ng3) + algo3(ng1/ng6/ng7).
    Unbind algo1 → parser receives node_group_ids_to_delete=[ng2,ng3] (not ng1).
    ng_status rows for ng2/ng3 removed; ng1/ng6/ng7 preserved.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb2s5', algo_id='algo1')
    mgr.update_kb('kb2s5', algo_id='algo3')

    fp, items = _upload_doc(manager_harness, 'kb2s5', 'doc_s5', 'doc_s5.txt')
    # Single task covers all algos
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc_s5']
    assert len(add_calls) == 1
    manager_harness.finish_task(add_calls[0]['task_id'])
    # Simulate parser writing SUCCESS for all ngs
    manager_harness.set_ng_status('kb2s5', 'doc_s5', ['ng1', 'ng2', 'ng3', 'ng6', 'ng7'], 'SUCCESS')

    statuses_before = _ng_statuses(manager_harness, 'kb2s5', 'doc_s5')
    assert set(statuses_before.keys()) == {'ng1', 'ng2', 'ng3', 'ng6', 'ng7'}

    # Unbind algo1
    result = mgr.unbind_algo('kb2s5', 'algo1')
    assert len(result['task_ids']) == 1

    # The delete task sent to parser must only target ng2/ng3 (algo1-exclusive)
    del_calls_with_ngs = [c for c in manager_harness.delete_calls if c['node_group_ids_to_delete']]
    assert len(del_calls_with_ngs) == 1
    ngs_to_delete = del_calls_with_ngs[0]['node_group_ids_to_delete']
    assert ngs_to_delete is not None
    assert set(ngs_to_delete) == {'ng2', 'ng3'}, \
        f'parser must only delete ng2/ng3; got {ngs_to_delete}'
    assert 'ng1' not in ngs_to_delete, 'shared ng1 must NOT be in node_group_ids_to_delete'

    # Finish the delete task
    manager_harness.finish_task(del_calls_with_ngs[0]['task_id'])

    # ng2/ng3 ng_status rows must be gone; ng1/ng6/ng7 must remain
    statuses_after = _ng_statuses(manager_harness, 'kb2s5', 'doc_s5')
    assert 'ng2' not in statuses_after, 'ng2 ng_status must be removed after unbind'
    assert 'ng3' not in statuses_after, 'ng3 ng_status must be removed after unbind'
    assert 'ng1' in statuses_after, 'shared ng1 ng_status must be preserved'
    assert 'ng6' in statuses_after
    assert 'ng7' in statuses_after

    # algo1 binding must be removed from kb_algorithm
    remaining_algos = mgr._get_kb_algorithms('kb2s5')
    assert 'algo1' not in remaining_algos
    assert 'algo3' in remaining_algos


# ---------------------------------------------------------------------------
# Scenario 6: reparse a single ng (ng3) in kb1
# ---------------------------------------------------------------------------

def test_multi_algo_scenario6_reparse_single_ng(manager_harness):
    '''Scenario 6: user requests reparse of ng3 only in kb1 (algo1). Only ng3 reset to PENDING.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1s6', algo_id='algo1')

    fp, items = _upload_doc(manager_harness, 'kb1s6', 'doc_s6', 'doc_s6.txt')
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc_s6']
    manager_harness.finish_task(add_calls[-1]['task_id'])
    # Simulate parser writing SUCCESS
    manager_harness.set_ng_status('kb1s6', 'doc_s6', ['ng1', 'ng2', 'ng3'], 'SUCCESS')

    statuses_before = _ng_statuses(manager_harness, 'kb1s6', 'doc_s6')
    assert statuses_before == {'ng1': 'SUCCESS', 'ng2': 'SUCCESS', 'ng3': 'SUCCESS'}

    task_ids = mgr.reparse(ReparseRequest(
        kb_id='kb1s6', doc_ids=['doc_s6'], ng_names=['ng3'],
    ))
    assert len(task_ids) == 1

    statuses_after = _ng_statuses(manager_harness, 'kb1s6', 'doc_s6')
    assert statuses_after['ng3'] == 'PENDING', 'ng3 must be reset to PENDING'
    assert statuses_after['ng1'] == 'SUCCESS', 'ng1 must remain SUCCESS'
    assert statuses_after['ng2'] == 'SUCCESS', 'ng2 must remain SUCCESS'

    # Parser must receive ng_names containing 'ng3' for the reparse task
    reparse_calls = [c for c in manager_harness.add_doc_calls
                     if c['algo_id'] == 'algo1' and c['task_type'] == TaskType.DOC_REPARSE.value]
    sent_ngs = reparse_calls[-1]['ng_names']
    assert sent_ngs is not None
    assert 'ng3' in sent_ngs


# ---------------------------------------------------------------------------
# Scenario 7: add_docs without algo_id → auto-resolves all bound algos
# ---------------------------------------------------------------------------

def test_multi_algo_scenario7_add_docs_auto_resolves_algos(manager_harness):
    '''Scenario 7: upload without algo_id → DocManager auto-queries kb bindings.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb1s7', algo_id='algo1')
    mgr.update_kb('kb1s7', algo_id='algo2')

    fp = manager_harness.make_file('doc_s7.txt', 'content')
    # No algo_id / algo_ids provided
    items = mgr.upload(UploadRequest(
        kb_id='kb1s7',
        items=[AddFileItem(file_path=fp, doc_id='doc_s7')],
    ))
    assert items[0]['accepted'] is True

    # Single task covers all algos' ng_names merged
    add_calls = [c for c in manager_harness.add_doc_calls if c['doc_id'] == 'doc_s7']
    assert len(add_calls) == 1
    sent_ngs = set(add_calls[0]['ng_names'] or [])
    assert sent_ngs == {'ng1', 'ng2', 'ng3', 'ng4', 'ng5'}

    # ng_status must cover ng1..ng5 (union of algo1 and algo2)
    statuses = _ng_statuses(manager_harness, 'kb1s7', 'doc_s7')
    assert set(statuses.keys()) == {'ng1', 'ng2', 'ng3', 'ng4', 'ng5'}


# ---------------------------------------------------------------------------
# Scenario 8: add_docs with no algo bound → must raise E_STATE_CONFLICT
# ---------------------------------------------------------------------------

def test_multi_algo_scenario8_upload_no_algo_bound_raises(manager_harness):
    '''Scenario 8: upload to a kb with no algo bound must raise E_STATE_CONFLICT.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb_noalgo', algo_id='algo1')
    # Manually remove the algo binding to simulate a kb with no algo
    KbAlgo = mgr._db_manager.get_table_orm_class('lazyllm_kb_algorithm')
    with mgr._db_manager.get_session() as session:
        session.query(KbAlgo).filter(KbAlgo.kb_id == 'kb_noalgo').delete()

    fp = manager_harness.make_file('noalgo.txt', 'content')
    with pytest.raises(DocServiceError) as exc_info:
        mgr.upload(UploadRequest(
            kb_id='kb_noalgo',
            items=[AddFileItem(file_path=fp, doc_id='doc_noalgo')],
        ))
    assert exc_info.value.biz_code == 'E_STATE_CONFLICT'


# ---------------------------------------------------------------------------
# Scenario 9: unbind last algo must be rejected
# ---------------------------------------------------------------------------

def test_multi_algo_scenario9_unbind_last_algo_rejected(manager_harness):
    '''Scenario 9: unbinding the last algo from a kb must raise E_STATE_CONFLICT.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb_last', algo_id='algo1')

    with pytest.raises(DocServiceError) as exc_info:
        mgr.unbind_algo('kb_last', 'algo1')
    assert exc_info.value.biz_code == 'E_STATE_CONFLICT'


# ---------------------------------------------------------------------------
# Scenario 10: reparse all when all ngs are shared → no task submitted
# ---------------------------------------------------------------------------

def test_multi_algo_scenario10_reparse_all_shared_ngs_no_task(manager_harness):
    '''Scenario 10: algo2 has ng1 (shared with algo1). Reparse all for algo2 when
    all its ngs are shared → pending_ng_ids is empty → no task submitted.'''
    mgr = manager_harness.manager

    # Create a special algo whose only ng is ng1 (fully shared with algo1)
    special_map = {'algo1': ['ng1', 'ng2', 'ng3'], 'algo_shared_only': ['ng1']}

    def get_algorithm_groups_special(algo_id):
        ng_ids = special_map.get(algo_id)
        if ng_ids is None:
            return BaseResponse(code=404, msg='algo not found', data=None)
        return BaseResponse(code=200, msg='success', data=[{'id': ng_id, 'name': ng_id} for ng_id in ng_ids])

    mgr._parser_client.list_algorithms = lambda: BaseResponse(
        code=200, msg='success',
        data=[{'algo_id': a, 'display_name': a, 'description': ''} for a in special_map],
    )
    mgr._parser_client.get_algorithm_groups = get_algorithm_groups_special

    mgr.create_kb('kb_shared_only', algo_id='algo1')
    mgr.update_kb('kb_shared_only', algo_id='algo_shared_only')

    fp, items = _upload_doc(manager_harness, 'kb_shared_only', 'doc_so', 'doc_so.txt')
    manager_harness.finish_task(items[0]['task_id'])
    # Finish all pending add_doc tasks (algo1 and algo_shared_only)
    for call in list(manager_harness.add_doc_calls):
        manager_harness.finish_task(call['task_id'])

    # Reparse all for algo_shared_only → all its ngs are shared → no new task
    add_calls_before = len(manager_harness.add_doc_calls)
    task_ids = mgr.reparse(ReparseRequest(
        kb_id='kb_shared_only', algo_ids=['algo_shared_only'], doc_ids=['doc_so'],
    ))
    assert task_ids == [], 'no reparse task should be submitted when all ngs are shared'
    assert len(manager_harness.add_doc_calls) == add_calls_before, \
        'no new add_doc call should be made'


# ======================================================================
# PR #1098 review-fix regression tests
# ======================================================================

# ---------------------------------------------------------------------------
# C2: _get_kb_algorithms merged — single-str guarantees non-empty
# ---------------------------------------------------------------------------

def test_get_kb_algorithms_single_str_raises_when_no_binding(manager_harness):
    '''_get_kb_algorithms(str) must raise E_STATE_CONFLICT when kb has no algo binding.'''
    mgr = manager_harness.manager
    # Create a kb then manually remove its algo binding
    mgr.create_kb('kb_no_binding', algo_id='__default__')
    KbAlgo = mgr._db_manager.get_table_orm_class('lazyllm_kb_algorithm')
    with mgr._db_manager.get_session() as session:
        session.query(KbAlgo).filter(KbAlgo.kb_id == 'kb_no_binding').delete()

    with pytest.raises(DocServiceError) as exc_info:
        mgr._get_kb_algorithms('kb_no_binding')
    assert exc_info.value.biz_code == 'E_STATE_CONFLICT'


def test_get_kb_algorithms_list_returns_dict_without_raising(manager_harness):
    '''_get_kb_algorithms(List[str]) must return a dict and never raise for missing kbs.'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_a', algo_id='__default__')
    result = mgr._get_kb_algorithms(['kb_a', 'kb_missing'])
    assert isinstance(result, dict)
    assert 'kb_a' in result
    assert len(result['kb_a']) > 0
    assert result.get('kb_missing', []) == []


def test_get_kb_algorithms_single_str_returns_list(manager_harness):
    '''_get_kb_algorithms(str) must return a non-empty List[str].'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_single', algo_id='__default__')
    result = mgr._get_kb_algorithms('kb_single')
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# B2: task record no longer stores algo_id; task_message stores algo_ids list
# ---------------------------------------------------------------------------

def test_task_record_has_no_algo_id_field(manager_harness):
    '''Task records must not have an algo_id column; algo_ids must be in task_message.'''
    mgr = manager_harness.manager
    mgr.create_kb('kb_task_b2', algo_id='__default__')
    fp = manager_harness.make_file('b2.txt', 'content')
    items = mgr.upload(UploadRequest(
        kb_id='kb_task_b2',
        items=[AddFileItem(file_path=fp, doc_id='doc_b2')],
    ))
    task_id = items[0]['task_id']

    task = mgr._get_task_record(task_id)
    assert task is not None
    # algo_id must NOT be a top-level field on the task record
    assert 'algo_id' not in task, 'algo_id should have been removed from task record'
    # algo_ids must be stored in task_message
    msg = task.get('message') or {}
    assert 'algo_ids' in msg, 'algo_ids must be stored in task_message'
    assert isinstance(msg['algo_ids'], list)
    assert len(msg['algo_ids']) > 0


# ---------------------------------------------------------------------------
# B1: unbind_algo fetches each snapshot exactly once
# ---------------------------------------------------------------------------

def test_unbind_algo_snapshot_fetched_once_in_filter_phase(manager_harness):
    '''Before the fix, the affected_doc_ids list comprehension called _get_parse_snapshot
    twice per doc (once for "is not None" and once for ".get(status)").
    After the fix, a pre-built snapshots dict is used so the filter phase calls it
    exactly once per doc.  We verify this by counting calls that happen *before*
    _enqueue_task is invoked (i.e. before any parser task is submitted).
    '''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb_b1', algo_id='algo1')
    mgr.update_kb('kb_b1', algo_id='algo2')

    # Upload 3 docs so the list-comprehension savings are visible
    doc_ids = ['doc_b1_a', 'doc_b1_b', 'doc_b1_c']
    for doc_id in doc_ids:
        fp = manager_harness.make_file(f'{doc_id}.txt', 'content')
        items = mgr.upload(UploadRequest(
            kb_id='kb_b1',
            items=[AddFileItem(file_path=fp, doc_id=doc_id)],
        ))
        manager_harness.finish_task(items[0]['task_id'])

    # Count calls that happen before _enqueue_task is first called
    filter_phase_calls = []
    enqueue_called = [False]
    original_snapshot = mgr._get_parse_snapshot
    original_enqueue = mgr._enqueue_task

    def tracking_snapshot(doc_id, kb_id):
        if not enqueue_called[0]:
            filter_phase_calls.append(doc_id)
        return original_snapshot(doc_id, kb_id)

    def tracking_enqueue(*args, **kwargs):
        enqueue_called[0] = True
        return original_enqueue(*args, **kwargs)

    mgr._get_parse_snapshot = tracking_snapshot
    mgr._enqueue_task = tracking_enqueue
    mgr.unbind_algo('kb_b1', 'algo1')

    # The filter phase (building snapshots dict + affected_doc_ids) should call
    # _get_parse_snapshot exactly once per doc — not twice as before the fix.
    from collections import Counter
    counts = Counter(filter_phase_calls)
    for doc_id in doc_ids:
        assert counts[doc_id] == 1, (
            f'doc {doc_id} snapshot fetched {counts[doc_id]} times in filter phase; '
            f'expected exactly 1 (before fix it was 2 due to double-call in list comprehension)'
        )


# ---------------------------------------------------------------------------
# B3: DOC_TRANSFER sets ng_status to PENDING
# ---------------------------------------------------------------------------

def test_transfer_sets_ng_status_pending(manager_harness):
    '''DOC_TRANSFER must set ng_status rows to PENDING for the target doc.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb_src_b3', algo_id='algo1')
    mgr.create_kb('kb_tgt_b3', algo_id='algo1')

    fp = manager_harness.make_file('b3_src.txt', 'content')
    items = mgr.upload(UploadRequest(
        kb_id='kb_src_b3',
        items=[AddFileItem(file_path=fp, doc_id='doc_src_b3')],
    ))
    manager_harness.finish_task(items[0]['task_id'])
    manager_harness.set_ng_status('kb_src_b3', 'doc_src_b3', ['ng1', 'ng2', 'ng3'], 'SUCCESS')

    from lazyllm.tools.rag.doc_service.base import TransferItem, TransferRequest
    mgr.transfer(TransferRequest(items=[TransferItem(
        doc_id='doc_src_b3', source_kb_id='kb_src_b3',
        target_doc_id='doc_tgt_b3', target_kb_id='kb_tgt_b3',
        mode='copy',
    )]))

    # ng_status rows for the target doc must be PENDING
    statuses = _ng_statuses(manager_harness, 'kb_tgt_b3', 'doc_tgt_b3')
    assert len(statuses) > 0, 'ng_status rows must be created for target doc on transfer'
    for ng_id, status in statuses.items():
        assert status == 'PENDING', f'ng {ng_id} status is {status!r}, expected PENDING'


# ---------------------------------------------------------------------------
# B5: unbind_algo_id empty string must not crash
# ---------------------------------------------------------------------------

def test_unbind_algo_empty_unbind_algo_id_does_not_crash(manager_harness):
    '''on_task_callback with unbind_algo=True but empty unbind_algo_id must not crash.'''
    mgr = manager_harness.manager
    _patch_multi_algo(manager_harness)
    mgr.create_kb('kb_b5', algo_id='algo1')
    mgr.update_kb('kb_b5', algo_id='algo2')

    fp = manager_harness.make_file('b5.txt', 'content')
    items = mgr.upload(UploadRequest(
        kb_id='kb_b5',
        items=[AddFileItem(file_path=fp, doc_id='doc_b5')],
    ))
    task_id = items[0]['task_id']

    # Manually inject a task_message with unbind_algo=True but empty unbind_algo_id
    Task = mgr._db_manager.get_table_orm_class('lazyllm_doc_service_tasks')
    import json as _json
    with mgr._db_manager.get_session() as session:
        row = session.query(Task).filter(Task.task_id == task_id).first()
        msg = _json.loads(row.message) if row.message else {}
        msg['unbind_algo'] = True
        msg['unbind_algo_id'] = ''  # intentionally empty
        msg['exclusive_ng_ids'] = []
        row.message = _json.dumps(msg)

    # Must not raise even with empty unbind_algo_id
    resp = manager_harness.finish_task(task_id)
    assert resp.get('ack') is True


# ---------------------------------------------------------------------------
# B7: ReparseRequest rejects empty algo_ids / ng_names lists
# ---------------------------------------------------------------------------

def test_reparse_request_rejects_empty_algo_ids():
    '''ReparseRequest must raise ValueError when algo_ids is an empty list.'''
    with pytest.raises(ValueError, match='algo_ids must not be an empty list'):
        ReparseRequest(doc_ids=['d1'], kb_id='kb1', algo_ids=[])


def test_reparse_request_rejects_empty_ng_names():
    '''ReparseRequest must raise ValueError when ng_names is an empty list.'''
    with pytest.raises(ValueError, match='ng_names must not be an empty list'):
        ReparseRequest(doc_ids=['d1'], kb_id='kb1', ng_names=[])


def test_reparse_request_accepts_none_algo_ids():
    '''ReparseRequest must accept algo_ids=None (reparse all algos).'''
    req = ReparseRequest(doc_ids=['d1'], kb_id='kb1', algo_ids=None)
    assert req.algo_ids is None


# ---------------------------------------------------------------------------
# C3: _get_algo_node_group_ids caches results per algo_id
# ---------------------------------------------------------------------------

def test_get_algo_node_group_ids_caches_result(manager_harness):
    '''_get_algo_node_group_ids must call the parser only once per algo_id.'''
    mgr = manager_harness.manager
    call_count = [0]
    original = mgr._parser_client.get_algorithm_groups

    def counting_get_algorithm_groups(algo_id):
        call_count[0] += 1
        return original(algo_id)

    mgr._parser_client.get_algorithm_groups = counting_get_algorithm_groups
    # Clear any existing cache
    mgr._algo_ng_cache.clear()

    mgr._get_algo_node_group_ids('__default__')
    mgr._get_algo_node_group_ids('__default__')
    mgr._get_algo_node_group_ids('__default__')

    assert call_count[0] == 1, \
        f'parser called {call_count[0]} times, expected 1 (cache should prevent repeated calls)'
