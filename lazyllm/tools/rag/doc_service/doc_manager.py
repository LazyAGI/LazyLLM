from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased

from lazyllm import LOG

from ..utils import BaseResponse, _get_default_db_config, _orm_to_dict
from ...sql import SqlManager
from .base import (
    AddRequest, CALLBACK_RECORDS_TABLE_INFO, CallbackEventType, DOC_SERVICE_TASKS_TABLE_INFO,
    DOC_PATH_LOCKS_TABLE_INFO, DOCUMENTS_TABLE_INFO, DeleteRequest, DocServiceError, DocStatus,
    DOC_NODE_GROUP_STATUS_TABLE_INFO, NodeGroupParseStatus,
    IDEMPOTENCY_RECORDS_TABLE_INFO, KB_ALGORITHM_TABLE_INFO, KB_DOCUMENTS_TABLE_INFO, KBS_TABLE_INFO,
    KBStatus, MetadataPatchRequest, PARSE_STATE_TABLE_INFO, ReparseRequest, SourceType,
    TaskCallbackRequest, TaskType, TransferRequest, UploadRequest,
)
from .parser_client import ParserClient
from .utils import (
    from_json, gen_doc_id, hash_payload, merge_transfer_metadata, resolve_transfer_target_path,
    sha256_file, stable_json, to_json,
)


class DocManager:
    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        parser_url: Optional[str] = None,
        callback_url: Optional[str] = None,
    ):
        if not parser_url:
            raise ValueError('parser_url is required')

        self._db_config = db_config or _get_default_db_config('doc_service')
        self._db_manager = SqlManager(
            **self._db_config,
            tables_info_dict={
                'tables': [
                    DOC_PATH_LOCKS_TABLE_INFO, DOCUMENTS_TABLE_INFO, KBS_TABLE_INFO, KB_DOCUMENTS_TABLE_INFO,
                    KB_ALGORITHM_TABLE_INFO, PARSE_STATE_TABLE_INFO, IDEMPOTENCY_RECORDS_TABLE_INFO,
                    CALLBACK_RECORDS_TABLE_INFO, DOC_SERVICE_TASKS_TABLE_INFO,
                    DOC_NODE_GROUP_STATUS_TABLE_INFO,
                ]
            },
        )
        self._ensure_indexes()
        self._parser_client = ParserClient(parser_url=parser_url)
        try:
            self._parser_client.health()
        except Exception as exc:
            raise RuntimeError(f'parser service is unavailable: {parser_url}') from exc
        self._cleanup_idempotency_records()
        self._callback_url = callback_url

    def set_callback_url(self, callback_url: str):
        self._callback_url = callback_url

    def close(self):
        '''Release the DB engine held by the underlying ``SqlManager``.

        Tests that point the manager at a temp-directory sqlite file need this so
        that ``TemporaryDirectory`` cleanup on Windows does not trip on an open
        sqlite handle.
        '''
        if getattr(self, '_db_manager', None) is not None:
            self._db_manager.dispose()

    def _ensure_indexes(self):
        stmts = [
            'DROP INDEX IF EXISTS uq_docs_path',
            'CREATE INDEX IF NOT EXISTS idx_docs_path ON lazyllm_documents(path)',
            'CREATE INDEX IF NOT EXISTS idx_documents_upload_status ON lazyllm_documents(upload_status)',
            'CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON lazyllm_documents(updated_at)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_display_name '
            'ON lazyllm_knowledge_bases(display_name) WHERE display_name IS NOT NULL',
            'CREATE INDEX IF NOT EXISTS idx_kb_created_at ON lazyllm_knowledge_bases(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_kb_updated_at ON lazyllm_knowledge_bases(updated_at)',
            'CREATE INDEX IF NOT EXISTS idx_kb_doc_count ON lazyllm_knowledge_bases(doc_count)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_documents ON lazyllm_kb_documents(kb_id, doc_id)',
            'CREATE INDEX IF NOT EXISTS idx_kb_documents_doc_id ON lazyllm_kb_documents(doc_id)',
            'CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_id ON lazyllm_kb_documents(kb_id)',
            'DROP INDEX IF EXISTS uq_kb_algorithm_kb_id',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_algorithm ON lazyllm_kb_algorithm(kb_id, algo_id)',
            'CREATE INDEX IF NOT EXISTS idx_kb_algorithm_algo_id ON lazyllm_kb_algorithm(algo_id)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_parse_state_key '
            'ON lazyllm_doc_parse_state(doc_id, kb_id, algo_id)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_parse_state_current_task '
            'ON lazyllm_doc_parse_state(current_task_id) WHERE current_task_id IS NOT NULL',
            'CREATE INDEX IF NOT EXISTS idx_parse_sched '
            'ON lazyllm_doc_parse_state(status, task_score, updated_at)',
            'CREATE INDEX IF NOT EXISTS idx_parse_lease '
            'ON lazyllm_doc_parse_state(status, lease_until)',
            'CREATE INDEX IF NOT EXISTS idx_parse_kb_algo_status '
            'ON lazyllm_doc_parse_state(kb_id, algo_id, status)',
            'CREATE INDEX IF NOT EXISTS idx_parse_task_type_status '
            'ON lazyllm_doc_parse_state(task_type, status, updated_at)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_idempotency_endpoint_key '
            'ON lazyllm_idempotency_records(endpoint, idempotency_key)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_callback_id '
            'ON lazyllm_callback_records(callback_id)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_service_task_id '
            'ON lazyllm_doc_service_tasks(task_id)',
            'CREATE INDEX IF NOT EXISTS idx_doc_service_task_status '
            'ON lazyllm_doc_service_tasks(status, updated_at)',
            'CREATE INDEX IF NOT EXISTS idx_doc_service_task_doc '
            'ON lazyllm_doc_service_tasks(doc_id, kb_id, algo_id)',
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_ng_status_key '
            'ON lazyllm_doc_node_group_status(doc_id, kb_id, node_group_id)',
            'CREATE INDEX IF NOT EXISTS idx_ng_status_kb_ng '
            'ON lazyllm_doc_node_group_status(kb_id, node_group_id, status)',
            'CREATE INDEX IF NOT EXISTS idx_ng_status_doc_kb '
            'ON lazyllm_doc_node_group_status(doc_id, kb_id)',
        ]
        for stmt in stmts:
            self._db_manager.execute_commit(stmt)

    def _ensure_default_kb(self):
        self._ensure_kb('__default__', display_name='__default__')
        self._ensure_kb_algorithm('__default__', '__default__')

    @staticmethod
    def _update_kb_row_fields(row, now: datetime, display_name: Optional[str] = None,
                              description: Optional[str] = None, owner_id: Optional[str] = None,
                              meta: Optional[Dict[str, Any]] = None, update_fields: Optional[Set[str]] = None):
        update_fields = update_fields or set()
        if 'display_name' in update_fields:
            row.display_name = display_name
        if 'description' in update_fields:
            row.description = description
        if 'owner_id' in update_fields:
            row.owner_id = owner_id
        if 'meta' in update_fields:
            row.meta = to_json(meta)
        if row.status == KBStatus.DELETED.value:
            row.status = KBStatus.ACTIVE.value
        row.updated_at = now

    def _ensure_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                   owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                   update_fields: Optional[Set[str]] = None):
        now = datetime.now()
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if row is None:
                row = Kb(kb_id=kb_id, display_name=display_name, description=description, doc_count=0,
                         status=KBStatus.ACTIVE.value, owner_id=owner_id, meta=to_json(meta),
                         created_at=now, updated_at=now)
            else:
                self._update_kb_row_fields(
                    row, now, display_name=display_name, description=description,
                    owner_id=owner_id, meta=meta, update_fields=update_fields,
                )
            session.add(row)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
                if row is None:
                    raise
                self._update_kb_row_fields(
                    row, now, display_name=display_name, description=description,
                    owner_id=owner_id, meta=meta, update_fields=update_fields,
                )
                session.add(row)

    def _ensure_kb_algorithm(self, kb_id: str, algo_id: str):
        '''Upsert a (kb_id, algo_id) binding. A kb can be bound to multiple algos.'''
        now = datetime.now()
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            row = session.query(Rel).filter(Rel.kb_id == kb_id, Rel.algo_id == algo_id).first()
            if row is None:
                session.add(Rel(kb_id=kb_id, algo_id=algo_id, created_at=now, updated_at=now))
            else:
                row.updated_at = now
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                # concurrent insert won the race; update updated_at on the existing row
                row = session.query(Rel).filter(Rel.kb_id == kb_id, Rel.algo_id == algo_id).first()
                if row is not None:
                    row.updated_at = now

    def _get_kb(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            return _orm_to_dict(row) if row else None

    def _get_kb_algorithms(self, kb_id: str) -> List[str]:
        '''Return all algo_ids bound to this kb.'''
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            rows = session.query(Rel).filter(Rel.kb_id == kb_id).all()
            return [row.algo_id for row in rows]

    @staticmethod
    def _build_kb_data(kb_row, algo_ids=None):
        return {
            'kb_id': kb_row.kb_id,
            'display_name': kb_row.display_name,
            'description': kb_row.description,
            'doc_count': kb_row.doc_count,
            'status': kb_row.status,
            'owner_id': kb_row.owner_id,
            'meta': from_json(kb_row.meta),
            'algo_ids': algo_ids or [],
            'created_at': kb_row.created_at,
            'updated_at': kb_row.updated_at,
        }

    def _list_active_kb_algo_pairs(self) -> List[tuple]:
        '''Return all active (kb_id, algo_id) bindings for scan iteration.'''
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            rows = (
                session.query(Rel.kb_id, Rel.algo_id)
                .join(Kb, Rel.kb_id == Kb.kb_id)
                .filter(Kb.status == KBStatus.ACTIVE.value)
                .all()
            )
            return [(row.kb_id, row.algo_id) for row in rows]

    def _validate_kb_algorithm(self, kb_id: str, algo_id: str):
        if kb_id == '__default__':
            self._ensure_default_kb()
        kb = self._get_kb(kb_id)
        if kb is None:
            raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
        if kb.get('status') != KBStatus.ACTIVE.value:
            raise DocServiceError('E_STATE_CONFLICT', f'kb is not active: {kb_id}', {'kb_id': kb_id})
        algo_ids = self._get_kb_algorithms(kb_id)
        if not algo_ids:
            raise DocServiceError('E_STATE_CONFLICT', f'kb has no algorithm binding: {kb_id}', {'kb_id': kb_id})
        if algo_id not in algo_ids:
            raise DocServiceError(
                'E_INVALID_PARAM', f'algo {algo_id} is not bound to kb {kb_id}',
                {'kb_id': kb_id, 'algo_ids': algo_ids, 'requested_algo_id': algo_id}
            )

    def _ensure_algorithm_exists(self, algo_id: str):
        algorithms = self.list_algorithms()
        if any(item.get('algo_id') == algo_id for item in algorithms):
            return
        raise DocServiceError('E_INVALID_PARAM', f'invalid algo_id: {algo_id}', {'algo_id': algo_id})

    def _refresh_kb_doc_count(self, kb_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            kb_row = sess.query(Kb).filter(Kb.kb_id == kb_id).first()
            if kb_row is None:
                return
            kb_row.doc_count = sess.query(Rel).filter(Rel.kb_id == kb_id).count()
            if kb_row.status == KBStatus.DELETING.value and kb_row.doc_count == 0:
                kb_row.status = KBStatus.DELETED.value
            kb_row.updated_at = datetime.now()
            sess.add(kb_row)

    def _ensure_kb_document(self, kb_id: str, doc_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            now = datetime.now()
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            row = sess.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first()
            if row is None:
                sess.add(Rel(kb_id=kb_id, doc_id=doc_id, created_at=now, updated_at=now))
                created = True
            else:
                row.updated_at = now
                sess.add(row)
                created = False
            if created:
                self._refresh_kb_doc_count(kb_id, session=sess)
            return created

    def _remove_kb_document(self, kb_id: str, doc_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            row = sess.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first()
            if row is None:
                return False
            sess.delete(row)
            self._refresh_kb_doc_count(kb_id, session=sess)
            return True

    def _load_idempotency_record(self, endpoint: str, idempotency_key: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            row = session.query(Record).filter(
                Record.endpoint == endpoint,
                Record.idempotency_key == idempotency_key,
            ).first()
            return _orm_to_dict(row) if row else None

    def _cleanup_idempotency_records(self, ttl_days: int = 7):
        cutoff = datetime.now() - timedelta(days=ttl_days)
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            session.query(Record).filter(Record.updated_at < cutoff).delete()

    def _claim_idempotency_key(self, endpoint: str, idempotency_key: str, req_hash: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            now = datetime.now()
            row = Record(
                endpoint=endpoint,
                idempotency_key=idempotency_key,
                req_hash=req_hash,
                status='PROCESSING',
                response_json=None,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
        return True

    def _complete_idempotency_record(self, endpoint: str, idempotency_key: str, response: Any):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            row = session.query(Record).filter(
                Record.endpoint == endpoint,
                Record.idempotency_key == idempotency_key,
            ).first()
            if row is None:
                return
            row.status = 'COMPLETED'
            row.response_json = stable_json(response)
            row.updated_at = datetime.now()
            session.add(row)

    def _drop_idempotency_claim(self, endpoint: str, idempotency_key: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            row = session.query(Record).filter(
                Record.endpoint == endpoint,
                Record.idempotency_key == idempotency_key,
            ).first()
            if row is not None and row.status == 'PROCESSING':
                session.delete(row)

    def run_idempotent(self, endpoint: str, idempotency_key: Optional[str], payload: Any, handler):
        if not idempotency_key:
            return handler()
        req_hash = hash_payload(payload)
        try:
            self._claim_idempotency_key(endpoint, idempotency_key, req_hash)
        except IntegrityError:
            record = self._load_idempotency_record(endpoint, idempotency_key)
            if record is None:
                raise DocServiceError('E_IDEMPOTENCY_IN_PROGRESS', 'idempotency request is being processed')
            if record['req_hash'] != req_hash:
                raise DocServiceError('E_IDEMPOTENCY_CONFLICT', 'idempotency key conflicts with different request')
            if record.get('status') == 'COMPLETED' and record.get('response_json'):
                return json.loads(record['response_json'])
            raise DocServiceError(
                'E_IDEMPOTENCY_IN_PROGRESS', 'idempotency request is being processed',
                {'endpoint': endpoint, 'idempotency_key': idempotency_key}
            )
        try:
            response = handler()
        except Exception:
            self._drop_idempotency_claim(endpoint, idempotency_key)
            raise
        try:
            self._complete_idempotency_record(endpoint, idempotency_key, response)
        except Exception:
            self._drop_idempotency_claim(endpoint, idempotency_key)
            raise
        return response

    def _record_callback(self, callback_id: str, task_id: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(CALLBACK_RECORDS_TABLE_INFO['name'])
            session.add(Record(callback_id=callback_id, task_id=task_id, created_at=datetime.now()))
            try:
                session.flush()
                return True
            except IntegrityError:
                session.rollback()
                return False

    def _forget_callback_record(self, callback_id: str, task_id: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(CALLBACK_RECORDS_TABLE_INFO['name'])
            session.query(Record).filter(
                Record.callback_id == callback_id,
                Record.task_id == task_id,
            ).delete()

    def _create_task_record(self, task_id: str, task_type: TaskType, doc_id: str, kb_id: str, algo_id: str,
                            status: DocStatus, message: Optional[Dict[str, Any]] = None):
        now = datetime.now()
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            row = Task(
                task_id=task_id,
                task_type=task_type.value,
                doc_id=doc_id,
                kb_id=kb_id,
                algo_id=algo_id,
                status=status.value,
                message=to_json(message),
                error_code=None,
                error_msg=None,
                created_at=now,
                updated_at=now,
                started_at=None,
                finished_at=None,
            )
            session.add(row)
            session.flush()
            task = _orm_to_dict(row)
            task['message'] = from_json(task.get('message'))
            return task

    def _get_task_record(self, task_id: str):
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            row = session.query(Task).filter(Task.task_id == task_id).first()
            if row is None:
                return None
            task = _orm_to_dict(row)
            task['message'] = from_json(task.get('message'))
            return task

    def _update_task_record(self, task_id: str, **fields):
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            row = session.query(Task).filter(Task.task_id == task_id).first()
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = datetime.now()
            session.add(row)
            session.flush()
            task = _orm_to_dict(row)
            task['message'] = from_json(task.get('message'))
            return task

    def _list_kb_doc_ids(self, kb_id: str) -> List[str]:
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            rows = session.query(Rel.doc_id).filter(Rel.kb_id == kb_id).all()
            return [row[0] for row in rows]

    def _has_kb_document(self, kb_id: str, doc_id: str):
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            return session.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first() is not None

    def _doc_relation_count(self, doc_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            return sess.query(Rel).filter(Rel.doc_id == doc_id).count()

    def _get_doc(self, doc_id: str):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            return _orm_to_dict(row) if row else None

    def _get_doc_by_path(self, path: str):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.path == path).first()
            return _orm_to_dict(row) if row else None

    def _list_kb_docs_by_path(self, kb_id: str, exclude_failed: bool = True) -> Dict[str, str]:
        '''Query documents+kb_documents tables, return {path: doc_id}.

        Args:
            kb_id: Knowledge base ID to filter by.
            exclude_failed: If True (default), also excludes FAILED and CANCELED docs so scan
                can retry them. If False, only excludes DELETED — useful for stale-file cleanup
                where we need to see failed docs whose source files were removed from disk.
        '''
        _exclude = [DocStatus.DELETED.value]
        if exclude_failed:
            _exclude.extend([DocStatus.FAILED.value, DocStatus.CANCELED.value])
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            rows = (
                session.query(Doc.path, Doc.doc_id)
                .join(Rel, Doc.doc_id == Rel.doc_id)
                .filter(Rel.kb_id == kb_id)
                .filter(~Doc.upload_status.in_(_exclude))
                .all()
            )
            return {row.path: row.doc_id for row in rows if row.path}

    def _upsert_doc(self, doc_id: str, filename: str, path: str, metadata: Dict[str, Any],
                    source_type: SourceType, upload_status: DocStatus = DocStatus.SUCCESS,
                    allowed_path_doc_ids: Optional[Set[str]] = None, session=None):
        try:
            with self._db_manager.get_session(session) as sess:
                now = datetime.now()
                file_type = os.path.splitext(path)[1].lstrip('.').lower() or None
                size_bytes = os.path.getsize(path) if os.path.exists(path) else None
                content_hash = sha256_file(path) if os.path.exists(path) else None
                allowed_path_doc_ids = allowed_path_doc_ids or set()
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                PathLock = self._db_manager.get_table_orm_class(DOC_PATH_LOCKS_TABLE_INFO['name'])
                sess.add(PathLock(path=path, created_at=now))
                sess.flush()
                try:
                    row = sess.query(Doc).filter(Doc.doc_id == doc_id).first()
                    path_rows = sess.query(Doc).filter(Doc.path == path).all()
                    conflict = next(
                        (r for r in path_rows if r.doc_id != doc_id and r.doc_id not in allowed_path_doc_ids),
                        None,
                    )
                    if conflict is not None:
                        raise DocServiceError('E_STATE_CONFLICT', f'doc path already exists: {path}',
                                              {'doc_id': conflict.doc_id, 'path': path})
                    if row is None:
                        row = Doc(doc_id=doc_id, filename=filename, path=path, meta=to_json(metadata),
                                  upload_status=upload_status.value, source_type=source_type.value,
                                  file_type=file_type, content_hash=content_hash, size_bytes=size_bytes,
                                  created_at=now, updated_at=now)
                    else:
                        row.filename, row.path, row.meta = filename, path, to_json(metadata)
                        row.upload_status = upload_status.value
                        row.source_type = source_type.value
                        row.file_type, row.content_hash, row.size_bytes = file_type, content_hash, size_bytes
                        row.updated_at = now
                    sess.add(row)
                    sess.flush()
                    return _orm_to_dict(row)
                finally:
                    try:
                        sess.query(PathLock).filter(PathLock.path == path).delete()
                        sess.flush()
                    except Exception:
                        pass
        except IntegrityError as exc:
            existing = self._get_doc_by_path(path)
            if (existing is not None and existing.get('doc_id') != doc_id
                    and existing.get('doc_id') not in (allowed_path_doc_ids or set())):
                raise DocServiceError('E_STATE_CONFLICT', f'doc path already exists: {path}',
                                      {'doc_id': existing['doc_id'], 'path': path}) from exc
            raise

    def _upsert_doc_and_bind(self, kb_id: str, doc_id: str, filename: str, path: str,
                             metadata: Dict[str, Any], source_type: SourceType,
                             upload_status: DocStatus = DocStatus.SUCCESS,
                             allowed_path_doc_ids: Optional[Set[str]] = None):
        with self._db_manager.get_session() as session:
            doc_dict = self._upsert_doc(doc_id, filename, path, metadata, source_type,
                                        upload_status=upload_status,
                                        allowed_path_doc_ids=allowed_path_doc_ids, session=session)
            self._ensure_kb_document(kb_id, doc_id, session=session)
            return doc_dict

    def _set_doc_upload_status(self, doc_id: str, status: DocStatus, session=None) -> None:
        with self._db_manager.get_session(session) as sess:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = sess.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                return
            row.upload_status = status.value
            row.updated_at = datetime.now()
            sess.add(row)

    def _get_parse_snapshot(self, doc_id: str, kb_id: str, algo_id: str):
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            row = (
                session.query(State)
                .filter(State.doc_id == doc_id, State.kb_id == kb_id, State.algo_id == algo_id)
                .first()
            )
            return _orm_to_dict(row) if row else None

    def _upsert_ng_status_pending(self, doc_id: str, kb_id: str,
                                  node_group_ids: List[str], file_path: Optional[str] = None,
                                  force: bool = False):
        if not node_group_ids:
            return
        now = datetime.now()
        with self._db_manager.get_session() as session:
            NgStatus = self._db_manager.get_table_orm_class(DOC_NODE_GROUP_STATUS_TABLE_INFO['name'])
            existing_rows = session.query(NgStatus).filter(
                NgStatus.doc_id == doc_id, NgStatus.kb_id == kb_id,
                NgStatus.node_group_id.in_(node_group_ids),
            ).all()
            existing_map = {row.node_group_id: row for row in existing_rows}
            for ng_id in node_group_ids:
                if ng_id in existing_map:
                    row = existing_map[ng_id]
                    if not force and row.status == NodeGroupParseStatus.SUCCESS.value:
                        continue
                    row.status = NodeGroupParseStatus.PENDING.value
                    row.file_path = file_path or row.file_path
                    row.updated_at = now
                else:
                    session.add(NgStatus(
                        doc_id=doc_id, kb_id=kb_id,
                        node_group_id=ng_id, file_path=file_path,
                        status=NodeGroupParseStatus.PENDING.value,
                        created_at=now, updated_at=now,
                    ))

    def _get_algo_node_group_ids(self, algo_id: str) -> List[str]:
        try:
            resp = self._parser_client.get_algorithm_groups(algo_id)
            if resp and resp.code == 200 and resp.data:
                data = resp.data
                # Support list-of-dict format: [{id, name, ...}, ...]
                if isinstance(data, list):
                    result = [g['id'] for g in data if g.get('id')]
                    if result:
                        return result
                # Support legacy dict format: {node_group_ids: [...]}
                if isinstance(data, dict):
                    ids = data.get('node_group_ids')
                    if ids:
                        result = json.loads(ids) if isinstance(ids, str) else ids
                        if result:
                            return result
            raise ValueError(f'Failed to get node_group_ids for algo {algo_id!r}: '
                             f'code={getattr(resp, "code", None)}, msg={getattr(resp, "msg", None)!r}')
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f'[DocManager] Failed to get node_group_ids for algo {algo_id}: {e}') from e

    def _get_shared_ng_ids(self, kb_id: str, algo_id: str, candidate_ng_ids: List[str]) -> Set[str]:
        # Returns the subset of candidate_ng_ids that are also owned by other algos in the same kb.
        other_algo_ids = [a for a in self._get_kb_algorithms(kb_id) if a != algo_id]
        shared: Set[str] = set()
        candidate_set = set(candidate_ng_ids)
        for other_algo_id in other_algo_ids:
            other_ngs = set(self._get_algo_node_group_ids(other_algo_id))
            shared |= candidate_set & other_ngs
        return shared

    def _get_latest_parse_snapshot(self, doc_id: str, kb_id: str):
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            row = (
                session.query(State)
                .filter(State.doc_id == doc_id, State.kb_id == kb_id)
                .order_by(State.updated_at.desc(), State.created_at.desc())
                .first()
            )
            return _orm_to_dict(row) if row else None

    def _delete_parse_snapshots(self, doc_id: str, kb_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            sess.query(State).filter(State.doc_id == doc_id, State.kb_id == kb_id).delete()

    def _delete_parse_snapshot_for_algo(self, doc_id: str, kb_id: str, algo_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            sess.query(State).filter(
                State.doc_id == doc_id, State.kb_id == kb_id, State.algo_id == algo_id,
            ).delete()

    def _delete_ng_status(self, doc_id: str, kb_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            NgStatus = self._db_manager.get_table_orm_class(DOC_NODE_GROUP_STATUS_TABLE_INFO['name'])
            sess.query(NgStatus).filter(NgStatus.doc_id == doc_id, NgStatus.kb_id == kb_id).delete()

    def _delete_ng_status_for_groups(self, doc_id: str, kb_id: str, ng_ids: List[str], session=None):
        if not ng_ids:
            return
        with self._db_manager.get_session(session) as sess:
            NgStatus = self._db_manager.get_table_orm_class(DOC_NODE_GROUP_STATUS_TABLE_INFO['name'])
            sess.query(NgStatus).filter(
                NgStatus.doc_id == doc_id,
                NgStatus.kb_id == kb_id,
                NgStatus.node_group_id.in_(ng_ids),
            ).delete(synchronize_session='fetch')

    def _all_algo_snapshots_deleted(self, doc_id: str, kb_id: str) -> bool:
        '''Return True if every algo's parse_state for (doc_id, kb_id) is DELETED (or absent).'''
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            rows = session.query(State).filter(
                State.doc_id == doc_id, State.kb_id == kb_id,
            ).all()
            return bool(rows) and all(r.status == DocStatus.DELETED.value for r in rows)

    def _count_non_deleted_snapshots_for_algo(self, kb_id: str, algo_id: str) -> int:
        '''Return count of docs in kb that still have a non-DELETED parse_state for algo_id.'''
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            return session.query(State).filter(
                State.kb_id == kb_id,
                State.algo_id == algo_id,
                State.status != DocStatus.DELETED.value,
            ).count()

    def _delete_doc_if_orphaned(self, doc_id: str, session=None) -> bool:
        with self._db_manager.get_session(session) as sess:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            sess.flush()
            if sess.query(Rel).filter(Rel.doc_id == doc_id).count() > 0:
                return False
            row = sess.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                return False
            sess.delete(row)
            return True

    def _remove_kb_document_and_delete_orphan(self, kb_id: str, doc_id: str) -> bool:
        with self._db_manager.get_session() as session:
            self._remove_kb_document(kb_id, doc_id, session=session)
            return self._delete_doc_if_orphaned(doc_id, session=session)

    def _purge_deleted_kb_doc_data(self, kb_id: str, doc_id: str, remove_relation: bool = False):
        doc_deleted = False
        if remove_relation:
            doc_deleted = self._remove_kb_document_and_delete_orphan(kb_id, doc_id)
        else:
            doc_deleted = self._delete_doc_if_orphaned(doc_id)
        self._delete_parse_snapshots(doc_id, kb_id)
        self._delete_ng_status(doc_id, kb_id)
        if not doc_deleted:
            self._sync_doc_upload_status(doc_id)

    def _mark_task_cleanup_policy(self, task_id: str, cleanup_policy: str):
        task = self._get_task_record(task_id)
        if task is None:
            return
        message = task.get('message') or {}
        if message.get('cleanup_policy') == cleanup_policy:
            return
        message['cleanup_policy'] = cleanup_policy
        self._update_task_record(task_id, message=to_json(message))

    def _finalize_kb_deletion_if_empty(self, kb_id: str) -> bool:
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            if session.query(Rel).filter(Rel.kb_id == kb_id).count() > 0:
                return False
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            AlgoRel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            kb_row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if kb_row is not None:
                session.delete(kb_row)
            session.query(AlgoRel).filter(AlgoRel.kb_id == kb_id).delete()
        return True

    def _prepare_kb_delete_items(self, kb_id: str) -> Dict[str, Any]:
        kb = self._get_kb(kb_id)
        if kb is None:
            raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
        binding = self._get_kb_algorithms(kb_id)
        default_algo_id = binding[0] if binding else '__default__'
        items = []
        for doc_id in self._list_kb_doc_ids(kb_id):
            snapshot = (
                self._get_latest_parse_snapshot(doc_id, kb_id)
            )
            if snapshot is None or snapshot.get('status') == DocStatus.DELETED.value:
                items.append({'action': 'purge_local', 'doc_id': doc_id})
                continue
            status = snapshot.get('status')
            task_type = snapshot.get('task_type')
            if status in (DocStatus.WAITING.value, DocStatus.WORKING.value):
                if task_type == TaskType.DOC_DELETE.value and snapshot.get('current_task_id'):
                    items.append({
                        'action': 'reuse_delete_task',
                        'doc_id': doc_id,
                        'task_id': snapshot['current_task_id'],
                    })
                    continue
                raise DocServiceError(
                    'E_STATE_CONFLICT',
                    f'cannot delete kb while doc {doc_id} task is {status}',
                    {'kb_id': kb_id, 'doc_id': doc_id, 'status': status, 'task_type': task_type},
                )
            # enqueue a delete task for each algo bound to this kb
            for idx, algo_id in enumerate(binding or [default_algo_id]):
                items.append({
                    'action': 'enqueue_delete',
                    'doc_id': doc_id,
                    'algo_id': algo_id,
                    # mark only the first algo's task as "primary" (responsible for ng_status cleanup)
                    'primary': idx == 0,
                })
        return {'kb': kb, 'items': items}

    def _assert_action_allowed(self, doc_id: str, kb_id: str, algo_id: str, action: str):
        snapshot = self._get_parse_snapshot(doc_id, kb_id, algo_id)
        status = snapshot.get('status') if snapshot is not None else None
        if status is None and action in ('add', 'upload'):
            doc = self._get_doc(doc_id)
            status = doc.get('upload_status') if doc else None

        if action in ('add', 'upload'):
            if status in (
                DocStatus.WAITING.value,
                DocStatus.WORKING.value,
                DocStatus.DELETING.value,
                DocStatus.SUCCESS.value,
            ):
                raise DocServiceError('E_STATE_CONFLICT', f'cannot {action} while state is {status}')
            return

        if status == DocStatus.WORKING.value and action in ('reparse', 'delete', 'transfer', 'metadata'):
            raise DocServiceError('E_STATE_CONFLICT', f'cannot {action} while state is WORKING')
        if status == DocStatus.DELETING.value and action in ('reparse', 'delete', 'transfer', 'metadata'):
            raise DocServiceError('E_STATE_CONFLICT', f'cannot {action} while state is DELETING')

    def _upsert_parse_snapshot(
        self,
        doc_id: str,
        kb_id: str,
        algo_id: str,
        status: DocStatus,
        task_type: Optional[TaskType] = None,
        current_task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        priority: int = 0,
        task_score: Optional[int] = None,
        retry_count: int = 0,
        max_retry: int = 3,
        lease_owner: Optional[str] = None,
        lease_until: Optional[datetime] = None,
        error_code: Optional[str] = None,
        error_msg: Optional[str] = None,
        failed_stage: Optional[str] = None,
        queued_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ):
        now = datetime.now()
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            row = (
                session.query(State)
                .filter(State.doc_id == doc_id, State.kb_id == kb_id, State.algo_id == algo_id)
                .first()
            )
            if row is None:
                row = State(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    algo_id=algo_id,
                    status=status.value,
                    task_type=task_type.value if task_type else None,
                    current_task_id=current_task_id,
                    idempotency_key=idempotency_key,
                    priority=priority,
                    task_score=task_score,
                    retry_count=retry_count,
                    max_retry=max_retry,
                    lease_owner=lease_owner,
                    lease_until=lease_until,
                    last_error_code=error_code,
                    last_error_msg=error_msg,
                    failed_stage=failed_stage,
                    queued_at=queued_at,
                    started_at=started_at,
                    finished_at=finished_at,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.status = status.value
                if task_type is not None:
                    row.task_type = task_type.value
                row.current_task_id = current_task_id
                row.idempotency_key = idempotency_key
                row.priority = priority
                row.task_score = task_score
                row.retry_count = retry_count
                row.max_retry = max_retry
                row.lease_owner = lease_owner
                row.lease_until = lease_until
                row.last_error_code = error_code
                row.last_error_msg = error_msg
                row.failed_stage = failed_stage
                row.queued_at = queued_at
                row.started_at = started_at
                row.finished_at = finished_at
                row.updated_at = now
            session.add(row)
        return self._get_parse_snapshot(doc_id, kb_id, algo_id)

    def _validate_unique_doc_ids(self, doc_ids: List[str], field_name: str = 'doc_id'):
        duplicated = set()
        seen = set()
        for doc_id in doc_ids:
            if doc_id in seen:
                duplicated.add(doc_id)
            seen.add(doc_id)
        if duplicated:
            duplicated_list = sorted(duplicated)
            raise DocServiceError(
                'E_INVALID_PARAM',
                f'duplicate {field_name} detected',
                {f'duplicate_{field_name}s': duplicated_list},
            )

    def _create_parser_task(self, task_id: str, doc_id: str, kb_id: str, algo_id: str, task_type: TaskType,
                            ng_ids: Optional[List[str]] = None,
                            file_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                            reparse_group: Optional[str] = None, parser_kb_id: Optional[str] = None,
                            transfer_params: Optional[Dict[str, Any]] = None,
                            node_group_ids_to_delete: Optional[List[str]] = None):
        if task_type in (TaskType.DOC_ADD, TaskType.DOC_TRANSFER):
            if not file_path:
                raise RuntimeError(f'file_path is required for task_type {task_type.value}')
            task_resp = self._parser_client.add_doc(
                task_id, parser_kb_id or kb_id, doc_id, file_path, metadata,
                ng_ids=ng_ids, callback_url=self._callback_url, transfer_params=transfer_params,
            )
        elif task_type == TaskType.DOC_REPARSE:
            if not file_path:
                raise RuntimeError('file_path is required for reparse task')
            task_resp = self._parser_client.add_doc(
                task_id, kb_id, doc_id, file_path, metadata,
                ng_ids=ng_ids, reparse_group=reparse_group or 'all', callback_url=self._callback_url,
            )
        elif task_type == TaskType.DOC_UPDATE_META:
            task_resp = self._parser_client.update_meta(
                task_id, kb_id, doc_id, metadata, file_path, callback_url=self._callback_url,
            )
        elif task_type == TaskType.DOC_DELETE:
            task_resp = self._parser_client.delete_doc(
                task_id, kb_id, doc_id, callback_url=self._callback_url,
                node_group_ids_to_delete=node_group_ids_to_delete,
            )
        else:
            raise RuntimeError(f'unsupported task type: {task_type.value}')
        if task_resp.code != 200:
            raise RuntimeError(f'failed to enqueue parser task: {task_resp.msg}')
        return task_id

    def _enqueue_task(
        self, doc_id: str, kb_id: str, algo_id: str, task_type: TaskType,
        idempotency_key: Optional[str] = None, priority: int = 0,
        file_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        reparse_group: Optional[str] = None, cleanup_policy: Optional[str] = None,
        parser_kb_id: Optional[str] = None, transfer_params: Optional[Dict[str, Any]] = None,
        extra_message: Optional[Dict[str, Any]] = None, parser_doc_id: Optional[str] = None,
    ):
        task_id = str(uuid4())
        task_message = {
            'doc_id': doc_id,
            'kb_id': kb_id,
            'algo_id': algo_id,
            'file_path': file_path,
            'metadata': metadata,
            'reparse_group': reparse_group,
        }
        if extra_message:
            task_message.update(extra_message)
        if cleanup_policy:
            task_message['cleanup_policy'] = cleanup_policy
        if transfer_params:
            task_message['transfer_params'] = transfer_params
        task_status = DocStatus.DELETING if task_type == TaskType.DOC_DELETE else DocStatus.WAITING
        self._create_task_record(task_id, task_type, doc_id, kb_id, algo_id, task_status, message=task_message)
        parse_status = DocStatus.DELETING if task_type == TaskType.DOC_DELETE else DocStatus.WAITING
        self._upsert_parse_snapshot(
            doc_id=doc_id,
            kb_id=kb_id,
            algo_id=algo_id,
            status=parse_status,
            task_type=task_type,
            current_task_id=task_id,
            idempotency_key=idempotency_key,
            priority=priority,
            queued_at=datetime.now(),
            started_at=None,
            finished_at=None,
            error_code=None,
            error_msg=None,
            failed_stage=None,
        )
        try:
            exclusive_ng_ids = (
                (extra_message or {}).get('exclusive_ng_ids') if task_type == TaskType.DOC_DELETE else None
            )
            # Resolve ng_ids for this algo before submitting to the parser queue.
            # For ADD/REPARSE tasks, ng_ids are passed to the Worker so it knows which ngs to process.
            ng_ids = None
            if task_type in (TaskType.DOC_ADD, TaskType.DOC_REPARSE, TaskType.DOC_TRANSFER):
                ng_ids = self._get_algo_node_group_ids(algo_id)
            elif task_type == TaskType.DOC_DELETE and exclusive_ng_ids:
                ng_ids = exclusive_ng_ids
            # Insert PENDING status for each node group before submitting the parser task,
            # so that on_task_callback can always find the status rows even if submission fails.
            if task_type == TaskType.DOC_ADD:
                self._upsert_ng_status_pending(doc_id, kb_id, ng_ids or [], file_path)
            self._create_parser_task(
                task_id, parser_doc_id or doc_id, kb_id, algo_id, task_type,
                ng_ids=ng_ids,
                file_path=file_path, metadata=metadata, reparse_group=reparse_group,
                parser_kb_id=parser_kb_id, transfer_params=transfer_params,
                node_group_ids_to_delete=exclusive_ng_ids,
            )
        except Exception as exc:
            finished_at = datetime.now()
            error_msg = str(exc)
            self._update_task_record(
                task_id,
                status=DocStatus.FAILED.value,
                error_code='PARSER_SUBMIT_FAILED',
                error_msg=error_msg,
                finished_at=finished_at,
            )
            self._upsert_parse_snapshot(
                doc_id=doc_id,
                kb_id=kb_id,
                algo_id=algo_id,
                status=DocStatus.FAILED,
                **self._build_snapshot_update(
                    self._get_parse_snapshot(doc_id, kb_id, algo_id),
                    task_type=task_type,
                    current_task_id=task_id,
                    error_code='PARSER_SUBMIT_FAILED',
                    error_msg=error_msg,
                    failed_stage='SUBMIT',
                    finished_at=finished_at,
                ),
            )
            self._apply_doc_upload_status(doc_id, task_type, DocStatus.FAILED)
            raise
        return task_id, self._get_parse_snapshot(doc_id, kb_id, algo_id)

    def _apply_doc_upload_status(self, doc_id: str, task_type: TaskType, status: DocStatus):
        if task_type == TaskType.DOC_ADD:
            if status in (DocStatus.WORKING, DocStatus.FAILED, DocStatus.CANCELED, DocStatus.SUCCESS):
                self._set_doc_upload_status(doc_id, status)
            return
        if task_type == TaskType.DOC_DELETE:
            if status == DocStatus.DELETING:
                relation_count = self._doc_relation_count(doc_id)
                if relation_count <= 1:
                    self._set_doc_upload_status(doc_id, DocStatus.DELETING)
                return
            if status == DocStatus.DELETED:
                relation_count = self._doc_relation_count(doc_id)
                target = DocStatus.SUCCESS if relation_count > 0 else DocStatus.DELETED
                self._set_doc_upload_status(doc_id, target)
                return
            if status in (DocStatus.FAILED, DocStatus.CANCELED):
                relation_count = self._doc_relation_count(doc_id)
                target = DocStatus.SUCCESS if relation_count > 0 else DocStatus.DELETED
                self._set_doc_upload_status(doc_id, target)
                return

    def _prepare_upload_items(self, request: UploadRequest, algo_ids: List[str]) -> List[Dict[str, Any]]:
        prepared_items: List[Dict[str, Any]] = []
        for item in request.items:
            file_path = item.file_path
            if not os.path.exists(file_path):
                raise DocServiceError('E_INVALID_PARAM', f'file not found: {file_path}')
            prepared_items.append({
                'file_path': file_path,
                'metadata': item.metadata,
                'doc_id': gen_doc_id(file_path, doc_id=item.doc_id),
                'filename': os.path.basename(file_path),
            })

        self._validate_unique_doc_ids([item['doc_id'] for item in prepared_items])

        for item in prepared_items:
            if self._has_kb_document(request.kb_id, item['doc_id']):
                for algo_id in algo_ids:
                    self._assert_action_allowed(item['doc_id'], request.kb_id, algo_id, 'upload')
        return prepared_items

    def _prepare_reparse_items(self, request: ReparseRequest, algo_id: str) -> List[Dict[str, Any]]:
        prepared_items = []
        for doc_id in request.doc_ids:
            doc = self._get_doc(doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}')
            self._assert_action_allowed(doc_id, request.kb_id, algo_id, 'reparse')
            prepared_items.append({
                'doc_id': doc_id,
                'file_path': doc.get('path'),
                'metadata': from_json(doc.get('meta')),
            })
        return prepared_items

    def _prepare_delete_items(self, request: DeleteRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        for doc_id in request.doc_ids:
            doc = self._get_doc(doc_id)
            snapshot = self._get_latest_parse_snapshot(doc_id, request.kb_id)
            if doc is None or not self._has_kb_document(request.kb_id, doc_id):
                if snapshot is not None and snapshot.get('status') in (
                    DocStatus.DELETING.value,
                    DocStatus.DELETED.value,
                ):
                    prepared_items.append({
                        'doc_id': doc_id,
                        'action': 'noop',
                        'status': snapshot['status'],
                        'task_id': snapshot.get('current_task_id'),
                    })
                    continue
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}')
            if snapshot is not None and snapshot.get('status') in (DocStatus.DELETING.value, DocStatus.DELETED.value):
                prepared_items.append({
                    'doc_id': doc_id,
                    'action': 'noop',
                    'status': snapshot['status'],
                    'task_id': snapshot.get('current_task_id'),
                })
                continue
            prepared_items.append({'doc_id': doc_id, 'action': 'execute'})
        return prepared_items

    def _prepare_metadata_patch_items(self, request: MetadataPatchRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        all_algo_ids = self._get_kb_algorithms(request.kb_id)
        for item in request.items:
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            for algo_id in all_algo_ids:
                self._assert_action_allowed(item.doc_id, request.kb_id, algo_id, 'metadata')
            merged = from_json(doc.get('meta'))
            merged.update(item.patch)
            prepared_items.append({'doc_id': item.doc_id, 'metadata': merged, 'file_path': doc.get('path')})
        return prepared_items

    def _prepare_transfer_items(self, request: TransferRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        seen_pairs = set()
        seen_targets = set()
        for item in request.items:
            if item.mode not in ('move', 'copy'):
                raise DocServiceError(
                    'E_INVALID_PARAM', f'invalid transfer mode: {item.mode}', {'mode': item.mode}
                )
            item_key = (item.doc_id, item.source_kb_id, item.target_kb_id)
            if item_key in seen_pairs:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    'duplicate transfer item detected',
                    {'doc_id': item.doc_id, 'source_kb_id': item.source_kb_id, 'target_kb_id': item.target_kb_id},
                )
            seen_pairs.add(item_key)
            target_key = (item.target_doc_id, item.target_kb_id)
            if target_key in seen_targets:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    'duplicate transfer target detected',
                    {'doc_id': item.target_doc_id, 'target_kb_id': item.target_kb_id},
                )
            seen_targets.add(target_key)
            # Require both kbs to have identical algo sets
            src_algos = set(self._get_kb_algorithms(item.source_kb_id))
            tgt_algos = set(self._get_kb_algorithms(item.target_kb_id))
            if src_algos != tgt_algos:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    'transfer requires source and target kb to have identical algo bindings',
                    {
                        'source_kb_id': item.source_kb_id,
                        'target_kb_id': item.target_kb_id,
                        'source_algos': sorted(src_algos),
                        'target_algos': sorted(tgt_algos),
                    },
                )
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(item.source_kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            # Check all algos: every parse_state must be SUCCESS
            for algo_id in src_algos:
                self._assert_action_allowed(item.doc_id, item.source_kb_id, algo_id, 'transfer')
                source_snapshot = self._get_parse_snapshot(item.doc_id, item.source_kb_id, algo_id)
                if source_snapshot is None or source_snapshot.get('status') != DocStatus.SUCCESS.value:
                    raise DocServiceError(
                        'E_STATE_CONFLICT',
                        f'doc transfer requires all algo parse states to be SUCCESS: {item.doc_id}',
                        {
                            'doc_id': item.doc_id,
                            'source_kb_id': item.source_kb_id,
                            'algo_id': algo_id,
                            'status': source_snapshot.get('status') if source_snapshot else None,
                        },
                    )
            if self._has_kb_document(item.target_kb_id, item.target_doc_id):
                raise DocServiceError(
                    'E_STATE_CONFLICT',
                    f'doc already exists in target kb: {item.target_doc_id}',
                    {'doc_id': item.target_doc_id, 'target_kb_id': item.target_kb_id},
                )
            source_metadata = from_json(doc.get('meta'))
            target_metadata = merge_transfer_metadata(source_metadata, item.target_metadata)
            target_path = resolve_transfer_target_path(doc.get('path'), item.target_filename, item.target_file_path)
            target_filename = os.path.basename(target_path)
            prepared_items.append({
                'doc_id': item.doc_id,
                'target_doc_id': item.target_doc_id,
                'source_kb_id': item.source_kb_id,
                'target_kb_id': item.target_kb_id,
                'algo_ids': sorted(src_algos),
                'mode': item.mode,
                'file_path': doc.get('path'),
                'target_file_path': target_path,
                'target_filename': target_filename,
                'metadata': target_metadata,
                'source_type': SourceType(doc.get('source_type')),
                'transfer_params': {
                    'mode': 'mv' if item.mode == 'move' else 'cp',
                    'target_doc_id': item.target_doc_id,
                    'target_kb_id': item.target_kb_id,
                },
            })
        return prepared_items

    def upload(self, request: UploadRequest) -> List[Dict[str, Any]]:
        # auto-resolve: use all algos bound to this kb
        algo_ids = self._get_kb_algorithms(request.kb_id)
        if not algo_ids:
            raise DocServiceError(
                'E_STATE_CONFLICT', f'kb has no algorithm binding: {request.kb_id}',
                {'kb_id': request.kb_id},
            )
        prepared_items = self._prepare_upload_items(request, algo_ids)
        source_type = request.source_type or SourceType.API
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            doc_id = item['doc_id']
            file_path = item['file_path']
            metadata = item['metadata']
            doc = self._upsert_doc_and_bind(
                kb_id=request.kb_id,
                doc_id=doc_id,
                filename=item['filename'],
                path=file_path,
                metadata=metadata,
                source_type=source_type,
                upload_status=DocStatus.SUCCESS,
            )
            task_id, snapshot = None, {}
            error_code = error_msg = None
            accepted = True
            try:
                # enqueue a DOC_ADD task for every algo — ng dedup is handled by _wait_and_decide_ng
                current_algo_id = algo_ids[0]
                for algo_id in algo_ids:
                    current_algo_id = algo_id
                    task_id, snapshot = self._enqueue_task(
                        doc_id, request.kb_id, algo_id, TaskType.DOC_ADD,
                        idempotency_key=request.idempotency_key,
                        file_path=file_path,
                        metadata=metadata,
                    )
            except Exception as exc:
                snapshot = self._get_parse_snapshot(doc_id, request.kb_id, current_algo_id) or {}
                doc = self._get_doc(doc_id) or doc
                task_id = snapshot.get('current_task_id')
                error_code = snapshot.get('last_error_code') or type(exc).__name__
                error_msg = snapshot.get('last_error_msg') or str(exc)
                accepted = False
            items.append({
                'doc_id': doc_id,
                'kb_id': request.kb_id,
                'algo_ids': algo_ids,
                'upload_status': doc['upload_status'],
                'parse_status': snapshot.get('status', DocStatus.FAILED.value),
                'task_id': task_id,
                'accepted': accepted,
                'error_code': snapshot.get('last_error_code') if accepted else error_code,
                'error_msg': snapshot.get('last_error_msg') if accepted else error_msg,
            })
        return items

    def add_files(self, request: AddRequest) -> List[Dict[str, Any]]:
        return self.upload(UploadRequest(
            items=request.items,
            kb_id=request.kb_id,
            source_type=request.source_type or SourceType.EXTERNAL,
            idempotency_key=request.idempotency_key,
        ))

    def reparse(self, request: ReparseRequest) -> List[str]:
        self._validate_unique_doc_ids(request.doc_ids, field_name='doc_id')
        # Build the list of algo_ids to process.
        if request.algo_ids is not None:
            algo_ids = request.algo_ids
            for aid in algo_ids:
                self._validate_kb_algorithm(request.kb_id, aid)
        elif request.ng_names is not None:
            # Resolve ng_names → ng_ids, then find which algos contain those ngs
            algo_ids = self._get_algos_for_ng_names(request.kb_id, request.ng_names)
            if not algo_ids:
                raise DocServiceError('E_INVALID_PARAM',
                                      f'no algos found for ng_names {request.ng_names!r} in kb {request.kb_id!r}')
        else:
            algo_ids = self._get_kb_algorithms(request.kb_id)
            if not algo_ids:
                raise DocServiceError('E_STATE_CONFLICT', f'kb has no algorithm binding: {request.kb_id}')
        task_ids = []
        for algo_id in algo_ids:
            task_ids.extend(self._reparse_for_algo(request, algo_id))
        return task_ids

    def _get_algos_for_ng_names(self, kb_id: str, ng_names: List[str]) -> List[str]:
        '''Return algo_ids bound to kb_id that contain at least one of the given ng_names.'''
        # Resolve ng_names → ng_ids via lazyllm_node_group table
        ng_ids_by_name = {}
        with self._db_manager.get_session() as session:
            NodeGroupInfo = self._db_manager.get_table_orm_class('lazyllm_node_group')
            rows = session.query(NodeGroupInfo).filter(NodeGroupInfo.name.in_(ng_names)).all()
            for row in rows:
                ng_ids_by_name[row.name] = row.id
        target_ng_ids = set(ng_ids_by_name.values())
        if not target_ng_ids:
            return []
        # Find algos bound to kb that contain any of the target ng_ids
        all_algo_ids = self._get_kb_algorithms(kb_id)
        matching = []
        for algo_id in all_algo_ids:
            algo_ng_ids = set(self._get_algo_node_group_ids(algo_id))
            if algo_ng_ids & target_ng_ids:
                matching.append(algo_id)
        return matching

    def _reparse_for_algo(self, request: ReparseRequest, algo_id: str) -> List[str]:
        prepared_items = self._prepare_reparse_items(request, algo_id)
        all_ng_ids = self._get_algo_node_group_ids(algo_id)
        if request.ng_names:
            # Resolve ng_names → ng_ids that belong to this algo
            with self._db_manager.get_session() as session:
                NodeGroupInfo = self._db_manager.get_table_orm_class('lazyllm_node_group')
                rows = session.query(NodeGroupInfo).filter(NodeGroupInfo.name.in_(request.ng_names)).all()
                name_to_id = {row.name: row.id for row in rows}
            pending_ng_ids = [name_to_id[n] for n in request.ng_names
                              if name_to_id.get(n) in all_ng_ids]
            effective_reparse_group = json.dumps(pending_ng_ids) if pending_ng_ids else None
        else:
            shared_ng_ids = self._get_shared_ng_ids(request.kb_id, algo_id, all_ng_ids)
            pending_ng_ids = [ng for ng in all_ng_ids if ng not in shared_ng_ids]
            effective_reparse_group = json.dumps(pending_ng_ids) if pending_ng_ids else None
        if not pending_ng_ids:
            LOG.info(f'[reparse] algo={algo_id!r} kb={request.kb_id!r}: all node groups are shared, nothing to reparse')
            return []
        task_ids = []
        for item in prepared_items:
            self._upsert_ng_status_pending(item['doc_id'], request.kb_id, pending_ng_ids,
                                           item['file_path'], force=True)
            task_id, _ = self._enqueue_task(
                item['doc_id'], request.kb_id, algo_id, TaskType.DOC_REPARSE,
                idempotency_key=request.idempotency_key,
                file_path=item['file_path'],
                metadata=item['metadata'],
                reparse_group=effective_reparse_group,
            )
            task_ids.append(task_id)
        return task_ids

    def _check_algos_for_delete(self, doc_id: str, kb_id: str, algo_ids: List[str]):
        # Returns (needs_delete_task, canceled_task_id); raises DocServiceError on conflict.
        needs_delete_task, canceled_task_id = False, None
        for algo_id in algo_ids:
            snap = self._get_parse_snapshot(doc_id, kb_id, algo_id)
            if snap is None:
                continue
            status = snap.get('status')
            if status == DocStatus.WORKING.value:
                raise DocServiceError('E_STATE_CONFLICT', f'cannot delete while algo {algo_id!r} is WORKING',
                                      {'doc_id': doc_id, 'algo_id': algo_id})
            if status == DocStatus.WAITING.value and snap.get('task_type') == TaskType.DOC_ADD.value \
                    and snap.get('current_task_id'):
                cancel_resp = self.cancel_task(snap['current_task_id'])
                if cancel_resp.code != 200:
                    err_data = (cancel_resp.data if isinstance(cancel_resp.data, dict)
                                else {'task_id': snap['current_task_id']})
                    raise DocServiceError('E_STATE_CONFLICT', cancel_resp.msg, err_data)
                canceled_task_id = snap['current_task_id']
            elif status not in (DocStatus.DELETED.value, DocStatus.CANCELED.value):
                needs_delete_task = True
        return needs_delete_task, canceled_task_id

    def delete(self, request: DeleteRequest) -> List[Dict[str, Any]]:
        # all algo_ids bound to this kb — each needs a delete task so their stores are cleaned up
        all_algo_ids = self._get_kb_algorithms(request.kb_id)
        self._validate_unique_doc_ids(request.doc_ids, field_name='doc_id')
        prepared_items = self._prepare_delete_items(request)
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            doc_id = item['doc_id']
            if item.get('action') == 'noop':
                items.append({'doc_id': doc_id, 'accepted': True, 'task_id': item.get('task_id'),
                              'status': item['status'], 'error_code': None})
                continue
            needs_delete_task, canceled_task_id = self._check_algos_for_delete(doc_id, request.kb_id, all_algo_ids)
            if not needs_delete_task and canceled_task_id is not None:
                self._purge_deleted_kb_doc_data(request.kb_id, doc_id, remove_relation=True)
                items.append({
                    'doc_id': doc_id,
                    'accepted': True,
                    'task_id': canceled_task_id,
                    'status': DocStatus.CANCELED.value,
                    'error_code': None,
                })
                continue
            with self._db_manager.get_session() as session:
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
                row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
                relation_count = session.query(Rel).filter(Rel.doc_id == doc_id).count()
                if relation_count <= 1:
                    row.upload_status = DocStatus.DELETING.value
                row.updated_at = datetime.now()
                session.add(row)

            # submit a delete task for every algo bound to this kb so all stores are cleaned up
            primary_task_id = None
            primary_snapshot = None
            for algo_id in all_algo_ids:
                algo_idem_key = (
                    f'{request.idempotency_key}:{algo_id}' if request.idempotency_key else None
                )
                task_id, snap = self._enqueue_task(
                    doc_id, request.kb_id, algo_id, TaskType.DOC_DELETE,
                    idempotency_key=algo_idem_key,
                )
                if primary_task_id is None:
                    primary_task_id = task_id
                    primary_snapshot = snap
            items.append({
                'doc_id': doc_id,
                'accepted': True,
                'task_id': primary_task_id,
                'status': primary_snapshot['status'] if primary_snapshot else DocStatus.DELETING.value,
                'error_code': None,
            })
        return items

    def transfer(self, request: TransferRequest) -> List[Dict[str, Any]]:
        prepared_items = self._prepare_transfer_items(request)
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            primary_task_id = None
            primary_snapshot: Dict = {}
            error_code = None
            error_msg = None
            accepted = True
            try:
                self._upsert_doc_and_bind(
                    kb_id=item['target_kb_id'],
                    doc_id=item['target_doc_id'],
                    filename=item['target_filename'],
                    path=item['target_file_path'],
                    metadata=item['metadata'],
                    source_type=item['source_type'],
                    upload_status=DocStatus.SUCCESS,
                    allowed_path_doc_ids={item['doc_id']},
                )
                for algo_id in item['algo_ids']:
                    task_id, snapshot = self._enqueue_task(
                        item['target_doc_id'], item['target_kb_id'], algo_id, TaskType.DOC_TRANSFER,
                        idempotency_key=request.idempotency_key if primary_task_id is None else None,
                        file_path=item['file_path'],
                        metadata=item['metadata'],
                        parser_kb_id=item['source_kb_id'],
                        transfer_params=item['transfer_params'],
                        extra_message={
                            'source_doc_id': item['doc_id'],
                            'source_kb_id': item['source_kb_id'],
                            'target_kb_id': item['target_kb_id'],
                            'target_doc_id': item['target_doc_id'],
                            'algo_id': algo_id,
                            'mode': item['mode'],
                        },
                        parser_doc_id=item['doc_id'],
                    )
                    if primary_task_id is None:
                        primary_task_id = task_id
                        primary_snapshot = snapshot
            except Exception as exc:
                primary_snapshot = self._get_parse_snapshot(
                    item['target_doc_id'], item['target_kb_id'], item['algo_ids'][0]
                ) or {}
                primary_task_id = primary_task_id or primary_snapshot.get('current_task_id')
                error_code = primary_snapshot.get('last_error_code')
                if not error_code:
                    error_code = exc.biz_code if isinstance(exc, DocServiceError) else type(exc).__name__
                error_msg = primary_snapshot.get('last_error_msg') or (
                    exc.msg if isinstance(exc, DocServiceError) else str(exc)
                )
                accepted = False
            items.append({
                'doc_id': item['doc_id'],
                'target_doc_id': item['target_doc_id'],
                'task_id': primary_task_id,
                'source_kb_id': item['source_kb_id'],
                'target_kb_id': item['target_kb_id'],
                'mode': item['mode'],
                'target_file_path': item['target_file_path'],
                'status': primary_snapshot.get('status', DocStatus.FAILED.value),
                'accepted': accepted,
                'error_code': error_code,
                'error_msg': error_msg,
            })
        return items

    def patch_metadata(self, request: MetadataPatchRequest):
        updated = []
        failed = []
        self._validate_unique_doc_ids([item.doc_id for item in request.items], field_name='doc_id')
        all_algo_ids = self._get_kb_algorithms(request.kb_id)
        prepared_items = self._prepare_metadata_patch_items(request)
        for item in prepared_items:
            with self._db_manager.get_session() as session:
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                row = session.query(Doc).filter(Doc.doc_id == item['doc_id']).first()
                row.meta = to_json(item['metadata'])
                row.updated_at = datetime.now()
                session.add(row)
            primary_task_id = None
            for algo_id in all_algo_ids:
                task_id, _ = self._enqueue_task(
                    item['doc_id'], request.kb_id, algo_id, TaskType.DOC_UPDATE_META,
                    idempotency_key=request.idempotency_key if primary_task_id is None else None,
                    file_path=item['file_path'],
                    metadata=item['metadata'],
                )
                if primary_task_id is None:
                    primary_task_id = task_id
            updated.append({'doc_id': item['doc_id'], 'task_id': primary_task_id})
        return {
            'updated_count': len(updated),
            'doc_ids': [u['doc_id'] for u in updated],
            'failed_items': failed,
            'items': updated,
        }

    def _sync_doc_upload_status(self, doc_id: str, session=None):
        with self._db_manager.get_session(session) as sess:
            target = (
                DocStatus.SUCCESS
                if self._doc_relation_count(doc_id, session=sess) > 0
                else DocStatus.DELETED
            )
            self._set_doc_upload_status(doc_id, target, session=sess)

    @staticmethod
    def _build_snapshot_update(snapshot: Optional[Dict[str, Any]], **overrides):
        snapshot = snapshot or {}
        data = {
            'task_type': TaskType(snapshot['task_type']) if snapshot.get('task_type') else None,
            'current_task_id': snapshot.get('current_task_id'),
            'idempotency_key': snapshot.get('idempotency_key'),
            'priority': snapshot.get('priority', 0),
            'task_score': snapshot.get('task_score'),
            'retry_count': snapshot.get('retry_count', 0),
            'max_retry': snapshot.get('max_retry', 3),
            'lease_owner': snapshot.get('lease_owner'),
            'lease_until': snapshot.get('lease_until'),
            'error_code': snapshot.get('last_error_code'),
            'error_msg': snapshot.get('last_error_msg'),
            'failed_stage': snapshot.get('failed_stage'),
            'queued_at': snapshot.get('queued_at'),
            'started_at': snapshot.get('started_at'),
            'finished_at': snapshot.get('finished_at'),
        }
        data.update(overrides)
        return data

    def _resolve_callback_task(self, callback: TaskCallbackRequest):
        task = self._get_task_record(callback.task_id)
        if task is not None:
            return task
        payload = callback.payload or {}
        # Fallback: reconstruct minimal task data from payload (no algo_id required)
        required_fields = {'task_type', 'doc_id', 'kb_id'}
        if required_fields.issubset(payload.keys()):
            return {
                'task_id': callback.task_id,
                'task_type': payload['task_type'],
                'doc_id': payload['doc_id'],
                'kb_id': payload['kb_id'],
                'algo_id': payload.get('algo_id'),  # may be None for new-style callbacks
            }
        return None

    def on_task_callback(self, callback: TaskCallbackRequest):  # noqa: C901
        if not self._record_callback(callback.callback_id, callback.task_id):
            return {'ack': True, 'deduped': True, 'ignored_reason': None}
        try:
            task_data = self._resolve_callback_task(callback)
            if task_data is None:
                return {'ack': True, 'ignored_reason': 'task_not_found'}
            doc_id = task_data['doc_id']
            kb_id = task_data['kb_id']
            algo_id = task_data['algo_id']
            task_type = TaskType(task_data['task_type'])
            snapshot = self._get_parse_snapshot(doc_id, kb_id, algo_id)
            if snapshot and snapshot.get('current_task_id') and snapshot['current_task_id'] != callback.task_id:
                return {'ack': True, 'deduped': False, 'ignored_reason': 'stale_task_callback'}
            if task_data.get('status') == DocStatus.CANCELED.value and callback.status != DocStatus.CANCELED:
                return {'ack': True, 'deduped': False, 'ignored_reason': 'canceled_task_callback'}

            if callback.event_type == CallbackEventType.START:
                self._update_task_record(
                    callback.task_id,
                    status=DocStatus.WORKING.value,
                    started_at=datetime.now(),
                    finished_at=None,
                    error_code=None,
                    error_msg=None,
                )
                start_status = DocStatus.DELETING if task_type == TaskType.DOC_DELETE else DocStatus.WORKING
                self._upsert_parse_snapshot(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    algo_id=algo_id,
                    status=start_status,
                    **self._build_snapshot_update(
                        snapshot,
                        task_type=task_type,
                        current_task_id=callback.task_id,
                        started_at=datetime.now(),
                        finished_at=None,
                        error_code=None,
                        error_msg=None,
                        failed_stage=None,
                    ),
                )
                if task_type == TaskType.DOC_ADD:
                    self._apply_doc_upload_status(doc_id, task_type, DocStatus.WORKING)
                elif task_type == TaskType.DOC_DELETE:
                    self._apply_doc_upload_status(doc_id, task_type, DocStatus.DELETING)
                return {'ack': True, 'deduped': False, 'ignored_reason': None}

            final_status = callback.status
            if task_type == TaskType.DOC_DELETE and final_status == DocStatus.SUCCESS:
                final_status = DocStatus.DELETED
            failed_stage = None
            if final_status == DocStatus.FAILED:
                failed_stage = 'DELETE' if task_type == TaskType.DOC_DELETE else 'PARSE'
            task_message = task_data.get('message') or {}
            cleanup_policy = task_message.get('cleanup_policy')

            if (
                task_type == TaskType.DOC_TRANSFER
                and final_status == DocStatus.SUCCESS
                and task_message.get('mode') == 'move'
            ):
                source_kb_id = task_message.get('source_kb_id')
                source_doc_id = task_message.get('source_doc_id') or doc_id
                if source_kb_id and source_kb_id != kb_id:
                    with self._db_manager.get_session() as session:
                        self._remove_kb_document(source_kb_id, source_doc_id, session=session)
                        self._delete_parse_snapshots(source_doc_id, source_kb_id, session=session)
                        self._sync_doc_upload_status(source_doc_id, session=session)

            self._update_task_record(
                callback.task_id,
                status=final_status.value,
                error_code=callback.error_code,
                error_msg=callback.error_msg,
                finished_at=datetime.now(),
            )

            self._upsert_parse_snapshot(
                doc_id=doc_id,
                kb_id=kb_id,
                algo_id=algo_id,
                status=final_status,
                **self._build_snapshot_update(
                    snapshot,
                    task_type=task_type,
                    current_task_id=callback.task_id,
                    error_code=callback.error_code,
                    error_msg=callback.error_msg,
                    failed_stage=failed_stage,
                    finished_at=datetime.now(),
                ),
            )

            upload_status_handled = False
            unbind_algo = task_message.get('unbind_algo', False)
            if task_type == TaskType.DOC_DELETE and final_status == DocStatus.DELETED:
                # Only purge kb_documents + ng_status when ALL algo delete tasks for this doc are done.
                all_deleted = self._all_algo_snapshots_deleted(doc_id, kb_id)
                if all_deleted:
                    if cleanup_policy == 'purge':
                        self._purge_deleted_kb_doc_data(kb_id, doc_id, remove_relation=True)
                        upload_status_handled = True
                    else:
                        self._remove_kb_document(kb_id, doc_id)
                        self._delete_ng_status(doc_id, kb_id)
                elif unbind_algo:
                    # Not all algos deleted yet — only clean up ng_status rows for ngs
                    # that are exclusive to the unbound algo (shared ngs must be preserved).
                    exclusive_ng_ids = task_message.get('exclusive_ng_ids') or []
                    self._delete_ng_status_for_groups(doc_id, kb_id, exclusive_ng_ids)
                    # Remove the parse_state row for this algo so it no longer affects
                    # _all_algo_snapshots_deleted checks for other algos.
                    self._delete_parse_snapshot_for_algo(doc_id, kb_id, algo_id)
                # If this was an unbind_algo task, check if all docs for this algo are now DELETED.
                # When all_deleted is True, purge already cleaned up parse_state rows, so count=0.
                # When all_deleted is False, we deleted this algo's parse_state above; count remaining.
                if unbind_algo:
                    if not all_deleted:
                        remaining = self._count_non_deleted_snapshots_for_algo(kb_id, algo_id)
                    else:
                        remaining = 0
                    if remaining == 0:
                        self._remove_kb_algo_binding(kb_id, algo_id)
            if not upload_status_handled:
                self._apply_doc_upload_status(doc_id, task_type, final_status)
            if task_type == TaskType.DOC_DELETE and final_status == DocStatus.DELETED and cleanup_policy == 'purge':
                self._finalize_kb_deletion_if_empty(kb_id)

            return {'ack': True, 'deduped': False, 'ignored_reason': None}
        except Exception:
            self._forget_callback_record(callback.callback_id, callback.task_id)
            raise

    def list_docs(self, status: Optional[List[str]] = None, kb_id: Optional[str] = None,
                  keyword: Optional[str] = None,
                  include_deleted_or_canceled: bool = True, page: int = 1, page_size: int = 20):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])

            rn_col = sqlalchemy.func.row_number().over(
                partition_by=(State.doc_id, State.kb_id),
                order_by=(State.updated_at.desc(), State.created_at.desc())).label('_rn')
            ranked_sq = session.query(State, rn_col).subquery('ranked_snapshot')
            latest_sq = session.query(ranked_sq).filter(ranked_sq.c._rn == 1).subquery('latest_snapshot')
            SnapshotAlias = aliased(State, latest_sq)
            snapshot_join_cond = sqlalchemy.and_(
                SnapshotAlias.doc_id == Doc.doc_id, SnapshotAlias.kb_id == Rel.kb_id)

            query = (session.query(Doc, Rel, SnapshotAlias).join(Rel, Doc.doc_id == Rel.doc_id)
                     .outerjoin(SnapshotAlias, snapshot_join_cond))

            if kb_id:
                query = query.filter(Rel.kb_id == kb_id)
            if keyword:
                like_expr = f'%{keyword}%'
                query = query.filter((Doc.filename.like(like_expr)) | (Doc.path.like(like_expr)))
            if not include_deleted_or_canceled:
                query = query.filter(~Doc.upload_status.in_([DocStatus.DELETED.value, DocStatus.CANCELED.value]))
            if status:
                query = query.filter(SnapshotAlias.status.in_(status))

            query = query.order_by(Rel.updated_at.desc(), Doc.updated_at.desc())
            result = self._db_manager.paginate(query, page=page, page_size=page_size)

            items = []
            for doc_row, rel_row, snap_row in result['items']:
                doc = _orm_to_dict(doc_row)
                doc['metadata'] = from_json(doc.get('meta'))
                items.append({'doc': doc, 'relation': _orm_to_dict(rel_row),
                              'snapshot': _orm_to_dict(snap_row) if snap_row is not None else None})
            result['items'] = items
            return result

    def get_doc_detail(self, doc_id: str):
        doc = self._get_doc(doc_id)
        if not doc:
            raise DocServiceError('E_NOT_FOUND', f'doc not found: {doc_id}', {'doc_id': doc_id})

        doc['metadata'] = from_json(doc.get('meta'))
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            rel_rows = session.query(Rel).filter(Rel.doc_id == doc_id).order_by(Rel.updated_at.desc()).all()
        matched_items = []
        for rel_row in rel_rows:
            relation = _orm_to_dict(rel_row)
            matched_items.append({
                'relation': relation,
                'snapshot': self._get_latest_parse_snapshot(doc_id, relation['kb_id']),
            })
        relation = matched_items[0].get('relation') if matched_items else None
        snapshot = matched_items[0].get('snapshot') if matched_items else None
        latest_task = None
        if snapshot and snapshot.get('current_task_id'):
            latest_task = self._get_task_record(snapshot['current_task_id'])
        return {
            'doc': doc,
            'relation': relation,
            'snapshot': snapshot,
            'latest_task': latest_task,
            'relations': [item.get('relation') for item in matched_items],
            'snapshots': [item.get('snapshot') for item in matched_items if item.get('snapshot') is not None],
        }

    def list_tasks(self, status: Optional[List[str]], page: int, page_size: int):
        parser_list_tasks = getattr(self._parser_client, 'list_tasks', None)
        if callable(parser_list_tasks):
            try:
                return parser_list_tasks(status=status, page=page, page_size=page_size)
            except Exception as exc:
                LOG.warning(f'[DocService] Fallback to local task list: {exc}')
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            query = session.query(Task)
            if status:
                query = query.filter(Task.status.in_(status))
            query = query.order_by(Task.created_at.desc())
            result = self._db_manager.paginate(query, page=page, page_size=page_size)
            items = []
            for row in result['items']:
                task = _orm_to_dict(row)
                task['message'] = from_json(task.get('message'))
                items.append(task)
            result['items'] = items
        return BaseResponse(code=200, msg='success', data=result)

    def get_task(self, task_id: str):
        parser_get_task = getattr(self._parser_client, 'get_task', None)
        if callable(parser_get_task):
            try:
                return parser_get_task(task_id)
            except Exception as exc:
                LOG.warning(f'[DocService] Fallback to local task query: {exc}')
        task = self._get_task_record(task_id)
        if task is None:
            return BaseResponse(code=404, msg='task not found', data=None)
        return BaseResponse(code=200, msg='success', data=task)

    def get_tasks_batch(self, task_ids: List[str]):
        items = []
        for task_id in task_ids:
            resp = self.get_task(task_id)
            if resp.code == 200 and resp.data is not None:
                items.append(resp.data)
        return {'items': items}

    def cancel_task(self, task_id: str):
        task = self._get_task_record(task_id)
        if task is None:
            return BaseResponse(code=404, msg='task not found', data={'task_id': task_id, 'cancel_status': False})
        if task.get('status') != DocStatus.WAITING.value:
            return BaseResponse(
                code=409,
                msg='task cannot be canceled',
                data={'task_id': task_id, 'cancel_status': False, 'status': task.get('status')},
            )
        resp = self._parser_client.cancel_task(task_id)
        if resp.code != 200:
            return resp
        resp_data = resp.data or {}
        if not resp_data.get('cancel_status'):
            return BaseResponse(
                code=409,
                msg=resp_data.get('message', 'task cannot be canceled'),
                data={'task_id': task_id, 'cancel_status': False, 'status': task.get('status')},
            )
        self.on_task_callback(TaskCallbackRequest(
            task_id=task_id,
            event_type=CallbackEventType.FINISH,
            status=DocStatus.CANCELED,
        ))
        return BaseResponse(
            code=200,
            msg='success',
            data={'task_id': task_id, 'cancel_status': True, 'status': DocStatus.CANCELED.value},
        )

    def list_algorithms(self):
        resp = self._parser_client.list_algorithms()
        if resp.code != 200:
            raise DocServiceError('E_UPSTREAM_ERROR', resp.msg, {'upstream_status': resp.code})
        return resp.data

    def get_algo_groups(self, algo_id: str):
        resp = self._parser_client.get_algorithm_groups(algo_id)
        if resp.code == 404:
            raise DocServiceError('E_NOT_FOUND', 'algo not found', {'algo_id': algo_id})
        if resp.code != 200:
            raise DocServiceError('E_UPSTREAM_ERROR', resp.msg, {'algo_id': algo_id, 'upstream_status': resp.code})
        return resp.data

    def list_algorithms_compat(self):
        items = self.list_algorithms()
        return {'items': items}

    def get_algorithm_info(self, algo_id: str):
        algorithms = self.list_algorithms()
        for item in algorithms:
            if item['algo_id'] == algo_id:
                data = dict(item)
                data['groups'] = self.get_algo_groups(algo_id)
                return data
        raise DocServiceError('E_NOT_FOUND', f'algo not found: {algo_id}')

    def _find_algo_for_group(self, kb_id: str, group: str) -> str:
        '''Find the first algo bound to kb_id that contains the given node group name.'''
        for algo_id in self._get_kb_algorithms(kb_id):
            try:
                groups = self.get_algo_groups(algo_id)
                if any(item.get('name') == group for item in (groups or [])):
                    return algo_id
            except DocServiceError:
                continue
        raise DocServiceError(
            'E_INVALID_PARAM',
            f'group {group!r} not found in any algo bound to kb {kb_id}',
            {'kb_id': kb_id, 'group': group},
        )

    def list_chunks(
        self,
        kb_id: str,
        doc_id: str,
        group: str,
        algo_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        offset: Optional[int] = None,
    ):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required', {'kb_id': kb_id})
        if not doc_id:
            raise DocServiceError('E_INVALID_PARAM', 'doc_id is required', {'doc_id': doc_id})
        if not group:
            raise DocServiceError('E_INVALID_PARAM', 'group is required', {'group': group})
        doc = self._get_doc(doc_id)
        if doc is None or not self._has_kb_document(kb_id, doc_id):
            raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}', {'kb_id': kb_id, 'doc_id': doc_id})
        page = max(page, 1)
        page_size = max(page_size, 1)
        offset = (page - 1) * page_size if offset is None else max(offset, 0)
        # Auto-detect algo if not provided; _find_algo_for_group validates group existence
        if not algo_id:
            algo_id = self._find_algo_for_group(kb_id, group)
        else:
            self._validate_kb_algorithm(kb_id, algo_id)
        resp = self._parser_client.list_doc_chunks(
            algo_id=algo_id, kb_id=kb_id, doc_id=doc_id, group=group,
            offset=offset, page_size=page_size,
        )
        if resp.code == 404:
            raise DocServiceError('E_NOT_FOUND', resp.msg, {'kb_id': kb_id, 'doc_id': doc_id, 'group': group})
        if resp.code == 400:
            raise DocServiceError('E_INVALID_PARAM', resp.msg, {'kb_id': kb_id, 'doc_id': doc_id, 'group': group})
        if resp.code != 200:
            raise DocServiceError(
                'E_UPSTREAM_ERROR', resp.msg,
                {'kb_id': kb_id, 'doc_id': doc_id, 'group': group, 'upstream_status': resp.code},
            )
        data = dict(resp.data or {})
        data.update({'page': page, 'page_size': page_size, 'offset': offset})
        data.setdefault('items', [])
        data.setdefault('total', 0)
        return data

    def health(self):
        parser_ok = False
        try:
            parser_ok = self._parser_client.health().code == 200
        except Exception as exc:
            LOG.warning(f'[DocService] Parser health check failed: {exc}')
        return {
            'status': 'ok',
            'version': 'v1',
            'deps': {
                'sql': True,
                'parser': parser_ok,
            },
        }

    def _batch_get_kb_algorithms(self, kb_ids: List[str]) -> Dict[str, List[str]]:
        if not kb_ids:
            return {}
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            rows = session.query(Rel).filter(Rel.kb_id.in_(kb_ids)).all()
        result: Dict[str, List[str]] = {kb_id: [] for kb_id in kb_ids}
        for row in rows:
            result[row.kb_id].append(row.algo_id)
        return result

    def list_kbs(self, page: int = 1, page_size: int = 20, keyword: Optional[str] = None,
                 status: Optional[List[str]] = None, owner_id: Optional[str] = None):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            query = session.query(Kb)
            query = query.filter(Kb.kb_id != '__default__')
            if keyword:
                like_expr = f'%{keyword}%'
                query = query.filter(sqlalchemy.or_(
                    Kb.kb_id.like(like_expr), Kb.display_name.like(like_expr), Kb.description.like(like_expr)))
            if status:
                query = query.filter(Kb.status.in_(status))
            if owner_id:
                query = query.filter(Kb.owner_id == owner_id)
            query = query.order_by(Kb.updated_at.desc(), Kb.created_at.desc())
            result = self._db_manager.paginate(query, page=page, page_size=page_size)
            kb_ids = [kb_row.kb_id for kb_row in result['items']]
            algo_map = self._batch_get_kb_algorithms(kb_ids)
            result['items'] = [
                self._build_kb_data(kb_row, algo_map.get(kb_row.kb_id, []))
                for kb_row in result['items']
            ]
            return result

    def get_kb(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if row is None:
                raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
        return self._build_kb_data(row, self._get_kb_algorithms(kb_id))

    def batch_get_kbs(self, kb_ids: List[str]):
        if not kb_ids:
            raise DocServiceError('E_INVALID_PARAM', 'kb_ids is required', {'kb_ids': kb_ids})
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            kb_rows = {row.kb_id: row for row in session.query(Kb).filter(Kb.kb_id.in_(kb_ids)).all()}
        algo_map = self._batch_get_kb_algorithms(list(kb_rows.keys()))
        items = []
        missing_kb_ids = []
        for kb_id in kb_ids:
            if kb_id in kb_rows:
                items.append(self._build_kb_data(kb_rows[kb_id], algo_map.get(kb_id, [])))
            else:
                missing_kb_ids.append(kb_id)
        return {'items': items, 'missing_kb_ids': missing_kb_ids}

    def create_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                  owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                  algo_id: str = '__default__'):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        if self._get_kb(kb_id) is not None:
            raise DocServiceError('E_STATE_CONFLICT', f'kb already exists: {kb_id}', {'kb_id': kb_id})
        self._ensure_algorithm_exists(algo_id)
        self._ensure_kb(
            kb_id, display_name=display_name, description=description, owner_id=owner_id, meta=meta,
        )
        self._ensure_kb_algorithm(kb_id, algo_id)
        return self.get_kb(kb_id)

    def update_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                  owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                  algo_id: Optional[str] = None, explicit_fields: Optional[Set[str]] = None):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        kb = self._get_kb(kb_id)
        if kb is None:
            raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
        explicit_fields = explicit_fields or set()
        if 'algo_id' in explicit_fields:
            if algo_id is None:
                raise DocServiceError('E_INVALID_PARAM', 'algo_id cannot be null', {'kb_id': kb_id})
            self._ensure_algorithm_exists(algo_id)
        if algo_id is not None:
            self._ensure_kb_algorithm(kb_id, algo_id)
        self._ensure_kb(
            kb_id, display_name=display_name, description=description, owner_id=owner_id, meta=meta,
            update_fields=explicit_fields & {'display_name', 'description', 'owner_id', 'meta'},
        )
        return self.get_kb(kb_id)

    def unbind_algo(self, kb_id: str, algo_id: str, dry_run: bool = False):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        if not algo_id:
            raise DocServiceError('E_INVALID_PARAM', 'algo_id is required')
        self._validate_kb_algorithm(kb_id, algo_id)
        algo_ids = self._get_kb_algorithms(kb_id)
        if len(algo_ids) <= 1:
            raise DocServiceError(
                'E_STATE_CONFLICT',
                f'cannot unbind the last algo from kb {kb_id}; '
                f'use delete_kb to remove the entire knowledge base instead',
                {'kb_id': kb_id, 'algo_id': algo_id},
            )
        # Compute ngs exclusive to this algo (not shared with any other algo in the same kb).
        all_ng_ids = self._get_algo_node_group_ids(algo_id)
        shared_ng_ids = self._get_shared_ng_ids(kb_id, algo_id, all_ng_ids)
        exclusive_ng_ids = [ng for ng in all_ng_ids if ng not in shared_ng_ids]
        doc_ids = self._list_kb_doc_ids(kb_id)
        affected_doc_ids = [
            doc_id for doc_id in doc_ids
            if self._get_parse_snapshot(doc_id, kb_id, algo_id) is not None
            and self._get_parse_snapshot(doc_id, kb_id, algo_id).get('status') not in (
                DocStatus.DELETED.value, DocStatus.DELETING.value
            )
        ]
        if dry_run:
            return {'task_ids': [], 'affected_doc_ids': affected_doc_ids, 'dry_run': True}
        task_ids = []
        for doc_id in affected_doc_ids:
            snapshot = self._get_parse_snapshot(doc_id, kb_id, algo_id)
            if snapshot is None or snapshot.get('status') in (DocStatus.DELETED.value, DocStatus.DELETING.value):
                continue
            status = snapshot.get('status')
            task_type = snapshot.get('task_type')
            if status == DocStatus.WORKING.value:
                raise DocServiceError(
                    'E_STATE_CONFLICT',
                    f'cannot unbind algo while doc {doc_id} task is {status}',
                    {'kb_id': kb_id, 'doc_id': doc_id, 'algo_id': algo_id, 'status': status},
                )
            is_waiting_add = (status == DocStatus.WAITING.value and task_type == TaskType.DOC_ADD.value
                              and snapshot.get('current_task_id'))
            if is_waiting_add:
                cancel_resp = self.cancel_task(snapshot['current_task_id'])
                if cancel_resp.code != 200:
                    err_data = (cancel_resp.data if isinstance(cancel_resp.data, dict)
                                else {'task_id': snapshot['current_task_id']})
                    raise DocServiceError('E_STATE_CONFLICT', cancel_resp.msg, err_data)
            task_id, _ = self._enqueue_task(
                doc_id, kb_id, algo_id, TaskType.DOC_DELETE,
                extra_message={'unbind_algo': True, 'exclusive_ng_ids': exclusive_ng_ids},
            )
            task_ids.append(task_id)
        # If no docs to clean up, remove the binding immediately
        if not task_ids:
            self._remove_kb_algo_binding(kb_id, algo_id)
        return {'task_ids': task_ids, 'affected_doc_ids': affected_doc_ids, 'dry_run': False}

    def _remove_kb_algo_binding(self, kb_id: str, algo_id: str):
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            session.query(Rel).filter(Rel.kb_id == kb_id, Rel.algo_id == algo_id).delete()

    def delete_kb(self, kb_id: str):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        prepared = self._prepare_kb_delete_items(kb_id)
        task_ids = []
        for item in prepared['items']:
            if item['action'] == 'purge_local':
                self._purge_deleted_kb_doc_data(kb_id, item['doc_id'], remove_relation=True)
                continue
            if item['action'] == 'reuse_delete_task':
                self._mark_task_cleanup_policy(item['task_id'], 'purge')
                task_ids.append(item['task_id'])
                continue
            if item['action'] == 'enqueue_delete':
                task_id, _ = self._enqueue_task(
                    item['doc_id'], kb_id, item['algo_id'], TaskType.DOC_DELETE, cleanup_policy='purge'
                )
                task_ids.append(task_id)
                continue
            raise RuntimeError(f'unsupported kb delete action: {item["action"]}')
        new_status = KBStatus.DELETING.value if task_ids else KBStatus.DELETED.value
        if task_ids:
            with self._db_manager.get_session() as session:
                Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
                kb_row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
                if kb_row is not None:
                    kb_row.status = new_status
                    kb_row.updated_at = datetime.now()
                    session.add(kb_row)
        else:
            self._finalize_kb_deletion_if_empty(kb_id)
        return {'kb_id': kb_id, 'status': new_status, 'task_ids': task_ids}

    def delete_kbs(self, kb_ids: List[str]):
        if not kb_ids:
            raise DocServiceError('E_INVALID_PARAM', 'kb_ids is required', {'kb_ids': kb_ids})
        items = []
        for kb_id in kb_ids:
            items.append(self.delete_kb(kb_id))
        return {'items': items}
