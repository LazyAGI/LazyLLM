from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import requests
import sqlalchemy
from sqlalchemy.exc import IntegrityError

from lazyllm.thirdparty import fastapi

from ..utils import BaseResponse, _get_default_db_config, _orm_to_dict
from ...sql import SqlManager
from .base import (
    AddRequest,
    CallbackEventType,
    CALLBACK_RECORDS_TABLE_INFO,
    DeleteRequest,
    DOC_SERVICE_TASKS_TABLE_INFO,
    DocServiceError,
    DOCUMENTS_TABLE_INFO,
    IDEMPOTENCY_RECORDS_TABLE_INFO,
    KB_ALGORITHM_TABLE_INFO,
    KB_DOCUMENTS_TABLE_INFO,
    KBStatus,
    KBS_TABLE_INFO,
    MetadataPatchRequest,
    PARSE_STATE_TABLE_INFO,
    ReparseRequest,
    SourceType,
    TaskCallbackRequest,
    TaskType,
    TransferRequest,
    UploadRequest,
    DocStatus,
    now_ts,
)
from ..parsing_service.base import (
    AddDocRequest as ParsingAddDocRequest,
    CancelTaskRequest as ParsingCancelTaskRequest,
    DeleteDocRequest as ParsingDeleteDocRequest,
    FileInfo as ParsingFileInfo,
    TransferParams as ParsingTransferParams,
    UpdateMetaRequest as ParsingUpdateMetaRequest,
)


def _to_json(data: Optional[Dict[str, Any]]) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def _from_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def gen_doc_id(file_path: str, doc_id: Optional[str] = None) -> str:
    if doc_id:
        return doc_id
    return hashlib.sha256(file_path.encode()).hexdigest()


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(data: Any) -> str:
    return hashlib.sha256(_stable_json(data).encode()).hexdigest()


def _sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _ParserClient:
    def __init__(self, parser_url: str):
        parser_url = parser_url.rstrip('/')
        if parser_url.endswith('/_call') or parser_url.endswith('/generate'):
            parser_url = parser_url.rsplit('/', 1)[0]
        self._parser_url = parser_url

    def _post(self, path: str, payload: Dict[str, Any]):
        url = f'{self._parser_url}{path}'
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code >= 400:
            raise RuntimeError(f'parser http error: {resp.status_code} {resp.text}')
        return resp.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None):
        url = f'{self._parser_url}{path}'
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code >= 400:
            raise RuntimeError(f'parser http error: {resp.status_code} {resp.text}')
        return resp.json()

    def _delete(self, path: str, payload: Optional[Dict[str, Any]] = None):
        url = f'{self._parser_url}{path}'
        resp = requests.delete(url, json=payload, timeout=8)
        if resp.status_code >= 400:
            raise RuntimeError(f'parser http error: {resp.status_code} {resp.text}')
        return resp.json()

    def _get_with_fallback(self, paths: List[str], params: Optional[Dict[str, Any]] = None):
        last_error = None
        for path in paths:
            try:
                return self._get(path, params=params)
            except RuntimeError as exc:
                last_error = exc
                if '404' not in str(exc):
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError('parser http error: no path provided')

    def add_doc(self, task_id: str, algo_id: str, kb_id: str, doc_id: str, file_path: str,
                metadata: Optional[Dict[str, Any]] = None, reparse_group: Optional[str] = None,
                callback_url: Optional[str] = None, transfer_params: Optional[Dict[str, Any]] = None):
        req = ParsingAddDocRequest(
            task_id=task_id,
            algo_id=algo_id,
            kb_id=kb_id,
            callback_url=callback_url,
            feedback_url=callback_url,
            file_infos=[ParsingFileInfo(
                file_path=file_path,
                doc_id=doc_id,
                metadata=metadata or {},
                reparse_group=reparse_group,
                transfer_params=(
                    ParsingTransferParams.model_validate(transfer_params)
                    if transfer_params is not None else None
                ),
            )],
        )
        data = self._post('/doc/add', req.model_dump(mode='json'))
        return BaseResponse.model_validate(data)

    def update_meta(self, task_id: str, algo_id: str, kb_id: str, doc_id: str,
                    metadata: Optional[Dict[str, Any]] = None, file_path: Optional[str] = None,
                    callback_url: Optional[str] = None):
        req = ParsingUpdateMetaRequest(
            task_id=task_id,
            algo_id=algo_id,
            kb_id=kb_id,
            callback_url=callback_url,
            feedback_url=callback_url,
            file_infos=[ParsingFileInfo(file_path=file_path, doc_id=doc_id, metadata=metadata or {})],
        )
        data = self._post('/doc/meta/update', req.model_dump(mode='json'))
        return BaseResponse.model_validate(data)

    def delete_doc(self, task_id: str, algo_id: str, kb_id: str, doc_id: str,
                   callback_url: Optional[str] = None):
        req = ParsingDeleteDocRequest(
            task_id=task_id,
            algo_id=algo_id,
            kb_id=kb_id,
            doc_ids=[doc_id],
            callback_url=callback_url,
            feedback_url=callback_url,
        )
        data = self._delete('/doc/delete', req.model_dump(mode='json'))
        return BaseResponse.model_validate(data)

    def cancel_task(self, task_id: str):
        req = ParsingCancelTaskRequest(task_id=task_id)
        data = self._post('/doc/cancel', req.model_dump(mode='json'))
        return BaseResponse.model_validate(data)

    def list_algorithms(self):
        data = self._get_with_fallback(['/v1/algo/list', '/algo/list'])
        return BaseResponse.model_validate(data)

    def get_algorithm_groups(self, algo_id: str):
        try:
            data = self._get_with_fallback([
                f'/v1/algo/{algo_id}/groups',
                f'/algo/{algo_id}/group/info',
            ])
            return BaseResponse.model_validate(data)
        except RuntimeError as exc:
            if '404' in str(exc):
                return BaseResponse(code=404, msg='algo not found', data=None)
            raise

    def list_doc_chunks(self, algo_id: str, kb_id: str, doc_id: str, group: str, offset: int, page_size: int):
        data = self._get('/doc/chunks', params={
            'algo_id': algo_id,
            'kb_id': kb_id,
            'doc_id': doc_id,
            'group': group,
            'offset': offset,
            'page_size': page_size,
        })
        return BaseResponse.model_validate(data)


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
            tables_info_dict={'tables': [DOCUMENTS_TABLE_INFO, KBS_TABLE_INFO, KB_DOCUMENTS_TABLE_INFO,
                                         KB_ALGORITHM_TABLE_INFO, PARSE_STATE_TABLE_INFO,
                                         IDEMPOTENCY_RECORDS_TABLE_INFO, CALLBACK_RECORDS_TABLE_INFO,
                                         DOC_SERVICE_TASKS_TABLE_INFO]},
        )
        self._ensure_indexes()
        self._parser_client = _ParserClient(parser_url=parser_url)
        self._callback_url = callback_url
        self._upsert_default_kb()

    def set_callback_url(self, callback_url: str):
        self._callback_url = callback_url

    def _ensure_indexes(self):
        stmts = [
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_docs_path ON lazyllm_documents(path)',
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
            'CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_algorithm_kb_id ON lazyllm_kb_algorithm(kb_id)',
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
        ]
        for stmt in stmts:
            self._db_manager.execute_commit(stmt)

    def _upsert_default_kb(self):
        self._ensure_kb('__default__', display_name='__default__')
        self._ensure_kb_algorithm('__default__', '__default__')
        self._cleanup_idempotency_records()

    def _ensure_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                   owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                   update_fields: Optional[Set[str]] = None):
        now = now_ts()
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if row is None:
                row = Kb(
                    kb_id=kb_id,
                    display_name=display_name,
                    description=description,
                    doc_count=0,
                    status=KBStatus.ACTIVE.value,
                    owner_id=owner_id,
                    meta=_to_json(meta),
                    created_at=now,
                    updated_at=now,
                )
            else:
                if update_fields is None:
                    update_fields = set()
                if 'display_name' in update_fields:
                    row.display_name = display_name
                if 'description' in update_fields:
                    row.description = description
                if 'owner_id' in update_fields:
                    row.owner_id = owner_id
                if 'meta' in update_fields:
                    row.meta = _to_json(meta)
                if row.status == KBStatus.DELETED.value:
                    row.status = KBStatus.ACTIVE.value
                row.updated_at = now
            session.add(row)

    def _ensure_kb_algorithm(self, kb_id: str, algo_id: str):
        now = now_ts()
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            row = session.query(Rel).filter(Rel.kb_id == kb_id).first()
            if row is None:
                row = Rel(kb_id=kb_id, algo_id=algo_id, created_at=now, updated_at=now)
            elif row.algo_id != algo_id:
                raise DocServiceError(
                    'E_STATE_CONFLICT', f'kb {kb_id} is already bound to algorithm {row.algo_id}',
                    {'kb_id': kb_id, 'bound_algo_id': row.algo_id, 'requested_algo_id': algo_id}
                )
            else:
                row.updated_at = now
            session.add(row)

    def _get_kb(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            return _orm_to_dict(row) if row else None

    def _get_kb_algorithm(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            row = session.query(Rel).filter(Rel.kb_id == kb_id).first()
            return _orm_to_dict(row) if row else None

    @staticmethod
    def _build_kb_data(kb_row, algo_row=None):
        return {
            'kb_id': kb_row.kb_id,
            'display_name': kb_row.display_name,
            'description': kb_row.description,
            'doc_count': kb_row.doc_count,
            'status': kb_row.status,
            'owner_id': kb_row.owner_id,
            'meta': _from_json(kb_row.meta),
            'algo_id': algo_row.algo_id if algo_row is not None else None,
            'created_at': kb_row.created_at,
            'updated_at': kb_row.updated_at,
        }

    def _validate_kb_algorithm(self, kb_id: str, algo_id: str):
        kb = self._get_kb(kb_id)
        if kb is None:
            raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
        if kb.get('status') != KBStatus.ACTIVE.value:
            raise DocServiceError('E_STATE_CONFLICT', f'kb is not active: {kb_id}', {'kb_id': kb_id})
        binding = self._get_kb_algorithm(kb_id)
        if binding is None:
            raise DocServiceError('E_STATE_CONFLICT', f'kb has no algorithm binding: {kb_id}', {'kb_id': kb_id})
        if binding['algo_id'] != algo_id:
            raise DocServiceError(
                'E_INVALID_PARAM', f'kb {kb_id} is bound to algorithm {binding["algo_id"]}',
                {'kb_id': kb_id, 'bound_algo_id': binding['algo_id'], 'requested_algo_id': algo_id}
            )
        return binding

    def _ensure_algorithm_exists(self, algo_id: str):
        algorithms = self.list_algorithms()
        if any(item.get('algo_id') == algo_id for item in algorithms):
            return
        raise DocServiceError('E_INVALID_PARAM', f'invalid algo_id: {algo_id}', {'algo_id': algo_id})

    def _ensure_kb_document(self, kb_id: str, doc_id: str):
        now = now_ts()
        created = False
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first()
            if row is None:
                created = True
                row = Rel(kb_id=kb_id, doc_id=doc_id, created_at=now, updated_at=now)
            else:
                row.updated_at = now
            session.add(row)
        if created:
            self._refresh_kb_doc_count(kb_id)
        return created

    def _remove_kb_document(self, kb_id: str, doc_id: str):
        removed = False
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first()
            if row is not None:
                session.delete(row)
                removed = True
        if removed:
            self._refresh_kb_doc_count(kb_id)
        return removed

    def _load_idempotency_record(self, endpoint: str, idempotency_key: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            row = session.query(Record).filter(
                Record.endpoint == endpoint,
                Record.idempotency_key == idempotency_key,
            ).first()
            return _orm_to_dict(row) if row else None

    def _cleanup_idempotency_records(self, ttl_days: int = 7):
        cutoff = now_ts() - timedelta(days=ttl_days)
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            session.query(Record).filter(Record.updated_at < cutoff).delete()

    def _claim_idempotency_key(self, endpoint: str, idempotency_key: str, req_hash: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(IDEMPOTENCY_RECORDS_TABLE_INFO['name'])
            now = now_ts()
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
            row.response_json = _stable_json(response)
            row.updated_at = now_ts()
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
        req_hash = _hash_payload(payload)
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
        self._complete_idempotency_record(endpoint, idempotency_key, response)
        return response

    def _record_callback(self, callback_id: str, task_id: str):
        with self._db_manager.get_session() as session:
            Record = self._db_manager.get_table_orm_class(CALLBACK_RECORDS_TABLE_INFO['name'])
            session.add(Record(callback_id=callback_id, task_id=task_id, created_at=now_ts()))
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
        now = now_ts()
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            session.add(Task(
                task_id=task_id,
                task_type=task_type.value,
                doc_id=doc_id,
                kb_id=kb_id,
                algo_id=algo_id,
                status=status.value,
                message=_to_json(message),
                error_code=None,
                error_msg=None,
                created_at=now,
                updated_at=now,
                started_at=None,
                finished_at=None,
            ))
        return self._get_task_record(task_id)

    def _get_task_record(self, task_id: str):
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            row = session.query(Task).filter(Task.task_id == task_id).first()
            if row is None:
                return None
            task = _orm_to_dict(row)
            task['message'] = _from_json(task.get('message'))
            return task

    def _update_task_record(self, task_id: str, **fields):
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            row = session.query(Task).filter(Task.task_id == task_id).first()
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = now_ts()
            session.add(row)
        return self._get_task_record(task_id)

    def _refresh_kb_doc_count(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            kb_row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if kb_row is None:
                return
            kb_row.doc_count = session.query(Rel).filter(Rel.kb_id == kb_id).count()
            if kb_row.status == KBStatus.DELETING.value and kb_row.doc_count == 0:
                kb_row.status = KBStatus.DELETED.value
            kb_row.updated_at = now_ts()
            session.add(kb_row)

    def _list_kb_doc_ids(self, kb_id: str) -> List[str]:
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            rows = session.query(Rel.doc_id).filter(Rel.kb_id == kb_id).all()
            return [row[0] for row in rows]

    def _has_kb_document(self, kb_id: str, doc_id: str):
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            return session.query(Rel).filter(Rel.kb_id == kb_id, Rel.doc_id == doc_id).first() is not None

    def _doc_relation_count(self, doc_id: str):
        with self._db_manager.get_session() as session:
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            return session.query(Rel).filter(Rel.doc_id == doc_id).count()

    def _get_doc(self, doc_id: str):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            return _orm_to_dict(row) if row else None

    def _upsert_doc(
        self,
        doc_id: str,
        filename: str,
        path: str,
        metadata: Dict[str, Any],
        source_type: SourceType,
        upload_status: DocStatus = DocStatus.SUCCESS,
    ):
        now = now_ts()
        file_type = os.path.splitext(path)[1].lstrip('.').lower() or None
        size_bytes = os.path.getsize(path) if os.path.exists(path) else None
        content_hash = _sha256_file(path) if os.path.exists(path) else None
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                row = Doc(
                    doc_id=doc_id,
                    filename=filename,
                    path=path,
                    meta=_to_json(metadata),
                    upload_status=upload_status.value,
                    source_type=source_type.value,
                    file_type=file_type,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.filename = filename
                row.path = path
                row.meta = _to_json(metadata)
                row.upload_status = upload_status.value
                row.source_type = source_type.value
                row.file_type = file_type
                row.content_hash = content_hash
                row.size_bytes = size_bytes
                row.updated_at = now
            session.add(row)
        return self._get_doc(doc_id)

    def _set_doc_upload_status(self, doc_id: str, status: DocStatus):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                return None
            row.upload_status = status.value
            row.updated_at = now_ts()
            session.add(row)
        return self._get_doc(doc_id)

    def _get_parse_snapshot(self, doc_id: str, kb_id: str, algo_id: str):
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            row = (
                session.query(State)
                .filter(State.doc_id == doc_id, State.kb_id == kb_id, State.algo_id == algo_id)
                .first()
            )
            return _orm_to_dict(row) if row else None

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

    def _delete_parse_snapshots(self, doc_id: str, kb_id: str):
        with self._db_manager.get_session() as session:
            State = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            session.query(State).filter(State.doc_id == doc_id, State.kb_id == kb_id).delete()

    def _delete_doc_if_orphaned(self, doc_id: str) -> bool:
        if self._doc_relation_count(doc_id) > 0:
            return False
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                return False
            session.delete(row)
        return True

    def _purge_deleted_kb_doc_data(self, kb_id: str, doc_id: str, remove_relation: bool = False):
        if remove_relation:
            self._remove_kb_document(kb_id, doc_id)
        self._delete_parse_snapshots(doc_id, kb_id)
        if not self._delete_doc_if_orphaned(doc_id):
            self._sync_doc_upload_status(doc_id)

    def _mark_task_cleanup_policy(self, task_id: str, cleanup_policy: str):
        task = self._get_task_record(task_id)
        if task is None:
            return
        message = task.get('message') or {}
        if message.get('cleanup_policy') == cleanup_policy:
            return
        message['cleanup_policy'] = cleanup_policy
        self._update_task_record(task_id, message=_to_json(message))

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
        binding = self._get_kb_algorithm(kb_id)
        default_algo_id = binding['algo_id'] if binding is not None else '__default__'
        items = []
        for doc_id in self._list_kb_doc_ids(kb_id):
            snapshot = (
                self._get_parse_snapshot(doc_id, kb_id, default_algo_id)
                or self._get_latest_parse_snapshot(doc_id, kb_id)
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
            items.append({
                'action': 'enqueue_delete',
                'doc_id': doc_id,
                'algo_id': snapshot.get('algo_id') or default_algo_id,
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
        now = now_ts()
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

    def _call_parser_client(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except TypeError as exc:
            compat_kwargs = dict(kwargs)
            removed = False
            for field in ('callback_url',):
                if field in compat_kwargs and field in str(exc):
                    compat_kwargs.pop(field, None)
                    removed = True
            if not removed:
                raise
            return method(*args, **compat_kwargs)

    def _create_parser_task(self, task_id: str, doc_id: str, kb_id: str, algo_id: str, task_type: TaskType,
                            file_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                            reparse_group: Optional[str] = None, parser_kb_id: Optional[str] = None,
                            transfer_params: Optional[Dict[str, Any]] = None):
        if task_type in (TaskType.DOC_ADD, TaskType.DOC_TRANSFER):
            if not file_path:
                raise RuntimeError(f'file_path is required for task_type {task_type.value}')
            task_resp = self._call_parser_client(
                self._parser_client.add_doc,
                task_id, algo_id, parser_kb_id or kb_id, doc_id, file_path, metadata,
                callback_url=self._callback_url, transfer_params=transfer_params,
            )
        elif task_type == TaskType.DOC_REPARSE:
            if not file_path:
                raise RuntimeError('file_path is required for reparse task')
            task_resp = self._call_parser_client(
                self._parser_client.add_doc,
                task_id, algo_id, kb_id, doc_id, file_path, metadata,
                reparse_group=reparse_group or 'all', callback_url=self._callback_url,
            )
        elif task_type == TaskType.DOC_UPDATE_META:
            task_resp = self._call_parser_client(
                self._parser_client.update_meta,
                task_id, algo_id, kb_id, doc_id, metadata, file_path, callback_url=self._callback_url,
            )
        elif task_type == TaskType.DOC_DELETE:
            task_resp = self._call_parser_client(
                self._parser_client.delete_doc,
                task_id, algo_id, kb_id, doc_id, callback_url=self._callback_url,
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
        extra_message: Optional[Dict[str, Any]] = None,
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
            queued_at=now_ts(),
            started_at=None,
            finished_at=None,
            error_code=None,
            error_msg=None,
            failed_stage=None,
        )
        try:
            self._create_parser_task(
                task_id, doc_id, kb_id, algo_id, task_type,
                file_path=file_path, metadata=metadata, reparse_group=reparse_group,
                parser_kb_id=parser_kb_id, transfer_params=transfer_params,
            )
        except Exception as exc:
            finished_at = now_ts()
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
            self._set_doc_upload_status(doc_id, status)
            return
        if task_type == TaskType.DOC_DELETE:
            if status == DocStatus.DELETING:
                if self._doc_relation_count(doc_id) <= 1:
                    self._set_doc_upload_status(doc_id, DocStatus.DELETING)
                return
            if status == DocStatus.DELETED:
                target = DocStatus.SUCCESS if self._doc_relation_count(doc_id) > 0 else DocStatus.DELETED
                self._set_doc_upload_status(doc_id, target)
                return
            if status in (DocStatus.FAILED, DocStatus.CANCELED):
                target = DocStatus.SUCCESS if self._doc_relation_count(doc_id) > 0 else DocStatus.DELETED
                self._set_doc_upload_status(doc_id, target)
                return

    def _prepare_upload_items(self, request: UploadRequest) -> List[Dict[str, Any]]:
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
                self._assert_action_allowed(item['doc_id'], request.kb_id, request.algo_id, 'upload')
        return prepared_items

    def _prepare_reparse_items(self, request: ReparseRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        for doc_id in request.doc_ids:
            doc = self._get_doc(doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}')
            self._assert_action_allowed(doc_id, request.kb_id, request.algo_id, 'reparse')
            prepared_items.append({
                'doc_id': doc_id,
                'file_path': doc.get('path'),
                'metadata': _from_json(doc.get('meta')),
            })
        return prepared_items

    def _prepare_delete_items(self, request: DeleteRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        for doc_id in request.doc_ids:
            doc = self._get_doc(doc_id)
            snapshot = (
                self._get_parse_snapshot(doc_id, request.kb_id, request.algo_id)
                or self._get_latest_parse_snapshot(doc_id, request.kb_id)
            )
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
            self._assert_action_allowed(doc_id, request.kb_id, request.algo_id, 'delete')
            prepared_items.append({'doc_id': doc_id, 'action': 'execute'})
        return prepared_items

    def _prepare_metadata_patch_items(self, request: MetadataPatchRequest) -> List[Dict[str, Any]]:
        prepared_items = []
        for item in request.items:
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            self._assert_action_allowed(item.doc_id, request.kb_id, request.algo_id, 'metadata')
            merged = _from_json(doc.get('meta'))
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
            target_key = (item.doc_id, item.target_kb_id, item.target_algo_id)
            if target_key in seen_targets:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    'duplicate transfer target detected',
                    {
                        'doc_id': item.doc_id,
                        'target_kb_id': item.target_kb_id,
                        'target_algo_id': item.target_algo_id,
                    },
                )
            seen_targets.add(target_key)
            if item.source_algo_id != item.target_algo_id:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    'transfer across different algorithms is not supported',
                    {
                        'doc_id': item.doc_id,
                        'source_algo_id': item.source_algo_id,
                        'target_algo_id': item.target_algo_id,
                    },
                )
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(item.source_kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            self._validate_kb_algorithm(item.source_kb_id, item.source_algo_id)
            self._validate_kb_algorithm(item.target_kb_id, item.target_algo_id)
            self._assert_action_allowed(item.doc_id, item.source_kb_id, item.source_algo_id, 'transfer')
            if self._has_kb_document(item.target_kb_id, item.doc_id):
                raise DocServiceError(
                    'E_STATE_CONFLICT',
                    f'doc already exists in target kb: {item.doc_id}',
                    {'doc_id': item.doc_id, 'target_kb_id': item.target_kb_id},
                )
            source_snapshot = self._get_parse_snapshot(item.doc_id, item.source_kb_id, item.source_algo_id)
            if source_snapshot is None or source_snapshot.get('status') != DocStatus.SUCCESS.value:
                raise DocServiceError(
                    'E_STATE_CONFLICT',
                    f'doc transfer requires source parse status SUCCESS: {item.doc_id}',
                    {
                        'doc_id': item.doc_id,
                        'source_kb_id': item.source_kb_id,
                        'source_algo_id': item.source_algo_id,
                        'status': source_snapshot.get('status') if source_snapshot else None,
                    },
                )
            prepared_items.append({
                'doc_id': item.doc_id,
                'source_kb_id': item.source_kb_id,
                'source_algo_id': item.source_algo_id,
                'target_kb_id': item.target_kb_id,
                'target_algo_id': item.target_algo_id,
                'mode': item.mode,
                'file_path': doc.get('path'),
                'metadata': _from_json(doc.get('meta')),
                'transfer_params': {
                    'mode': 'mv' if item.mode == 'move' else 'cp',
                    'target_algo_id': item.target_algo_id,
                    'target_doc_id': item.doc_id,
                    'target_kb_id': item.target_kb_id,
                },
            })
        return prepared_items

    def upload(self, request: UploadRequest) -> List[Dict[str, Any]]:
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        prepared_items = self._prepare_upload_items(request)
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            doc_id = item['doc_id']
            file_path = item['file_path']
            metadata = item['metadata']
            doc = self._upsert_doc(
                doc_id=doc_id,
                filename=item['filename'],
                path=file_path,
                metadata=metadata,
                source_type=request.source_type,
                upload_status=DocStatus.WAITING,
            )
            self._ensure_kb_document(request.kb_id, doc_id)
            try:
                task_id, snapshot = self._enqueue_task(
                    doc_id, request.kb_id, request.algo_id, TaskType.DOC_ADD,
                    idempotency_key=request.idempotency_key,
                    file_path=file_path,
                    metadata=metadata,
                )
                error_code = None
                error_msg = None
                accepted = True
            except Exception as exc:
                snapshot = self._get_parse_snapshot(doc_id, request.kb_id, request.algo_id) or {}
                doc = self._get_doc(doc_id) or doc
                task_id = snapshot.get('current_task_id')
                error_code = snapshot.get('last_error_code') or type(exc).__name__
                error_msg = snapshot.get('last_error_msg') or str(exc)
                accepted = False
            items.append({
                'doc_id': doc_id,
                'kb_id': request.kb_id,
                'algo_id': request.algo_id,
                'upload_status': doc['upload_status'],
                'parse_status': snapshot.get('status', DocStatus.FAILED.value),
                'task_id': task_id,
                'accepted': accepted,
                'error_code': error_code,
                'error_msg': error_msg,
            })
        return items

    def add_files(self, request: AddRequest) -> List[Dict[str, Any]]:
        return self.upload(UploadRequest(
            items=request.items,
            kb_id=request.kb_id,
            algo_id=request.algo_id,
            source_type=request.source_type,
            idempotency_key=request.idempotency_key,
        ))

    def reparse(self, request: ReparseRequest) -> List[str]:
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        self._validate_unique_doc_ids(request.doc_ids, field_name='doc_id')
        prepared_items = self._prepare_reparse_items(request)
        task_ids = []
        for item in prepared_items:
            task_id, _ = self._enqueue_task(
                item['doc_id'], request.kb_id, request.algo_id, TaskType.DOC_REPARSE,
                idempotency_key=request.idempotency_key,
                file_path=item['file_path'],
                metadata=item['metadata'],
                reparse_group='all',
            )
            task_ids.append(task_id)
        return task_ids

    def delete(self, request: DeleteRequest) -> List[Dict[str, Any]]:
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        self._validate_unique_doc_ids(request.doc_ids, field_name='doc_id')
        prepared_items = self._prepare_delete_items(request)
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            doc_id = item['doc_id']
            if item.get('action') == 'noop':
                items.append({
                    'doc_id': doc_id,
                    'accepted': True,
                    'task_id': item.get('task_id'),
                    'status': item['status'],
                    'error_code': None,
                })
                continue
            snapshot = self._get_parse_snapshot(doc_id, request.kb_id, request.algo_id)
            if (
                snapshot is not None
                and snapshot.get('status') == DocStatus.WAITING.value
                and snapshot.get('task_type') == TaskType.DOC_ADD.value
                and snapshot.get('current_task_id')
            ):
                cancel_resp = self.cancel_task(snapshot['current_task_id'])
                if cancel_resp.code != 200:
                    raise DocServiceError(
                        'E_STATE_CONFLICT',
                        cancel_resp.msg,
                        (
                            cancel_resp.data
                            if isinstance(cancel_resp.data, dict)
                            else {'task_id': snapshot['current_task_id']}
                        ),
                    )
                items.append({
                    'doc_id': doc_id,
                    'accepted': True,
                    'task_id': snapshot['current_task_id'],
                    'status': DocStatus.CANCELED.value,
                    'error_code': None,
                })
                continue
            with self._db_manager.get_session() as session:
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
                if self._doc_relation_count(doc_id) <= 1:
                    row.upload_status = DocStatus.DELETING.value
                row.updated_at = now_ts()
                session.add(row)

            task_id, snapshot = self._enqueue_task(
                doc_id, request.kb_id, request.algo_id, TaskType.DOC_DELETE,
                idempotency_key=request.idempotency_key,
            )
            items.append({
                'doc_id': doc_id,
                'accepted': True,
                'task_id': task_id,
                'status': snapshot['status'],
                'error_code': None,
            })
        return items

    def transfer(self, request: TransferRequest) -> List[Dict[str, Any]]:
        prepared_items = self._prepare_transfer_items(request)
        items: List[Dict[str, Any]] = []
        for item in prepared_items:
            task_id = None
            try:
                self._ensure_kb_document(item['target_kb_id'], item['doc_id'])
                task_id, snapshot = self._enqueue_task(
                    item['doc_id'], item['target_kb_id'], item['target_algo_id'], TaskType.DOC_TRANSFER,
                    idempotency_key=request.idempotency_key,
                    file_path=item['file_path'],
                    metadata=item['metadata'],
                    parser_kb_id=item['source_kb_id'],
                    transfer_params=item['transfer_params'],
                    extra_message={
                        'source_kb_id': item['source_kb_id'],
                        'source_algo_id': item['source_algo_id'],
                        'target_kb_id': item['target_kb_id'],
                        'target_algo_id': item['target_algo_id'],
                        'mode': item['mode'],
                    },
                )
                error_code = None
                error_msg = None
                accepted = True
            except Exception as exc:
                snapshot = self._get_parse_snapshot(item['doc_id'], item['target_kb_id'], item['target_algo_id']) or {}
                task_id = task_id or snapshot.get('current_task_id')
                error_code = snapshot.get('last_error_code')
                if not error_code:
                    error_code = exc.biz_code if isinstance(exc, DocServiceError) else type(exc).__name__
                error_msg = snapshot.get('last_error_msg') or (exc.msg if isinstance(exc, DocServiceError) else str(exc))
                accepted = False
            items.append({
                'doc_id': item['doc_id'],
                'task_id': task_id,
                'source_kb_id': item['source_kb_id'],
                'target_kb_id': item['target_kb_id'],
                'source_algo_id': item['source_algo_id'],
                'target_algo_id': item['target_algo_id'],
                'mode': item['mode'],
                'status': snapshot.get('status', DocStatus.FAILED.value),
                'accepted': accepted,
                'error_code': error_code,
                'error_msg': error_msg,
            })
        return items

    def patch_metadata(self, request: MetadataPatchRequest):
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        updated = []
        failed = []
        self._validate_unique_doc_ids([item.doc_id for item in request.items], field_name='doc_id')
        prepared_items = self._prepare_metadata_patch_items(request)
        for item in prepared_items:
            with self._db_manager.get_session() as session:
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                row = session.query(Doc).filter(Doc.doc_id == item['doc_id']).first()
                row.meta = _to_json(item['metadata'])
                row.updated_at = now_ts()
                session.add(row)

            task_id, _ = self._enqueue_task(
                item['doc_id'], request.kb_id, request.algo_id, TaskType.DOC_UPDATE_META,
                idempotency_key=request.idempotency_key,
                file_path=item['file_path'],
                metadata=item['metadata'],
            )
            updated.append({'doc_id': item['doc_id'], 'task_id': task_id})
        return {
            'updated_count': len(updated),
            'doc_ids': [u['doc_id'] for u in updated],
            'failed_items': failed,
            'items': updated,
        }

    def _sync_doc_upload_status(self, doc_id: str):
        target = DocStatus.SUCCESS if self._doc_relation_count(doc_id) > 0 else DocStatus.DELETED
        self._set_doc_upload_status(doc_id, target)

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
        required_fields = {'task_type', 'doc_id', 'kb_id', 'algo_id'}
        if required_fields.issubset(payload.keys()):
            return {
                'task_id': callback.task_id,
                'task_type': payload['task_type'],
                'doc_id': payload['doc_id'],
                'kb_id': payload['kb_id'],
                'algo_id': payload['algo_id'],
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
                    started_at=now_ts(),
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
                        started_at=now_ts(),
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
                if source_kb_id and source_kb_id != kb_id:
                    self._remove_kb_document(source_kb_id, doc_id)
                    self._delete_parse_snapshots(doc_id, source_kb_id)
                    self._sync_doc_upload_status(doc_id)

            self._update_task_record(
                callback.task_id,
                status=final_status.value,
                error_code=callback.error_code,
                error_msg=callback.error_msg,
                finished_at=now_ts(),
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
                    finished_at=now_ts(),
                ),
            )

            if task_type == TaskType.DOC_DELETE and final_status == DocStatus.DELETED:
                self._remove_kb_document(kb_id, doc_id)
            self._apply_doc_upload_status(doc_id, task_type, final_status)
            if task_type == TaskType.DOC_DELETE and final_status == DocStatus.DELETED and cleanup_policy == 'purge':
                self._purge_deleted_kb_doc_data(kb_id, doc_id)
                self._finalize_kb_deletion_if_empty(kb_id)

            return {'ack': True, 'deduped': False, 'ignored_reason': None}
        except Exception:
            self._forget_callback_record(callback.callback_id, callback.task_id)
            raise

    def list_docs(
        self,
        status: Optional[List[str]] = None,
        kb_id: Optional[str] = None,
        algo_id: Optional[str] = None,
        keyword: Optional[str] = None,
        include_deleted_or_canceled: bool = True,
        page: int = 1,
        page_size: int = 20,
    ):
        page = max(page, 1)
        page_size = max(1, page_size)
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            query = session.query(Doc, Rel).join(Rel, Doc.doc_id == Rel.doc_id)

            if kb_id:
                query = query.filter(Rel.kb_id == kb_id)
            if keyword:
                like_expr = f'%{keyword}%'
                query = query.filter((Doc.filename.like(like_expr)) | (Doc.path.like(like_expr)))
            if not include_deleted_or_canceled:
                query = query.filter(~Doc.upload_status.in_([DocStatus.DELETED.value, DocStatus.CANCELED.value]))

            rows = query.order_by(Rel.updated_at.desc(), Doc.updated_at.desc()).all()
            items = []
            for doc_row, rel_row in rows:
                doc = _orm_to_dict(doc_row)
                relation = _orm_to_dict(rel_row)
                snapshot = (
                    self._get_parse_snapshot(doc['doc_id'], relation['kb_id'], algo_id)
                    if algo_id else
                    self._get_latest_parse_snapshot(doc['doc_id'], relation['kb_id'])
                )
                if status and (snapshot is None or snapshot.get('status') not in status):
                    continue
                doc['metadata'] = _from_json(doc.get('meta'))
                items.append({'doc': doc, 'relation': relation, 'snapshot': snapshot})
            total = len(items)
            page_items = items[(page - 1) * page_size:page * page_size]
            return {'items': page_items, 'total': total, 'page': page, 'page_size': page_size}

    def get_doc_detail(self, doc_id: str):
        doc = self._get_doc(doc_id)
        if not doc:
            raise DocServiceError('E_NOT_FOUND', f'doc not found: {doc_id}', {'doc_id': doc_id})

        doc['metadata'] = _from_json(doc.get('meta'))
        snapshots = self.list_docs(page=1, page_size=2000)['items']
        matched_items = []
        for item in snapshots:
            if item['doc']['doc_id'] == doc_id:
                matched_items.append(item)
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
            except Exception:
                pass
        page = max(page, 1)
        page_size = max(page_size, 1)
        with self._db_manager.get_session() as session:
            Task = self._db_manager.get_table_orm_class(DOC_SERVICE_TASKS_TABLE_INFO['name'])
            query = session.query(Task)
            if status:
                query = query.filter(Task.status.in_(status))
            total = query.count()
            rows = (
                query.order_by(Task.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            items = []
            for row in rows:
                task = _orm_to_dict(row)
                task['message'] = _from_json(task.get('message'))
                items.append(task)
        return BaseResponse(code=200, msg='success', data={
            'items': items,
            'total': total,
            'page': page,
            'page_size': page_size,
        })

    def get_task(self, task_id: str):
        parser_get_task = getattr(self._parser_client, 'get_task', None)
        if callable(parser_get_task):
            try:
                return parser_get_task(task_id)
            except Exception:
                pass
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
            raise fastapi.HTTPException(status_code=502, detail=resp.msg)
        return resp.data

    def get_algo_groups(self, algo_id: str):
        resp = self._parser_client.get_algorithm_groups(algo_id)
        if resp.code == 404:
            raise fastapi.HTTPException(status_code=404, detail='algo not found')
        if resp.code != 200:
            raise fastapi.HTTPException(status_code=502, detail=resp.msg)
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

    def list_chunks(
        self,
        kb_id: str,
        doc_id: str,
        group: str,
        algo_id: str = '__default__',
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
        self._validate_kb_algorithm(kb_id, algo_id)
        doc = self._get_doc(doc_id)
        if doc is None or not self._has_kb_document(kb_id, doc_id):
            raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}', {'kb_id': kb_id, 'doc_id': doc_id})
        groups = self.get_algo_groups(algo_id)
        if not any(item.get('name') == group for item in groups):
            raise DocServiceError(
                'E_INVALID_PARAM',
                f'invalid group: {group}',
                {'algo_id': algo_id, 'group': group},
            )
        page = max(page, 1)
        page_size = max(page_size, 1)
        offset = (page - 1) * page_size if offset is None else max(offset, 0)
        resp = self._parser_client.list_doc_chunks(
            algo_id=algo_id,
            kb_id=kb_id,
            doc_id=doc_id,
            group=group,
            offset=offset,
            page_size=page_size,
        )
        if resp.code == 404:
            raise DocServiceError('E_NOT_FOUND', resp.msg, {'kb_id': kb_id, 'doc_id': doc_id, 'group': group})
        if resp.code == 400:
            raise DocServiceError('E_INVALID_PARAM', resp.msg, {'kb_id': kb_id, 'doc_id': doc_id, 'group': group})
        if resp.code != 200:
            raise fastapi.HTTPException(status_code=502, detail=resp.msg)
        data = dict(resp.data or {})
        data['page'] = page
        data['page_size'] = page_size
        data['offset'] = offset
        data.setdefault('items', [])
        data.setdefault('total', 0)
        return data

    def health(self):
        return {
            'status': 'ok',
            'version': 'v1',
            'deps': {
                'sql': True,
                'parser': bool(getattr(self._parser_client, '_parser_url', None)),
            },
        }

    def list_kbs(
        self,
        page: int = 1,
        page_size: int = 20,
        keyword: Optional[str] = None,
        status: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
    ):
        page = max(page, 1)
        page_size = max(page_size, 1)
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            query = session.query(Kb, Rel).outerjoin(Rel, Rel.kb_id == Kb.kb_id)
            query = query.filter(Kb.kb_id != '__default__')
            if keyword:
                like_expr = f'%{keyword}%'
                query = query.filter(
                    sqlalchemy.or_(
                        Kb.kb_id.like(like_expr),
                        Kb.display_name.like(like_expr),
                        Kb.description.like(like_expr),
                    )
                )
            if status:
                query = query.filter(Kb.status.in_(status))
            if owner_id:
                query = query.filter(Kb.owner_id == owner_id)
            total = query.count()
            rows = (
                query.order_by(Kb.updated_at.desc(), Kb.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            items = [self._build_kb_data(kb_row, algo_row) for kb_row, algo_row in rows]
            return {'items': items, 'total': total, 'page': page, 'page_size': page_size}

    def get_kb(self, kb_id: str):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            row = (
                session.query(Kb, Rel)
                .outerjoin(Rel, Rel.kb_id == Kb.kb_id)
                .filter(Kb.kb_id == kb_id)
                .first()
            )
            if row is None:
                raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
            kb_row, algo_row = row
            return self._build_kb_data(kb_row, algo_row)

    def batch_get_kbs(self, kb_ids: List[str]):
        if not kb_ids:
            raise DocServiceError('E_INVALID_PARAM', 'kb_ids is required', {'kb_ids': kb_ids})
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_ALGORITHM_TABLE_INFO['name'])
            rows = (
                session.query(Kb, Rel)
                .outerjoin(Rel, Rel.kb_id == Kb.kb_id)
                .filter(Kb.kb_id.in_(kb_ids))
                .all()
            )
            row_map = {kb_row.kb_id: self._build_kb_data(kb_row, algo_row) for kb_row, algo_row in rows}
        items = []
        missing_kb_ids = []
        for kb_id in kb_ids:
            if kb_id in row_map:
                items.append(row_map[kb_id])
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
            binding = self._get_kb_algorithm(kb_id)
            if binding is None:
                self._ensure_kb_algorithm(kb_id, algo_id)
            elif binding['algo_id'] != algo_id:
                raise DocServiceError(
                    'E_STATE_CONFLICT', f'kb {kb_id} is already bound to algorithm {binding["algo_id"]}',
                    {'kb_id': kb_id, 'bound_algo_id': binding['algo_id'], 'requested_algo_id': algo_id}
                )
        self._ensure_kb(
            kb_id, display_name=display_name, description=description, owner_id=owner_id, meta=meta,
            update_fields=explicit_fields & {'display_name', 'description', 'owner_id', 'meta'},
        )
        return self.get_kb(kb_id)

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
                    kb_row.updated_at = now_ts()
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
