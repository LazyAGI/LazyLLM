from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
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
    TaskCreateRequest,
    TaskType,
    TransferRequest,
    UploadRequest,
    DocStatus,
    now_ts,
)
from .parsing_server import ParsingTaskServer


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
    def __init__(self, parser_server: Optional[ParsingTaskServer] = None, parser_url: Optional[str] = None):
        self._parser_server = parser_server
        if parser_url:
            parser_url = parser_url.rstrip('/')
            if parser_url.endswith('/_call') or parser_url.endswith('/generate'):
                parser_url = parser_url.rsplit('/', 1)[0]
            self._parser_url = parser_url
        else:
            self._parser_url = None

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

    def create_task(self, req: TaskCreateRequest):
        if self._parser_server:
            return self._parser_server.create_task(req)
        data = self._post('/v1/internal/tasks/create', req.model_dump(mode='json'))
        return BaseResponse.model_validate(data)

    def cancel_task(self, task_id: str):
        if self._parser_server:
            return self._parser_server.cancel_task(task_id)
        data = self._post('/v1/internal/tasks/cancel', {'task_id': task_id})
        return BaseResponse.model_validate(data)

    def list_tasks(self, status: Optional[List[str]], page: int, page_size: int):
        if self._parser_server:
            return self._parser_server.list_tasks(status=status, page=page, page_size=page_size)
        params: Dict[str, Any] = {'page': page, 'page_size': page_size}
        if status:
            params['status'] = status
        data = self._get('/v1/tasks', params=params)
        return BaseResponse.model_validate(data)

    def get_task(self, task_id: str):
        if self._parser_server:
            try:
                return self._parser_server.get_task(task_id)
            except (fastapi.HTTPException, requests.RequestException):
                return BaseResponse(code=404, msg='task not found', data=None)
        try:
            data = self._get(f'/v1/tasks/{task_id}')
            return BaseResponse.model_validate(data)
        except RuntimeError as exc:
            if '404' in str(exc):
                return BaseResponse(code=404, msg='task not found', data=None)
            raise

    def list_algorithms(self):
        if self._parser_server:
            return self._parser_server.list_algorithms()
        data = self._get('/v1/algo/list')
        return BaseResponse.model_validate(data)

    def get_algorithm_groups(self, algo_id: str):
        if self._parser_server:
            return self._parser_server.get_algorithm_groups(algo_id)
        try:
            data = self._get(f'/v1/algo/{algo_id}/groups')
            return BaseResponse.model_validate(data)
        except RuntimeError as exc:
            if '404' in str(exc):
                return BaseResponse(code=404, msg='algo not found', data=None)
            raise


class DocManager:
    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        parser_server: Optional[ParsingTaskServer] = None,
        parser_url: Optional[str] = None,
        callback_url: Optional[str] = None,
    ):
        if parser_server is None and not parser_url:
            raise ValueError('Either parser_server or parser_url must be provided')

        self._db_config = db_config or _get_default_db_config('doc_service')
        self._db_manager = SqlManager(
            **self._db_config,
            tables_info_dict={'tables': [DOCUMENTS_TABLE_INFO, KBS_TABLE_INFO, KB_DOCUMENTS_TABLE_INFO,
                                         KB_ALGORITHM_TABLE_INFO, PARSE_STATE_TABLE_INFO,
                                         IDEMPOTENCY_RECORDS_TABLE_INFO, CALLBACK_RECORDS_TABLE_INFO]},
        )
        self._ensure_indexes()
        self._parser_client = _ParserClient(parser_server=parser_server, parser_url=parser_url)
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
        ]
        for stmt in stmts:
            self._db_manager.execute_commit(stmt)

    def _upsert_default_kb(self):
        self._ensure_kb('__default__', display_name='__default__')
        self._ensure_kb_algorithm('__default__', '__default__')
        self._cleanup_idempotency_records()

    def _ensure_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                   owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
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
                if display_name is not None:
                    row.display_name = display_name
                if description is not None:
                    row.description = description
                if owner_id is not None:
                    row.owner_id = owner_id
                if meta is not None:
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
                    upload_status=DocStatus.SUCCESS.value,
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
                row.upload_status = DocStatus.SUCCESS.value
                row.source_type = source_type.value
                row.file_type = file_type
                row.content_hash = content_hash
                row.size_bytes = size_bytes
                row.updated_at = now
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

    def _assert_action_allowed(self, doc_id: str, kb_id: str, algo_id: str, action: str):
        snapshot = self._get_parse_snapshot(doc_id, kb_id, algo_id)
        if snapshot is None:
            return
        status = snapshot.get('status')
        if status == DocStatus.WORKING.value and action in ('upload', 'reparse', 'delete', 'transfer', 'metadata'):
            raise DocServiceError('E_STATE_CONFLICT', f'cannot {action} while state is WORKING')
        if status == DocStatus.DELETING.value and action in ('upload', 'reparse', 'delete', 'transfer', 'metadata'):
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

    def _create_parser_task(self, doc_id: str, kb_id: str, algo_id: str, task_type: TaskType):
        task_id = str(uuid4())
        req = TaskCreateRequest(
            task_id=task_id,
            task_type=task_type,
            doc_id=doc_id,
            kb_id=kb_id,
            algo_id=algo_id,
            callback_url=self._callback_url,
        )
        task_resp = self._parser_client.create_task(req)
        if task_resp.code != 200:
            raise RuntimeError(f'failed to enqueue parser task: {task_resp.msg}')
        return task_id

    def _enqueue_task(
        self, doc_id: str, kb_id: str, algo_id: str, task_type: TaskType,
        idempotency_key: Optional[str] = None, priority: int = 0,
    ):
        task_id = self._create_parser_task(doc_id, kb_id, algo_id, task_type)
        parse_status = DocStatus.DELETING if task_type == TaskType.DOC_DELETE else DocStatus.WAITING
        snapshot = self._upsert_parse_snapshot(
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
        return task_id, snapshot

    def upload(self, request: UploadRequest) -> List[Dict[str, Any]]:
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        items: List[Dict[str, Any]] = []
        for item in request.items:
            file_path = item.file_path
            if not os.path.exists(file_path):
                raise DocServiceError('E_INVALID_PARAM', f'file not found: {file_path}')
            doc_id = gen_doc_id(file_path, doc_id=item.doc_id)
            if self._has_kb_document(request.kb_id, doc_id):
                self._assert_action_allowed(doc_id, request.kb_id, request.algo_id, 'upload')
            doc = self._upsert_doc(
                doc_id=doc_id,
                filename=os.path.basename(file_path),
                path=file_path,
                metadata=item.metadata,
                source_type=request.source_type,
            )
            self._ensure_kb_document(request.kb_id, doc_id)
            task_id, snapshot = self._enqueue_task(
                doc_id, request.kb_id, request.algo_id, TaskType.DOC_ADD,
                idempotency_key=request.idempotency_key,
            )
            items.append({
                'doc_id': doc_id,
                'kb_id': request.kb_id,
                'algo_id': request.algo_id,
                'upload_status': doc['upload_status'],
                'parse_status': snapshot['status'],
                'task_id': task_id,
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
        task_ids = []
        for doc_id in request.doc_ids:
            if self._get_doc(doc_id) is None or not self._has_kb_document(request.kb_id, doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}')
            self._assert_action_allowed(doc_id, request.kb_id, request.algo_id, 'reparse')
            task_id, _ = self._enqueue_task(
                doc_id, request.kb_id, request.algo_id, TaskType.DOC_REPARSE,
                idempotency_key=request.idempotency_key,
            )
            task_ids.append(task_id)
        return task_ids

    def delete(self, request: DeleteRequest) -> List[Dict[str, Any]]:
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        items: List[Dict[str, Any]] = []
        for doc_id in request.doc_ids:
            doc = self._get_doc(doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {doc_id}')
            self._assert_action_allowed(doc_id, request.kb_id, request.algo_id, 'delete')
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
        items: List[Dict[str, Any]] = []
        for item in request.items:
            if item.mode not in ('move', 'copy'):
                raise DocServiceError(
                    'E_INVALID_PARAM', f'invalid transfer mode: {item.mode}', {'mode': item.mode}
                )
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(item.source_kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            self._validate_kb_algorithm(item.source_kb_id, item.source_algo_id)
            self._validate_kb_algorithm(item.target_kb_id, item.target_algo_id)
            self._assert_action_allowed(item.doc_id, item.source_kb_id, item.source_algo_id, 'transfer')
            self._ensure_kb_document(item.target_kb_id, item.doc_id)
            if item.mode == 'move':
                self._remove_kb_document(item.source_kb_id, item.doc_id)
            task_id, snapshot = self._enqueue_task(
                item.doc_id, item.target_kb_id, item.target_algo_id, TaskType.DOC_TRANSFER,
                idempotency_key=request.idempotency_key,
            )
            items.append({
                'doc_id': item.doc_id,
                'task_id': task_id,
                'source_kb_id': item.source_kb_id,
                'target_kb_id': item.target_kb_id,
                'source_algo_id': item.source_algo_id,
                'target_algo_id': item.target_algo_id,
                'mode': item.mode,
                'status': snapshot['status'],
            })
        return items

    def patch_metadata(self, request: MetadataPatchRequest):
        self._validate_kb_algorithm(request.kb_id, request.algo_id)
        updated = []
        failed = []
        for item in request.items:
            doc = self._get_doc(item.doc_id)
            if doc is None or not self._has_kb_document(request.kb_id, item.doc_id):
                raise DocServiceError('E_NOT_FOUND', f'doc not found in kb: {item.doc_id}')
            self._assert_action_allowed(item.doc_id, request.kb_id, request.algo_id, 'metadata')
            merged = _from_json(doc.get('meta'))
            merged.update(item.patch)
            with self._db_manager.get_session() as session:
                Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
                row = session.query(Doc).filter(Doc.doc_id == item.doc_id).first()
                row.meta = _to_json(merged)
                row.updated_at = now_ts()
                session.add(row)

            task_id, _ = self._enqueue_task(
                item.doc_id, request.kb_id, request.algo_id, TaskType.DOC_UPDATE_META,
                idempotency_key=request.idempotency_key,
            )
            updated.append({'doc_id': item.doc_id, 'task_id': task_id})
        return {
            'updated_count': len(updated),
            'doc_ids': [u['doc_id'] for u in updated],
            'failed_items': failed,
            'items': updated,
        }

    def _sync_doc_upload_status(self, doc_id: str):
        with self._db_manager.get_session() as session:
            Doc = self._db_manager.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
            if row is None:
                return
            has_rel = session.query(Rel).filter(Rel.doc_id == doc_id).first() is not None
            row.upload_status = DocStatus.SUCCESS.value if has_rel else DocStatus.DELETED.value
            row.updated_at = now_ts()
            session.add(row)

    def on_task_callback(self, callback: TaskCallbackRequest):
        if not self._record_callback(callback.callback_id, callback.task_id):
            return {'ack': True, 'deduped': True, 'ignored_reason': None}
        task = self._parser_client.get_task(callback.task_id)
        if task.code != 200:
            return {'ack': True, 'ignored_reason': 'task_not_found'}
        task_data = task.data
        doc_id = task_data['doc_id']
        kb_id = task_data['kb_id']
        algo_id = task_data['algo_id']
        task_type = TaskType(task_data['task_type'])
        snapshot = self._get_parse_snapshot(doc_id, kb_id, algo_id)
        if snapshot and snapshot.get('current_task_id') and snapshot['current_task_id'] != callback.task_id:
            return {'ack': True, 'deduped': False, 'ignored_reason': 'stale_task_callback'}

        if callback.event_type == CallbackEventType.START:
            self._upsert_parse_snapshot(
                doc_id=doc_id,
                kb_id=kb_id,
                algo_id=algo_id,
                status=DocStatus.WORKING,
                task_type=task_type,
                current_task_id=callback.task_id,
                started_at=now_ts(),
                queued_at=None,
                finished_at=None,
            )
            return {'ack': True, 'deduped': False, 'ignored_reason': None}

        final_status = callback.status
        failed_stage = None
        if final_status == DocStatus.FAILED:
            failed_stage = 'DELETE' if task_type == TaskType.DOC_DELETE else 'PARSE'

        self._upsert_parse_snapshot(
            doc_id=doc_id,
            kb_id=kb_id,
            algo_id=algo_id,
            status=final_status,
            task_type=task_type,
            current_task_id=callback.task_id,
            error_code=callback.error_code,
            error_msg=callback.error_msg,
            failed_stage=failed_stage,
            finished_at=now_ts(),
        )

        if task_type == TaskType.DOC_DELETE and final_status == DocStatus.DELETED:
            self._remove_kb_document(kb_id, doc_id)
        self._sync_doc_upload_status(doc_id)

        return {'ack': True, 'deduped': False, 'ignored_reason': None}

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
            latest_task_resp = self._parser_client.get_task(snapshot['current_task_id'])
            if latest_task_resp.code == 200:
                latest_task = latest_task_resp.data
        return {
            'doc': doc,
            'relation': relation,
            'snapshot': snapshot,
            'latest_task': latest_task,
            'relations': [item.get('relation') for item in matched_items],
            'snapshots': [item.get('snapshot') for item in matched_items if item.get('snapshot') is not None],
        }

    def list_tasks(self, status: Optional[List[str]], page: int, page_size: int):
        return self._parser_client.list_tasks(status=status, page=page, page_size=page_size)

    def get_task(self, task_id: str):
        return self._parser_client.get_task(task_id)

    def get_tasks_batch(self, task_ids: List[str]):
        items = []
        for task_id in task_ids:
            resp = self._parser_client.get_task(task_id)
            if resp.code == 200 and resp.data is not None:
                items.append(resp.data)
        return {'items': items}

    def cancel_task(self, task_id: str):
        return self._parser_client.cancel_task(task_id)

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

    def list_chunks(self, page: int = 1, page_size: int = 20):
        return {'items': [], 'total': 0, 'page': page, 'page_size': page_size}

    def health(self):
        return {
            'status': 'ok',
            'version': 'v1-mock',
            'deps': {'sql': True},
        }

    def list_kbs(self):
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            rows = session.query(Kb).order_by(Kb.updated_at.desc()).all()
            items = []
            for row in rows:
                items.append({
                    'kb_id': row.kb_id,
                    'display_name': row.display_name,
                    'description': row.description,
                    'doc_count': row.doc_count,
                    'status': row.status,
                    'owner_id': row.owner_id,
                    'meta': _from_json(row.meta),
                    'created_at': row.created_at,
                    'updated_at': row.updated_at,
                })
            return {'items': items}

    def create_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                  owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                  algo_id: str = '__default__'):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        binding = self._get_kb_algorithm(kb_id)
        if binding is not None and binding['algo_id'] != algo_id:
            raise DocServiceError(
                'E_STATE_CONFLICT', f'kb {kb_id} is already bound to algorithm {binding["algo_id"]}',
                {'kb_id': kb_id, 'bound_algo_id': binding['algo_id'], 'requested_algo_id': algo_id}
            )
        self._ensure_kb(kb_id, display_name=display_name, description=description, owner_id=owner_id, meta=meta)
        self._ensure_kb_algorithm(kb_id, algo_id)
        return {'kb_id': kb_id, 'status': KBStatus.ACTIVE.value}

    def delete_kb(self, kb_id: str):
        if not kb_id:
            raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            Rel = self._db_manager.get_table_orm_class(KB_DOCUMENTS_TABLE_INFO['name'])
            Snap = self._db_manager.get_table_orm_class(PARSE_STATE_TABLE_INFO['name'])
            kb_row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            if kb_row is None:
                raise DocServiceError('E_NOT_FOUND', f'kb not found: {kb_id}', {'kb_id': kb_id})
            states = (
                session.query(Snap)
                .join(Rel, sqlalchemy.and_(Snap.doc_id == Rel.doc_id, Snap.kb_id == Rel.kb_id))
                .filter(Rel.kb_id == kb_id, ~Snap.status.in_([DocStatus.DELETED.value, DocStatus.CANCELED.value]))
                .all()
            )
        task_ids = []
        for row in states:
            task_id, _ = self._enqueue_task(row.doc_id, row.kb_id, row.algo_id, TaskType.DOC_DELETE)
            task_ids.append(task_id)
        new_status = KBStatus.DELETING.value if task_ids else KBStatus.DELETED.value
        with self._db_manager.get_session() as session:
            Kb = self._db_manager.get_table_orm_class(KBS_TABLE_INFO['name'])
            kb_row = session.query(Kb).filter(Kb.kb_id == kb_id).first()
            kb_row.status = new_status
            kb_row.updated_at = now_ts()
            session.add(kb_row)
        return {'kb_id': kb_id, 'status': new_status, 'task_ids': task_ids}

    def delete_kbs(self, kb_ids: List[str]):
        if not kb_ids:
            raise DocServiceError('E_INVALID_PARAM', 'kb_ids is required', {'kb_ids': kb_ids})
        items = []
        for kb_id in kb_ids:
            items.append(self.delete_kb(kb_id))
        return {'items': items}
