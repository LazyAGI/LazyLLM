from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from ..parsing_service.base import TaskType


class DocStatus(str, Enum):
    WAITING = 'WAITING'
    WORKING = 'WORKING'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'
    DELETING = 'DELETING'
    DELETED = 'DELETED'


class KBStatus(str, Enum):
    ACTIVE = 'ACTIVE'
    DELETING = 'DELETING'
    DELETED = 'DELETED'


class SourceType(str, Enum):
    API = 'API'
    SCAN = 'SCAN'
    TEMP = 'TEMP'
    EXTERNAL = 'EXTERNAL'


class CallbackEventType(str, Enum):
    START = 'START'
    FINISH = 'FINISH'


BIZ_HTTP_CODE = {
    'E_INVALID_PARAM': 400,
    'E_NOT_FOUND': 404,
    'E_STATE_CONFLICT': 409,
    'E_IDEMPOTENCY_CONFLICT': 409,
    'E_IDEMPOTENCY_IN_PROGRESS': 409,
}


class DocServiceError(Exception):
    def __init__(self, biz_code: str, msg: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(biz_code, msg, data)
        self.biz_code = biz_code
        self.msg = msg
        self.data = data or {}

    @property
    def http_status(self):
        return BIZ_HTTP_CODE.get(self.biz_code, 500)


class TaskCreateRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: TaskType
    doc_id: str
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    callback_url: Optional[str] = None


class TaskCallbackRequest(BaseModel):
    callback_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    event_type: CallbackEventType
    status: DocStatus
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class AddFileItem(BaseModel):
    file_path: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddRequest(BaseModel):
    items: List[AddFileItem]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    source_type: SourceType = SourceType.EXTERNAL
    idempotency_key: Optional[str] = None


class UploadRequest(BaseModel):
    items: List[AddFileItem]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    source_type: SourceType = SourceType.API
    idempotency_key: Optional[str] = None


class ReparseRequest(BaseModel):
    doc_ids: List[str]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None


class DeleteRequest(BaseModel):
    doc_ids: List[str]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None


class TransferItem(BaseModel):
    doc_id: str
    source_kb_id: str = '__default__'
    source_algo_id: str = '__default__'
    target_kb_id: str
    target_algo_id: str
    mode: str = 'copy'


class TransferRequest(BaseModel):
    items: List[TransferItem]
    idempotency_key: Optional[str] = None


class MetadataPatchItem(BaseModel):
    doc_id: str
    patch: Dict[str, Any] = Field(default_factory=dict)


class MetadataPatchRequest(BaseModel):
    items: List[MetadataPatchItem]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None


IDEMPOTENCY_RECORDS_TABLE_INFO = {
    'name': 'lazyllm_idempotency_records',
    'comment': 'Idempotency replay records',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'endpoint', 'data_type': 'string', 'nullable': False, 'comment': 'Endpoint name'},
        {'name': 'idempotency_key', 'data_type': 'string', 'nullable': False, 'comment': 'Idempotency key'},
        {'name': 'req_hash', 'data_type': 'string', 'nullable': False, 'comment': 'Request hash'},
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'comment': 'Processing status'},
        {'name': 'response_json', 'data_type': 'text', 'nullable': True, 'comment': 'Response json'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


CALLBACK_RECORDS_TABLE_INFO = {
    'name': 'lazyllm_callback_records',
    'comment': 'Processed callback records',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'callback_id', 'data_type': 'string', 'nullable': False, 'comment': 'Callback ID'},
        {'name': 'task_id', 'data_type': 'string', 'nullable': False, 'comment': 'Task ID'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
    ],
}


DOCUMENTS_TABLE_INFO = {
    'name': 'lazyllm_documents',
    'comment': 'Document metadata table',
    'columns': [
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
         'comment': 'Document ID'},
        {'name': 'filename', 'data_type': 'string', 'nullable': False, 'comment': 'Filename'},
        {'name': 'path', 'data_type': 'string', 'nullable': False, 'comment': 'Absolute file path'},
        {'name': 'meta', 'data_type': 'text', 'nullable': True, 'comment': 'Document metadata in JSON string'},
        {'name': 'upload_status', 'data_type': 'string', 'nullable': False, 'comment': 'Document upload status'},
        {'name': 'source_type', 'data_type': 'string', 'nullable': False, 'comment': 'Source type'},
        {'name': 'file_type', 'data_type': 'string', 'nullable': True, 'comment': 'File type suffix'},
        {'name': 'content_hash', 'data_type': 'string', 'nullable': True, 'comment': 'Content hash'},
        {'name': 'size_bytes', 'data_type': 'integer', 'nullable': True, 'comment': 'File size in bytes'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


KBS_TABLE_INFO = {
    'name': 'lazyllm_knowledge_bases',
    'comment': 'Knowledge base table',
    'columns': [
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
         'comment': 'Knowledge base ID'},
        {'name': 'display_name', 'data_type': 'string', 'nullable': True, 'comment': 'Display name'},
        {'name': 'description', 'data_type': 'string', 'nullable': True, 'comment': 'Description'},
        {'name': 'doc_count', 'data_type': 'integer', 'nullable': False, 'default': 0, 'comment': 'Document count'},
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'default': KBStatus.ACTIVE.value,
         'comment': 'KB status'},
        {'name': 'owner_id', 'data_type': 'string', 'nullable': True, 'comment': 'Owner ID'},
        {'name': 'meta', 'data_type': 'text', 'nullable': True, 'comment': 'KB metadata in JSON string'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


KB_DOCUMENTS_TABLE_INFO = {
    'name': 'lazyllm_kb_documents',
    'comment': 'KB and document binding table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'comment': 'Document ID'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


KB_ALGORITHM_TABLE_INFO = {
    'name': 'lazyllm_kb_algorithm',
    'comment': 'KB and algorithm binding table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'algo_id', 'data_type': 'string', 'nullable': False, 'comment': 'Algorithm ID'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


PARSE_STATE_TABLE_INFO = {
    'name': 'lazyllm_doc_parse_state',
    'comment': 'Latest parse state table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'comment': 'Document ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'algo_id', 'data_type': 'string', 'nullable': False, 'comment': 'Algorithm ID'},
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'comment': 'Current parse status'},
        {'name': 'current_task_id', 'data_type': 'string', 'nullable': True, 'comment': 'Current task ID'},
        {'name': 'task_type', 'data_type': 'string', 'nullable': True, 'comment': 'Current task type'},
        {'name': 'idempotency_key', 'data_type': 'string', 'nullable': True, 'comment': 'Idempotency key'},
        {'name': 'priority', 'data_type': 'integer', 'nullable': False, 'default': 0, 'comment': 'Task priority'},
        {'name': 'task_score', 'data_type': 'integer', 'nullable': True, 'comment': 'Task score'},
        {'name': 'retry_count', 'data_type': 'integer', 'nullable': False, 'default': 0, 'comment': 'Retry count'},
        {'name': 'max_retry', 'data_type': 'integer', 'nullable': False, 'default': 3, 'comment': 'Max retry'},
        {'name': 'lease_owner', 'data_type': 'string', 'nullable': True, 'comment': 'Lease owner'},
        {'name': 'lease_until', 'data_type': 'datetime', 'nullable': True, 'comment': 'Lease deadline'},
        {'name': 'last_error_code', 'data_type': 'string', 'nullable': True, 'comment': 'Last error code'},
        {'name': 'last_error_msg', 'data_type': 'text', 'nullable': True, 'comment': 'Last error message'},
        {'name': 'failed_stage', 'data_type': 'string', 'nullable': True, 'comment': 'Failure stage'},
        {'name': 'queued_at', 'data_type': 'datetime', 'nullable': True, 'comment': 'Queued time'},
        {'name': 'started_at', 'data_type': 'datetime', 'nullable': True, 'comment': 'Started time'},
        {'name': 'finished_at', 'data_type': 'datetime', 'nullable': True, 'comment': 'Finished time'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}


def now_ts() -> datetime:
    return datetime.now()
