from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field, model_validator
from ..parsing_service.base import TaskType


class DocStatus(str, Enum):
    WAITING = 'WAITING'
    WORKING = 'WORKING'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'
    DELETING = 'DELETING'
    DELETED = 'DELETED'


class NodeGroupParseStatus(str, Enum):
    PENDING = 'PENDING'
    WORKING = 'WORKING'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'


class KBStatus(str, Enum):
    ACTIVE = 'ACTIVE'
    DELETING = 'DELETING'
    DELETED = 'DELETED'


KbStatus = KBStatus


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
    'E_UPSTREAM_ERROR': 502,
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

    @model_validator(mode='after')
    def validate_file_path(self):
        if not self.file_path or not self.file_path.strip():
            raise ValueError('file_path is required')
        return self


class DocItemsRequest(BaseModel):
    items: List[AddFileItem]
    kb_id: str = '__default__'
    algo_id: Optional[str] = None
    algo_ids: Optional[List[str]] = None
    source_type: Optional[SourceType] = None
    idempotency_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_items(self):
        if not self.items:
            raise ValueError('items is required')
        if self.algo_id is not None and self.algo_ids is not None:
            raise ValueError('algo_id and algo_ids cannot both be provided')
        if self.algo_id is None and self.algo_ids is None:
            raise ValueError('one of algo_id or algo_ids is required')
        if self.algo_ids is not None and len(set(self.algo_ids)) != len(self.algo_ids):
            raise ValueError('algo_ids must not contain duplicates')
        return self

    @property
    def effective_algo_ids(self) -> List[str]:
        return self.algo_ids if self.algo_ids is not None else [self.algo_id or '__default__']


AddRequest = DocItemsRequest
UploadRequest = DocItemsRequest


class _DocMutationRequest(BaseModel):
    doc_ids: List[str]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_doc_ids(self):
        if not self.doc_ids:
            raise ValueError('doc_ids is required')
        return self


class ReparseRequest(_DocMutationRequest):
    reparse_group: Optional[str] = None  # None or 'all' = reparse all ng; ng_id = reparse single ng


class DeleteRequest(_DocMutationRequest):
    pass


class TransferItem(BaseModel):
    doc_id: str
    target_doc_id: str
    kb_id: str = Field(default='__default__', validation_alias=AliasChoices('kb_id', 'source_kb_id'))
    algo_id: str = Field(default='__default__', validation_alias=AliasChoices('algo_id', 'source_algo_id'))
    target_kb_id: str
    target_algo_id: str
    target_metadata: Optional[Dict[str, Any]] = None
    target_filename: Optional[str] = None
    target_file_path: Optional[str] = None
    mode: str = 'copy'

    @model_validator(mode='before')
    @classmethod
    def normalize_source_fields(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if 'kb_id' not in normalized and 'source_kb_id' in normalized:
            normalized['kb_id'] = normalized['source_kb_id']
        if 'algo_id' not in normalized and 'source_algo_id' in normalized:
            normalized['algo_id'] = normalized['source_algo_id']
        return normalized

    @property
    def source_kb_id(self) -> str:
        return self.kb_id

    @property
    def source_algo_id(self) -> str:
        return self.algo_id


class TransferRequest(BaseModel):
    items: List[TransferItem]
    idempotency_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_items(self):
        if not self.items:
            raise ValueError('items is required')
        return self


class MetadataPatchItem(BaseModel):
    doc_id: str
    patch: Dict[str, Any] = Field(default_factory=dict)


class MetadataPatchRequest(BaseModel):
    items: List[MetadataPatchItem]
    kb_id: str = '__default__'
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_items(self):
        if not self.items:
            raise ValueError('items is required')
        return self


class KbDeleteBatchRequest(BaseModel):
    kb_ids: List[str]
    idempotency_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_kb_ids(self):
        if not self.kb_ids:
            raise ValueError('kb_ids is required')
        return self


class AlgorithmInfoRequest(BaseModel):
    algo_id: str


class TaskInfoRequest(BaseModel):
    task_id: str


class TaskBatchRequest(BaseModel):
    task_ids: List[str]

    @model_validator(mode='after')
    def validate_task_ids(self):
        if not self.task_ids:
            raise ValueError('task_ids is required')
        return self


class TaskCancelRequest(BaseModel):
    task_id: str
    idempotency_key: Optional[str] = None


class TaskCallbackPayload(BaseModel):
    '''Parser -> DocService callback payload.

    ``status`` is the document-level state after this callback (what
    ``parse_state.status`` should transition to); ``task_status`` is the outcome
    of the individual task execution (SUCCESS / FAILED / CANCELED). They can
    diverge during cancel flows where the task is CANCELED but the doc may
    remain in SUCCESS on a previous version.
    '''
    callback_id: Optional[str] = None
    task_id: Optional[str] = None
    event_type: Optional[CallbackEventType] = None
    status: Optional[DocStatus] = None
    task_status: Optional[DocStatus] = None
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    task_type: Optional[TaskType] = None
    doc_id: Optional[str] = None
    kb_id: Optional[str] = None
    algo_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class KbRequest(BaseModel):
    kb_id: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    owner_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    algo_id: str = '__default__'
    idempotency_key: Optional[str] = None


KbCreateRequest = KbRequest
KbUpdateRequest = KbRequest


class KbBatchQueryRequest(BaseModel):
    kb_ids: List[str]

    @model_validator(mode='after')
    def validate_kb_ids(self):
        if not self.kb_ids:
            raise ValueError('kb_ids is required')
        return self


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


DOC_SERVICE_TASKS_TABLE_INFO = {
    'name': 'lazyllm_doc_service_tasks',
    'comment': 'Doc service task history table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'task_id', 'data_type': 'string', 'nullable': False, 'comment': 'Task ID'},
        {'name': 'task_type', 'data_type': 'string', 'nullable': False, 'comment': 'Task type'},
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'comment': 'Document ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'algo_id', 'data_type': 'string', 'nullable': False, 'comment': 'Algorithm ID'},
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'comment': 'Current task status'},
        {'name': 'message', 'data_type': 'text', 'nullable': True, 'comment': 'Task payload in JSON string'},
        {'name': 'error_code', 'data_type': 'string', 'nullable': True, 'comment': 'Error code'},
        {'name': 'error_msg', 'data_type': 'text', 'nullable': True, 'comment': 'Error message'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
        {'name': 'started_at', 'data_type': 'datetime', 'nullable': True, 'comment': 'Started time'},
        {'name': 'finished_at', 'data_type': 'datetime', 'nullable': True, 'comment': 'Finished time'},
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


DOC_PATH_LOCKS_TABLE_INFO = {
    'name': 'lazyllm_doc_path_locks',
    'comment': 'Transient lock table for serializing document path writes',
    'columns': [
        {'name': 'path', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
         'comment': 'Absolute file path'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
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
    'comment': 'Latest parse state snapshot table',
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

DOC_NODE_GROUP_STATUS_TABLE_INFO = {
    'name': 'lazyllm_doc_node_group_status',
    'comment': 'Per-document per-node-group parse status table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'comment': 'Document ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'node_group_id', 'data_type': 'string', 'nullable': False,
         'comment': 'Node group ID (logical reference to node group definition)'},
        {'name': 'file_path', 'data_type': 'string', 'nullable': True,
         'comment': 'Document file path (cached for reparse without querying store)'},
        {'name': 'status', 'data_type': 'string', 'nullable': False,
         'comment': 'PENDING / WORKING / SUCCESS / FAILED'},
        {'name': 'error_msg', 'data_type': 'text', 'nullable': True, 'comment': 'Error message on failure'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Created time'},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False, 'default': datetime.now,
         'comment': 'Updated time'},
    ],
}
