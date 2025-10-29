from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import uuid4


class FileInfo(BaseModel):
    file_path: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    reparse_group: Optional[str] = None


class DBInfo(BaseModel):
    db_type: str
    db_name: str
    user: str
    password: str
    host: str
    port: int
    table_name: str
    options_str: Optional[str] = None


class AddDocRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    file_infos: List[FileInfo]
    db_info: Optional[DBInfo] = None
    feedback_url: Optional[str] = None


class UpdateMetaRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    file_infos: List[FileInfo]
    db_info: Optional[DBInfo] = None


class DeleteDocRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    kb_id: str
    doc_ids: List[str]
    db_info: Optional[DBInfo] = None


class CancelDocRequest(BaseModel):
    task_id: str


class TaskStatus(str, Enum):
    WAITING = 'WAITING'
    WORKING = 'WORKING'
    CANCEL_REQUESTED = 'CANCEL_REQUESTED'
    CANCELED = 'CANCELED'
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


class TaskType(str, Enum):
    DOC_ADD = 'DOC_ADD'
    DOC_DELETE = 'DOC_DELETE'
    DOC_UPDATE_META = 'DOC_UPDATE_META'
    DOC_REPARSE = 'DOC_REPARSE'


class TaskCancelled(Exception):
    pass

TABLES_INFO = {
    'tables': [
        # lazyllm_doc_task_detail - Task detail record table
        {
            'name': 'lazyllm_doc_task_detail',
            'comment': 'Task detail record table',
            'columns': [
                {'name': 'task_id', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
                 'comment': 'Task ID (uuid4)'},
                {'name': 'task_type', 'data_type': 'string', 'nullable': False,
                 'comment': 'Task type: DOC_ADD, DOC_DELETE, DOC_UPDATE_META, DOC_REPARSE'},
                {'name': 'algo_id', 'data_type': 'string', 'nullable': False,
                 'comment': 'Algorithm ID (foreign key)'},
                {'name': 'task_status', 'data_type': 'string', 'nullable': False, 'default': 'WAITING',
                 'comment': 'Task status: WAITING / WORKING / CANCEL_REQUESTED / CANCELED / FINISHED / FAILED'},
                {'name': 'priority', 'data_type': 'integer', 'nullable': True, 'default': 0,
                 'comment': 'Task priority'},
                {'name': 'payload', 'data_type': 'string', 'nullable': False,
                 'comment': 'Request body payload (json string)'},
                {'name': 'idempotency_key', 'data_type': 'string', 'nullable': False,
                 'comment': 'Idempotency key, encoded from algo_id + type + file_infos'},
                {'name': 'create_at', 'data_type': 'datetime', 'nullable': False,
                 'comment': 'Creation time (auto-generated)'},
                {'name': 'started_at', 'data_type': 'datetime', 'nullable': True,
                 'comment': 'Start time (set when processing begins)'},
                {'name': 'finished_at', 'data_type': 'datetime', 'nullable': True,
                 'comment': 'Finish time (set when processing completes)'},
                {'name': 'lease_owner', 'data_type': 'string', 'nullable': True,
                 'comment': 'Lease owner identifier (pod_id or hostname)'},
                {'name': 'lease_expire_ts', 'data_type': 'integer', 'nullable': True,
                 'comment': 'Lease expiration timestamp (currently unused)'},
                {'name': 'retries', 'data_type': 'integer', 'nullable': True, 'default': 0,
                 'comment': 'Retry count, incremented when worker retries'},
                {'name': 'error_code', 'data_type': 'string', 'nullable': True,
                 'comment': 'Error code (varchar64)'},
                {'name': 'error_msg', 'data_type': 'string', 'nullable': True,
                 'comment': 'Error message (up to 512 chars)'},
            ]
        },
        # lazyllm_doc_task_queue - Task queue table
        {
            'name': 'lazyllm_doc_task_queue',
            'comment': 'Task queue table (records tasks that have not started)',
            'columns': [
                {'name': 'task_id', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
                 'comment': 'Task ID (uuid4)'},
                {'name': 'task_type', 'data_type': 'string', 'nullable': False,
                 'comment': 'Task type'},
                {'name': 'priority', 'data_type': 'integer', 'nullable': True, 'default': 0,
                 'comment': 'Task priority'},
                {'name': 'create_at', 'data_type': 'datetime', 'nullable': False,
                 'comment': 'Creation time (auto-generated)'},
            ]
        },
        # lazyllm_doc_task_record - Task audit record table
        {
            'name': 'lazyllm_doc_task_record',
            'comment': 'Task status transition audit table',
            'columns': [
                {'name': 'record_id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
                 'comment': 'Task record ID (auto-increment)'},
                {'name': 'task_id', 'data_type': 'string', 'nullable': False,
                 'comment': 'Task ID (uuid4)'},
                {'name': 'from_status', 'data_type': 'string', 'nullable': False,
                 'comment': 'Previous status: WAITING / WORKING / CANCEL_REQUESTED / CANCELED / FINISHED / FAILED'},
                {'name': 'to_status', 'data_type': 'string', 'nullable': False,
                 'comment': 'New status: WAITING / WORKING / CANCEL_REQUESTED / CANCELED / FINISHED / FAILED'},
                {'name': 'create_at', 'data_type': 'datetime', 'nullable': False,
                 'comment': 'Creation time (auto-generated)'},
                {'name': 'attempt_seq', 'data_type': 'integer', 'nullable': True, 'default': 0,
                 'comment': 'Attempt sequence number'},
                {'name': 'reason', 'data_type': 'string', 'nullable': True,
                 'comment': 'Reason or description for the status transition'},
            ]
        }
    ]
}
