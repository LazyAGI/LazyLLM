from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import uuid4
from datetime import datetime

from ..store.store_base import DEFAULT_KB_ID


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
    priority: Optional[int] = 0
    # NOTE: (db_info, feedback_url) is deprecated, will be removed in the future
    db_info: Optional[DBInfo] = None
    feedback_url: Optional[str] = None


class UpdateMetaRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    file_infos: List[FileInfo]
    priority: Optional[int] = 0
    # NOTE: (db_info) is deprecated, will be removed in the future
    db_info: Optional[DBInfo] = None


class DeleteDocRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    kb_id: Optional[str] = DEFAULT_KB_ID
    doc_ids: List[str]
    priority: Optional[int] = 0
    # NOTE: (db_info) is deprecated, will be removed in the future
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


def get_task_type_weight(task_type: str) -> int:
    '''get task type weight'''
    weight_map = {
        TaskType.DOC_DELETE.value: 10,
        TaskType.DOC_UPDATE_META.value: 30,
        TaskType.DOC_ADD.value: 100,
        TaskType.DOC_REPARSE.value: 100,
    }
    return weight_map.get(task_type, 100)


def calculate_task_score(task_type: str, user_priority: int) -> int:
    '''calculate task score'''
    type_weight = get_task_type_weight(task_type)
    return type_weight * 10 - user_priority * 15


# Waiting task queue table
WAITING_TASK_QUEUE_TABLE_INFO = {
    'name': 'lazyllm_waiting_task_queue',
    'comment': 'Waiting task queue table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'task_id', 'data_type': 'string', 'nullable': False,
         'comment': 'Task ID (uuid4)'},
        {'name': 'task_type', 'data_type': 'string', 'nullable': False,
         'comment': 'Task type: DOC_ADD, DOC_DELETE, DOC_UPDATE_META, DOC_REPARSE'},
        {'name': 'user_priority', 'data_type': 'integer', 'nullable': False, 'default': 0,
         'comment': 'User-specified priority (0-10, higher is more important)'},
        {'name': 'task_score', 'data_type': 'integer', 'nullable': False, 'default': 1000,
         'comment': 'Calculated task score (for sorting, lower score is higher priority)'},
        {'name': 'message', 'data_type': 'string', 'nullable': False,
         'comment': 'Task message (json string, serialized from request body)'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Creation time (auto-generated)', 'default': datetime.now()},
    ]
}

# Finished task queue table
FINISHED_TASK_QUEUE_TABLE_INFO = {
    'name': 'lazyllm_finished_task_queue',
    'comment': 'Finished task queue table',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'task_id', 'data_type': 'string', 'nullable': False,
         'comment': 'Task ID (uuid4)'},
        {'name': 'task_type', 'data_type': 'string', 'nullable': False,
         'comment': 'Task type: DOC_ADD, DOC_DELETE, DOC_UPDATE_META, DOC_REPARSE'},
        {'name': 'task_status', 'data_type': 'string', 'nullable': False,
         'comment': 'Task message (json string, serialized from request body)'},
        {'name': 'finished_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Finish time (set when processing completes)', 'default': datetime.now()},
        {'name': 'error_code', 'data_type': 'string', 'nullable': True,
         'comment': 'Error code (varchar64)'},
        {'name': 'error_msg', 'data_type': 'string', 'nullable': True,
         'comment': 'Error message (up to 512 chars)'},
    ]
}

# Algorithm table
ALGORITHM_TABLE_INFO = {
    'name': 'lazyllm_algorithm',
    'comment': 'LazyLLM algorithm registration table',
    'columns': [
        {'name': 'id', 'data_type': 'string', 'nullable': False, 'is_primary_key': True,
         'comment': 'Algorithm ID'},
        {'name': 'name', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm name'},
        {'name': 'description', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm description'},
        {'name': 'hash_key', 'data_type': 'string', 'nullable': True,
         'comment': 'Algorithm hash key'},
        {'name': 'info_pickle', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm info from pickle string'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Creation time (auto-generated)', 'default': datetime.now()},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Last update time (set when updating)', 'default': datetime.now()},
        {'name': 'is_active', 'data_type': 'boolean', 'nullable': False,
         'comment': 'Whether the algorithm is active', 'default': True},
    ]
}
