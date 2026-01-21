from pydantic import BaseModel, Field, BeforeValidator
from typing import Dict, List, Optional, Any, Annotated
from enum import Enum
from uuid import uuid4
from datetime import datetime

class TransferParams(BaseModel):
    mode: Optional[str] = 'cp'  # cp or mv
    target_algo_id: str
    target_doc_id: str
    target_kb_id: str


EmptyTransfer = Annotated[TransferParams | None, BeforeValidator(lambda v: None if v == {} else v)]

class FileInfo(BaseModel):
    file_path: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    reparse_group: Optional[str] = None
    transformed_file_path: Optional[str] = None
    transfer_params: EmptyTransfer = None


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
    kb_id: Optional[str] = None
    file_infos: List[FileInfo]
    priority: Optional[int] = 0
    # NOTE: (db_info, feedback_url) is deprecated, will be removed in the future
    db_info: Optional[DBInfo] = None
    feedback_url: Optional[str] = None


class UpdateMetaRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    kb_id: Optional[str] = None
    file_infos: List[FileInfo]
    priority: Optional[int] = 0
    # NOTE: (db_info) is deprecated, will be removed in the future
    db_info: Optional[DBInfo] = None


class DeleteDocRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    algo_id: Optional[str] = '__default__'
    kb_id: Optional[str] = None
    doc_ids: List[str]
    priority: Optional[int] = 0
    # NOTE: (db_info) is deprecated, will be removed in the future
    db_info: Optional[DBInfo] = None


class CancelTaskRequest(BaseModel):
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
    DOC_TRANSFER = 'DOC_TRANSFER'


def _get_task_type_weight(task_type: str) -> int:
    '''get task type weight'''
    weight_map = {
        TaskType.DOC_DELETE.value: 10,
        TaskType.DOC_UPDATE_META.value: 30,
        TaskType.DOC_ADD.value: 100,
        TaskType.DOC_REPARSE.value: 100,
        TaskType.DOC_TRANSFER.value: 100,
    }
    return weight_map.get(task_type, 100)


def _calculate_task_score(task_type: str, user_priority: int) -> int:
    '''calculate task score'''
    type_weight = _get_task_type_weight(task_type)
    return type_weight * 10 - user_priority * 15


def _resolve_add_doc_task_type(request: AddDocRequest) -> str:  # noqa: C901
    new_file_ids = []
    reparse_file_ids = []
    transfer_mode = None
    target_algo_id = None
    target_kb_id = None

    for file_info in request.file_infos:
        if file_info.reparse_group is not None:
            reparse_file_ids.append(file_info.doc_id)
        else:
            new_file_ids.append(file_info.doc_id)
            if file_info.transfer_params:
                if target_algo_id is not None and target_algo_id != file_info.transfer_params.target_algo_id:
                    raise ValueError('transfer_params.target_algo_id must be the same for all files')
                if target_kb_id is not None and target_kb_id != file_info.transfer_params.target_kb_id:
                    raise ValueError('transfer_params.target_kb_id must be the same for all files')
                if transfer_mode is not None and transfer_mode != file_info.transfer_params.mode:
                    raise ValueError('transfer_params.mode must be the same for all files')
                if file_info.transfer_params.target_algo_id != request.algo_id:
                    raise ValueError('transfer_params.target_algo_id must be the same for all files')
                target_algo_id = file_info.transfer_params.target_algo_id
                target_kb_id = file_info.transfer_params.target_kb_id
                transfer_mode = file_info.transfer_params.mode
                if transfer_mode not in ['cp', 'mv']:
                    raise ValueError('transfer_params.mode must be one of [cp, mv]')

    if new_file_ids and reparse_file_ids:
        raise ValueError('new_file_ids and reparse_file_ids cannot be specified at the same time')
    if transfer_mode:
        return TaskType.DOC_TRANSFER.value
    if new_file_ids:
        return TaskType.DOC_ADD.value
    if reparse_file_ids:
        return TaskType.DOC_REPARSE.value
    raise ValueError('no input files or reparse group specified')


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
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'default': TaskStatus.WAITING.value,
         'comment': 'Task status: WAITING, WORKING'},
        {'name': 'worker_id', 'data_type': 'string', 'nullable': True,
         'comment': 'Worker ID holding the lease'},
        {'name': 'lease_expires_at', 'data_type': 'datetime', 'nullable': True,
         'comment': 'Lease expiration time for in-progress task'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Creation time (auto-generated)', 'default': datetime.now},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Last update time (auto-generated)', 'default': datetime.now},
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
         'comment': 'Task status: WAITING, WORKING, CANCEL_REQUESTED, CANCELED, FINISHED, FAILED'},
        {'name': 'finished_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Finish time (set when processing completes)', 'default': datetime.now},
        {'name': 'error_code', 'data_type': 'string', 'nullable': True, 'default': '200',
         'comment': 'Error code (varchar64)'},
        {'name': 'error_msg', 'data_type': 'string', 'nullable': True, 'default': 'success',
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
        {'name': 'display_name', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm display name'},
        {'name': 'description', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm description'},
        {'name': 'info_pickle', 'data_type': 'string', 'nullable': False,
         'comment': 'Algorithm info from pickle string'},
        {'name': 'created_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Creation time (auto-generated)', 'default': datetime.now},
        {'name': 'updated_at', 'data_type': 'datetime', 'nullable': False,
         'comment': 'Last update time (set when updating)', 'default': datetime.now},
    ]
}
