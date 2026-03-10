from __future__ import annotations
'''
Mock parsing execution service used in phase-1 refactor validation.

Note:
This is intentionally isolated from `lazyllm.tools.rag.parsing_service` so that
DocService API contract and state machine can be validated without requiring the
full parser runtime and algorithm registry.
'''

import json
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

import cloudpickle
import requests

from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule, UrlModule, once_wrapper
from lazyllm.thirdparty import fastapi

from ..utils import BaseResponse, _get_default_db_config, _orm_to_dict, ensure_call_endpoint
from ...sql import SqlManager
from ..parsing_service.base import ALGORITHM_TABLE_INFO
from .base import (
    CallbackEventType,
    TaskCallbackRequest,
    TaskCreateRequest,
    DocStatus,
    now_ts,
)

PARSER_TASK_TABLE_INFO = {
    'name': 'lazyllm_parse_tasks',
    'comment': 'Parse task table for mock parser service',
    'columns': [
        {'name': 'id', 'data_type': 'integer', 'nullable': False, 'is_primary_key': True,
         'comment': 'Auto increment ID'},
        {'name': 'task_id', 'data_type': 'string', 'nullable': False, 'comment': 'Task ID'},
        {'name': 'task_type', 'data_type': 'string', 'nullable': False, 'comment': 'Task type'},
        {'name': 'doc_id', 'data_type': 'string', 'nullable': False, 'comment': 'Document ID'},
        {'name': 'kb_id', 'data_type': 'string', 'nullable': False, 'comment': 'Knowledge base ID'},
        {'name': 'algo_id', 'data_type': 'string', 'nullable': False, 'comment': 'Algorithm ID'},
        {'name': 'status', 'data_type': 'string', 'nullable': False, 'comment': 'Task status'},
        {'name': 'priority', 'data_type': 'integer', 'nullable': False, 'default': 0, 'comment': 'Task priority'},
        {'name': 'message', 'data_type': 'text', 'nullable': False, 'comment': 'Task payload'},
        {'name': 'callback_url', 'data_type': 'string', 'nullable': True, 'comment': 'Callback URL'},
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


class ParsingTaskServer(ModuleBase):
    class _Impl:
        def __init__(
            self,
            db_config: Optional[Dict[str, Any]] = None,
            poll_interval: float = 0.05,
            callback_func: Optional[Callable[[TaskCallbackRequest], None]] = None,
        ):
            self._db_config = db_config or _get_default_db_config('doc_service_parser')
            self._poll_interval = poll_interval
            self._db_manager = None
            self._task_thread = None
            self._shutdown = False
            self._callback_func = callback_func

        @once_wrapper(reset_on_pickle=True)
        def _lazy_init(self):
            self._db_manager = SqlManager(
                **self._db_config,
                tables_info_dict={'tables': [PARSER_TASK_TABLE_INFO, ALGORITHM_TABLE_INFO]},
            )
            self._ensure_indexes()
            self._upsert_default_algorithm()
            self._shutdown = False
            self._task_thread = threading.Thread(target=self._task_worker, daemon=True)
            self._task_thread.start()

        def stop(self):
            self._shutdown = True
            if self._task_thread and self._task_thread.is_alive():
                self._task_thread.join(timeout=2)

        def _ensure_indexes(self):
            stmts = [
                'CREATE UNIQUE INDEX IF NOT EXISTS uq_parse_tasks_task_id ON lazyllm_parse_tasks(task_id)',
                'CREATE INDEX IF NOT EXISTS idx_parse_tasks_status ON lazyllm_parse_tasks(status, updated_at)',
                'CREATE INDEX IF NOT EXISTS idx_parse_tasks_doc ON lazyllm_parse_tasks(doc_id, kb_id, algo_id)',
            ]
            for stmt in stmts:
                self._db_manager.execute_commit(stmt)

        def _upsert_default_algorithm(self):
            default_group_info = [
                {'name': 'CoarseChunk', 'type': 'chunk', 'display_name': 'Coarse Chunk'},
                {'name': 'line', 'type': 'chunk', 'display_name': 'Line Chunk'},
            ]
            default_info = {
                'store': None,
                'reader': None,
                'node_groups': {
                    item['name']: {'group_type': item['type'], 'display_name': item['display_name']}
                    for item in default_group_info
                },
                'schema_extractor': None,
            }
            with self._db_manager.get_session() as session:
                Algo = self._db_manager.get_table_orm_class(ALGORITHM_TABLE_INFO['name'])
                row = session.query(Algo).filter(Algo.id == '__default__').first()
                if row is None:
                    session.add(
                        Algo(
                            id='__default__',
                            display_name='Default',
                            description='Default mock parsing algorithm',
                            info_pickle=cloudpickle.dumps(default_info),
                            created_at=now_ts(),
                            updated_at=now_ts(),
                        )
                    )

        def register_callback(self, callback_func: Callable[[TaskCallbackRequest], None]):
            self._callback_func = callback_func

        def _load_task(self, task_id: str):
            with self._db_manager.get_session() as session:
                Task = self._db_manager.get_table_orm_class(PARSER_TASK_TABLE_INFO['name'])
                row = session.query(Task).filter(Task.task_id == task_id).first()
                return _orm_to_dict(row) if row else None

        def _emit_callback(self, callback_payload: TaskCallbackRequest, callback_url: Optional[str]):
            if self._callback_func:
                self._callback_func(callback_payload)
                return
            if callback_url:
                response = requests.post(callback_url, json=callback_payload.model_dump(), timeout=5)
                if response.status_code >= 400:
                    raise RuntimeError(f'callback failed: {response.status_code} {response.text}')

        def _update_task(self, task_id: str, **fields):
            with self._db_manager.get_session() as session:
                Task = self._db_manager.get_table_orm_class(PARSER_TASK_TABLE_INFO['name'])
                row = session.query(Task).filter(Task.task_id == task_id).first()
                if row is None:
                    return None
                for key, value in fields.items():
                    setattr(row, key, value)
                row.updated_at = now_ts()
                session.add(row)
                return _orm_to_dict(row)

        def _task_worker(self):
            while not self._shutdown:
                try:
                    waiting_task = None
                    with self._db_manager.get_session() as session:
                        Task = self._db_manager.get_table_orm_class(PARSER_TASK_TABLE_INFO['name'])
                        waiting_task = (
                            session.query(Task)
                            .filter(Task.status == DocStatus.WAITING.value)
                            .order_by(Task.priority.desc(), Task.created_at.asc())
                            .first()
                        )
                        if waiting_task is not None:
                            waiting_task.status = DocStatus.WORKING.value
                            waiting_task.started_at = now_ts()
                            waiting_task.updated_at = now_ts()
                            session.add(waiting_task)
                            waiting_task = _orm_to_dict(waiting_task)
                    if waiting_task is None:
                        time.sleep(self._poll_interval)
                        continue

                    start_callback = TaskCallbackRequest(
                        task_id=waiting_task['task_id'],
                        event_type=CallbackEventType.START,
                        status=DocStatus.WORKING,
                        payload={
                            'task_type': waiting_task['task_type'],
                            'doc_id': waiting_task['doc_id'],
                            'kb_id': waiting_task['kb_id'],
                            'algo_id': waiting_task['algo_id'],
                        },
                    )
                    self._emit_callback(start_callback, waiting_task.get('callback_url'))

                    # Mock workload
                    time.sleep(self._poll_interval)
                    final_status = (
                        DocStatus.DELETED.value
                        if waiting_task['task_type'] == 'DOC_DELETE'
                        else DocStatus.SUCCESS.value
                    )
                    done = self._update_task(
                        waiting_task['task_id'],
                        status=final_status,
                        finished_at=now_ts(),
                        error_code=None,
                        error_msg=None,
                    )
                    finish_callback = TaskCallbackRequest(
                        task_id=waiting_task['task_id'],
                        event_type=CallbackEventType.FINISH,
                        status=DocStatus(final_status),
                        error_code=None,
                        error_msg=None,
                        payload={
                            'task_type': waiting_task['task_type'],
                            'doc_id': waiting_task['doc_id'],
                            'kb_id': waiting_task['kb_id'],
                            'algo_id': waiting_task['algo_id'],
                            'result': done,
                        },
                    )
                    self._emit_callback(finish_callback, waiting_task.get('callback_url'))
                except Exception as exc:
                    LOG.error(f'[ParsingTaskServer] worker loop error: {exc}, {traceback.format_exc()}')
                    time.sleep(self._poll_interval)

        @app.post('/v1/internal/tasks/create')
        def create_task(self, request: TaskCreateRequest):
            self._lazy_init()
            now = now_ts()
            payload = request.model_dump(mode='json')
            with self._db_manager.get_session() as session:
                Task = self._db_manager.get_table_orm_class(PARSER_TASK_TABLE_INFO['name'])
                exists = session.query(Task).filter(Task.task_id == request.task_id).first()
                if exists is not None:
                    return BaseResponse(code=200, msg='success', data={'task': _orm_to_dict(exists), 'deduped': True})
                session.add(
                    Task(
                        task_id=request.task_id,
                        task_type=request.task_type.value,
                        doc_id=request.doc_id,
                        kb_id=request.kb_id,
                        algo_id=request.algo_id,
                        status=DocStatus.WAITING.value,
                        priority=request.priority,
                        message=json.dumps(payload, ensure_ascii=False),
                        callback_url=request.callback_url,
                        error_code=None,
                        error_msg=None,
                        created_at=now,
                        updated_at=now,
                        started_at=None,
                        finished_at=None,
                    )
                )
            task = self._load_task(request.task_id)
            return BaseResponse(code=200, msg='success', data={'task': task, 'deduped': False})

        @app.post('/v1/internal/tasks/cancel')
        def cancel_task(self, request: Dict[str, str]):
            self._lazy_init()
            task_id = request.get('task_id')
            if not task_id:
                raise fastapi.HTTPException(status_code=400, detail='task_id is required')
            task = self._load_task(task_id)
            if not task:
                return BaseResponse(code=404, msg='task not found', data={'task_id': task_id, 'cancel_status': False})
            if task['status'] != DocStatus.WAITING.value:
                return BaseResponse(
                    code=409,
                    msg='task cannot be canceled',
                    data={'task_id': task_id, 'cancel_status': False, 'status': task['status']},
                )
            task = self._update_task(task_id, status=DocStatus.CANCELED.value, finished_at=now_ts())
            callback = TaskCallbackRequest(
                task_id=task_id,
                event_type=CallbackEventType.FINISH,
                status=DocStatus.CANCELED,
                payload={
                    'task_type': task.get('task_type'),
                    'doc_id': task.get('doc_id'),
                    'kb_id': task.get('kb_id'),
                    'algo_id': task.get('algo_id'),
                },
            )
            try:
                self._emit_callback(callback, task.get('callback_url'))
            except Exception as exc:
                LOG.warning(f'[ParsingTaskServer] cancel callback failed: {exc}')
            return BaseResponse(
                code=200,
                msg='success',
                data={'task_id': task_id, 'cancel_status': True, 'status': DocStatus.CANCELED.value},
            )

        @app.get('/v1/tasks')
        def list_tasks(self, status: Optional[List[str]] = None, page: int = 1, page_size: int = 20):
            self._lazy_init()
            with self._db_manager.get_session() as session:
                Task = self._db_manager.get_table_orm_class(PARSER_TASK_TABLE_INFO['name'])
                query = session.query(Task)
                if status:
                    query = query.filter(Task.status.in_(status))
                total = query.count()
                rows = (
                    query.order_by(Task.created_at.desc())
                    .offset(max(page - 1, 0) * page_size)
                    .limit(page_size)
                    .all()
                )
                items = [_orm_to_dict(row) for row in rows]
            return BaseResponse(code=200, msg='success', data={
                'items': items,
                'total': total,
                'page': page,
                'page_size': page_size,
            })

        @app.get('/v1/tasks/{task_id}')
        def get_task(self, task_id: str):
            self._lazy_init()
            task = self._load_task(task_id)
            if task is None:
                raise fastapi.HTTPException(status_code=404, detail='task not found')
            return BaseResponse(code=200, msg='success', data=task)

        @app.get('/v1/algo/list')
        def list_algorithms(self):
            self._lazy_init()
            with self._db_manager.get_session() as session:
                Algo = self._db_manager.get_table_orm_class(ALGORITHM_TABLE_INFO['name'])
                rows = session.query(Algo).order_by(Algo.created_at.asc()).all()
                data = [
                    {'algo_id': row.id, 'display_name': row.display_name, 'description': row.description}
                    for row in rows
                ]
            return BaseResponse(code=200, msg='success', data=data)

        @app.get('/v1/algo/{algo_id}/groups')
        def get_algorithm_groups(self, algo_id: str):
            self._lazy_init()
            with self._db_manager.get_session() as session:
                Algo = self._db_manager.get_table_orm_class(ALGORITHM_TABLE_INFO['name'])
                row = session.query(Algo).filter(Algo.id == algo_id).first()
                if row is None:
                    raise fastapi.HTTPException(status_code=404, detail='algo not found')
            info = cloudpickle.loads(row.info_pickle)
            node_groups = info.get('node_groups', {}) if isinstance(info, dict) else {}
            data = []
            for name, group in node_groups.items():
                data.append({
                    'name': name,
                    'type': group.get('group_type'),
                    'display_name': group.get('display_name'),
                })
            return BaseResponse(code=200, msg='success', data=data)

        @app.get('/v1/health')
        def health(self):
            self._lazy_init()
            healthy = self._task_thread is not None and self._task_thread.is_alive()
            return BaseResponse(code=200 if healthy else 503, msg='success' if healthy else 'unhealthy', data={
                'status': 'ok' if healthy else 'degraded',
                'version': 'v1-mock',
                'deps': {
                    'sql': bool(self._db_manager),
                    'worker': healthy,
                },
            })

    def __init__(
        self,
        port: Optional[int] = None,
        url: Optional[str] = None,
        db_config: Optional[Dict[str, Any]] = None,
        poll_interval: float = 0.05,
        callback_func: Optional[Callable[[TaskCallbackRequest], None]] = None,
        launcher=None,
    ):
        super().__init__()
        self._raw_impl = None
        self._db_config = db_config or _get_default_db_config('doc_service_parser')
        if url:
            self._impl = UrlModule(url=ensure_call_endpoint(url))
        else:
            self._raw_impl = ParsingTaskServer._Impl(
                db_config=self._db_config,
                poll_interval=poll_interval,
                callback_func=callback_func,
            )
            self._impl = ServerModule(self._raw_impl, port=port, launcher=launcher)

    def start(self):
        result = super().start()
        if self._raw_impl:
            self._dispatch('_lazy_init')
        return result

    def stop(self):
        if self._raw_impl:
            self._dispatch('stop')
        if isinstance(self._impl, ServerModule):
            self._impl.stop()

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._impl
        if isinstance(impl, ServerModule):
            return impl._call(method, *args, **kwargs)
        return getattr(impl, method)(*args, **kwargs)

    def register_callback(self, callback_func: Callable[[TaskCallbackRequest], None]):
        if self._raw_impl:
            self._raw_impl.register_callback(callback_func)

    def create_task(self, request: TaskCreateRequest):
        return self._dispatch('create_task', request)

    def cancel_task(self, task_id: str):
        return self._dispatch('cancel_task', {'task_id': task_id})

    def list_tasks(self, status: Optional[List[str]] = None, page: int = 1, page_size: int = 20):
        return self._dispatch('list_tasks', status, page, page_size)

    def get_task(self, task_id: str):
        return self._dispatch('get_task', task_id)

    def list_algorithms(self):
        return self._dispatch('list_algorithms')

    def get_algorithm_groups(self, algo_id: str):
        return self._dispatch('get_algorithm_groups', algo_id)
