import json
import random
import inspect
import threading
import time
import traceback
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from uuid import NAMESPACE_URL, uuid5
from typing import Any, Callable, Dict, Optional, Tuple, List

from lazyllm import (
    LOG, ModuleBase, ServerModule, UrlModule, FastapiApp as app,
    LazyLLMLaunchersBase as Launcher, load_obj, once_wrapper, dump_obj
)
import requests
from lazyllm.thirdparty import fastapi

from .base import (
    ALGORITHM_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO, FINISHED_TASK_QUEUE_TABLE_INFO,
    TaskStatus, TaskStatus, TaskType, UpdateMetaRequest, AddDocRequest, CancelTaskRequest, DeleteDocRequest,
   
    _calculate_task_score, _resolve_add_doc_task_type
)
from .worker import DocumentProcessorWorker as Worker
from .queue import _SQLBasedQueue as Queue

from ..data_loaders import DirectoryReader
from ..store.document_store import _DocumentStore
from ..store.utils import create_file_path
from ..utils import BaseResponse, ensure_call_endpoint, _get_default_db_config, _orm_to_dict
from ..doc_to_db import SchemaExtractor
from ...sql import SqlManager

CALLBACK_RETRY_MIN_INTERVAL = 5.0
CALLBACK_RETRY_MAX_INTERVAL = 300.0
CALLBACK_RETRY_MAX_ATTEMPTS = 5


class DocumentProcessor(ModuleBase):

    class _Impl():
        def __init__(self, db_config: Optional[Dict[str, Any]] = None, num_workers: int = 1,
                     post_func: Optional[Callable] = None, path_prefix: Optional[str] = None,
                     callback_url: Optional[str] = None,
                     lease_duration: float = 300.0, lease_renew_interval: float = 60.0,
                     high_priority_task_types: Optional[List[str]] = None,
                     high_priority_workers: int = 1):
            self._db_config = db_config
            self._num_workers = num_workers
            self._post_func = post_func
            self._callback_url = self._normalize_callback_url(callback_url)
            if not self._check_post_func():
                raise ValueError('Invalid post function!')
            self._shutdown = False
            self._path_prefix = path_prefix
            self._lease_duration = lease_duration
            self._lease_renew_interval = lease_renew_interval
            self._high_priority_task_types = (
                high_priority_task_types
                if high_priority_task_types is not None
                else [TaskType.DOC_DELETE.value]
            )
            self._high_priority_workers = max(high_priority_workers, 0)
            self._callback_retry_attempts: Dict[int, int] = {}

            self._db_manager = None
            self._waiting_task_queue = None
            self._finished_task_queue = None
            self._post_func_thread = None
            self._workers = None
            self._high_priority_workers_module = None

        @once_wrapper(reset_on_pickle=True)
        def _lazy_init(self):
            LOG.info('[DocumentProcessor._Impl] Starting lazy initialization...')
            self._db_manager = SqlManager(**self._db_config, tables_info_dict={'tables': [ALGORITHM_TABLE_INFO]})

            self._waiting_task_queue = Queue(
                table_name=WAITING_TASK_QUEUE_TABLE_INFO['name'],
                columns=WAITING_TASK_QUEUE_TABLE_INFO['columns'],
                db_config=self._db_config,
                order_by='task_score',
                order_desc=False,
            )
            self._finished_task_queue = Queue(
                table_name=FINISHED_TASK_QUEUE_TABLE_INFO['name'],
                columns=FINISHED_TASK_QUEUE_TABLE_INFO['columns'],
                db_config=self._db_config,
                order_by='finished_at',
                order_desc=False,
            )

            self._post_func_thread = threading.Thread(target=self.process_finished_task, daemon=True)
            self._post_func_thread.start()

            if self._num_workers > 0:
                high_priority_types = [t for t in (self._high_priority_task_types or []) if t]
                high_priority_workers = 0
                if high_priority_types and self._high_priority_workers > 0:
                    if self._num_workers <= 1:
                        LOG.warning('[DocumentProcessor] num_workers <= 1, high priority workers disabled')
                    else:
                        high_priority_workers = min(self._high_priority_workers, self._num_workers - 1)
                        if high_priority_workers < self._high_priority_workers:
                            LOG.warning('[DocumentProcessor] high_priority_workers trimmed to fit num_workers')
                normal_workers = self._num_workers - high_priority_workers
                if high_priority_workers > 0:
                    self._high_priority_workers_module = Worker(
                        db_config=self._db_config,
                        num_workers=high_priority_workers,
                        lease_duration=self._lease_duration,
                        lease_renew_interval=self._lease_renew_interval,
                        high_priority_task_types=high_priority_types,
                        high_priority_only=True,
                    )
                    self._high_priority_workers_module.start()
                if normal_workers > 0:
                    self._workers = Worker(
                        db_config=self._db_config,
                        num_workers=normal_workers,
                        lease_duration=self._lease_duration,
                        lease_renew_interval=self._lease_renew_interval,
                        high_priority_task_types=high_priority_types,
                    )
                    self._workers.start()
            LOG.info('[DocumentProcessor] Lazy initialization completed!')

        def __getstate__(self):
            state = self.__dict__.copy()
            state['_db_manager'] = None
            state['_waiting_task_queue'] = None
            state['_finished_task_queue'] = None
            state['_post_func_thread'] = None
            state['_workers'] = None
            state['_high_priority_workers_module'] = None
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)

        def process_finished_task(self):
            '''process finished task in background thread'''
            while True:
                try:
                    finished_task = self._finished_task_queue.peek()
                    if finished_task:
                        if not self._is_callback_due(finished_task):
                            time.sleep(0.5)
                            continue
                        try:
                            self._callback(finished_task)
                        except Exception as exc:
                            self._schedule_callback_retry(finished_task, exc)
                            time.sleep(0.1)
                            continue
                        self._finished_task_queue.clear(filter_by={'id': finished_task['id']})
                        self._callback_retry_attempts.pop(finished_task['id'], None)
                        time.sleep(0.1)
                    else:
                        time.sleep(1)
                except Exception as e:
                    LOG.error(f'[DocumentProcessor] Failed to process finished task: {e}, {traceback.format_exc()}')
                    time.sleep(10)

        @staticmethod
        def _normalize_callback_url(callback_url: Optional[str]) -> Optional[str]:
            if not callback_url:
                return None
            return callback_url.rstrip('/')

        def set_callback_url(self, callback_url: Optional[str]):
            self._callback_url = self._normalize_callback_url(callback_url)

        @staticmethod
        def _normalize_queue_datetime(value: Any) -> Optional[datetime]:
            if value is None or isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    return None
            return None

        def _is_callback_due(self, finished_task: Dict[str, Any]) -> bool:
            finished_at = self._normalize_queue_datetime(finished_task.get('finished_at'))
            return finished_at is None or finished_at <= datetime.now()

        @staticmethod
        def _load_task_context(finished_task: Dict[str, Any]) -> Dict[str, Any]:
            task_context_json = finished_task.get('task_context_json')
            if not task_context_json:
                raise ValueError('task_context_json is missing in finished task queue')
            try:
                task_context = json.loads(task_context_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid task_context_json: {exc}') from exc
            if not isinstance(task_context, dict):
                raise ValueError('task_context_json must decode to dict')
            return task_context

        def _drop_callback_task(self, finished_task: Dict[str, Any], exc: Exception, attempt: int, reason: str):
            LOG.error('[DocumentProcessor] Callback delivery dropped queue item.'
                      f' queue_id={finished_task.get("id")}'
                      f' task_id={finished_task.get("task_id")}'
                      f' task_status={finished_task.get("task_status")}'
                      f' reason={reason}'
                      f' attempts={attempt}'
                      f' callback_url={finished_task.get("callback_url")}'
                      f' task_context_json={finished_task.get("task_context_json")}'
                      f' error={type(exc).__name__}: {exc}')
            self._finished_task_queue.clear(filter_by={'id': finished_task['id']})
            self._callback_retry_attempts.pop(finished_task['id'], None)

        @staticmethod
        def _parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
            if not value:
                return None
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                pass
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if retry_at.tzinfo is not None:
                delay = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
            else:
                delay = (retry_at - datetime.now()).total_seconds()
            return max(delay, 0.0)

        def _should_retry_callback_error(self, exc: Exception) -> Tuple[bool, Optional[float], str]:
            if isinstance(exc, ValueError):
                return False, None, 'invalid_callback_payload'
            if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                return True, None, 'transient_network_error'
            if isinstance(exc, requests.HTTPError):
                response = exc.response
                if response is None:
                    return True, None, 'http_error_without_response'
                status_code = response.status_code
                if status_code in (408, 425):
                    return True, None, f'http_{status_code}'
                if status_code == 429:
                    return True, self._parse_retry_after_seconds(response.headers.get('Retry-After')), 'http_429'
                if status_code >= 500:
                    return True, None, f'http_{status_code}'
                if 400 <= status_code < 500:
                    return False, None, f'http_{status_code}'
            return True, None, type(exc).__name__

        def _schedule_callback_retry(self, finished_task: Dict[str, Any], exc: Exception) -> bool:
            queue_id = finished_task['id']
            attempt = self._callback_retry_attempts.get(queue_id, 0) + 1
            self._callback_retry_attempts[queue_id] = attempt
            should_retry, retry_after_seconds, reason = self._should_retry_callback_error(exc)
            if not should_retry:
                self._drop_callback_task(finished_task, exc, attempt, reason)
                return False
            if attempt >= CALLBACK_RETRY_MAX_ATTEMPTS:
                self._drop_callback_task(finished_task, exc, attempt, 'retry_exhausted')
                return False
            delay = CALLBACK_RETRY_MIN_INTERVAL * (2 ** (attempt - 1))
            delay *= random.uniform(0.8, 1.2)
            if retry_after_seconds is not None:
                delay = max(delay, retry_after_seconds)
            delay = min(delay, CALLBACK_RETRY_MAX_INTERVAL)
            retry_at = datetime.now() + timedelta(seconds=delay)
            self._finished_task_queue.update(filter_by={'id': queue_id}, finished_at=retry_at)
            LOG.warning(f'[DocumentProcessor] Callback delivery failed for queue_id={queue_id},'
                        f' task_id={finished_task.get("task_id")}, retry in {delay:.1f}s'
                        f' (attempt={attempt}/{CALLBACK_RETRY_MAX_ATTEMPTS})'
                        f' reason={reason}'
                        f' error={type(exc).__name__}: {exc}')
            return True

        def _resolve_callback_url(self, payload: Dict[str, Any]) -> Optional[str]:
            return self._normalize_callback_url(
                payload.get('callback_url') or payload.get('feedback_url') or self._callback_url
            )

        def _default_post_func(self, finished_task: Dict[str, Any]):
            task_id = finished_task.get('task_id')
            task_status = finished_task.get('task_status')
            error_code = finished_task.get('error_code')
            error_msg = finished_task.get('error_msg')
            callback_url = self._normalize_callback_url(finished_task.get('callback_url'))
            if not callback_url:
                raise ValueError(f'callback_url is missing for task {task_id}')
            task_context = self._load_task_context(finished_task)

            base_payload = {'task_type': task_context.get('task_type'),
                            'kb_id': task_context.get('kb_id'),
                            'algo_id': task_context.get('algo_id')}
            items = task_context.get('items') or [{}]
            for index, item in enumerate(items):
                callback_payload = {
                    'callback_id': str(uuid5(NAMESPACE_URL, f'{task_id}:{task_status}:{index}')),
                    'task_id': task_id,
                    'task_status': task_status,
                    'payload': {k: v for k, v in {**base_payload, **item}.items() if v is not None},
                }
                for field in ('task_type', 'kb_id', 'algo_id'):
                    if base_payload.get(field) is not None:
                        callback_payload[field] = base_payload[field]
                if item.get('doc_id') is not None:
                    callback_payload['doc_id'] = item['doc_id']
                if error_code is not None:
                    callback_payload['error_code'] = error_code
                if error_msg is not None:
                    callback_payload['error_msg'] = error_msg

                response = requests.post(callback_url, json=callback_payload, timeout=8)
                response.raise_for_status()
            return True

        def register_algorithm(self, name: str, store: _DocumentStore, reader: DirectoryReader,
                               node_groups: Dict[str, Dict], schema_extractor: Optional[SchemaExtractor] = None,
                               display_name: Optional[str] = None, description: Optional[str] = None):
            # NOTE: name is the algorithm id, display_name is the algorithm display name
            self._lazy_init()
            LOG.info((f'[DocumentProcessor] Get register algorithm request: name={name},'
                      f'display_name={display_name}, description={description}'))
            # write the processor to database
            try:
                info_dict = {
                    'store': store,
                    'reader': reader,
                    'node_groups': node_groups,
                    'schema_extractor': schema_extractor,
                }
                info_pickle = dump_obj(info_dict)
                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    existing_algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == name).first()
                    if not existing_algorithm:
                        # new algorithm
                        new_algo_info = AlgoInfo(
                            id=name,
                            display_name=display_name,
                            description=description,
                            info_pickle=info_pickle,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                        )
                        session.add(new_algo_info)
                    else:
                        # existing algorithm
                        existing_algorithm.info_pickle = info_pickle
                        existing_algorithm.display_name = display_name
                        existing_algorithm.description = description
                        existing_algorithm.updated_at = datetime.now()
                LOG.info(f'[DocumentProcessor] Algorithm {name} {display_name} {description} registered!')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to register algorithm: {e}, {traceback.format_exc()}')
                raise e

        def drop_algorithm(self, name: str) -> None:
            try:
                self._lazy_init()
                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    existing_algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == name).first()
                    if existing_algorithm:
                        session.delete(existing_algorithm)
                        LOG.info(f'[DocumentProcessor] Algorithm {name} dropped!')
                    else:
                        LOG.warning(f'[DocumentProcessor] Algorithm {name} not found')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to drop algorithm: {e}, {traceback.format_exc()}')
                raise e

        def _get_algo(self, algo_id: str) -> Dict[str, Any]:
            with self._db_manager.get_session() as session:
                AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == algo_id).first()
                if algorithm is None:
                    return None
                return _orm_to_dict(algorithm)

        @app.get('/health')
        def get_health(self) -> None:
            self._lazy_init()
            if self._post_func_thread is None or not self._post_func_thread.is_alive():
                return BaseResponse(code=503, msg='Post function thread not alive')

            return BaseResponse(code=200, msg='success')

        @app.get('/prestop')
        def get_prestop(self) -> None:
            '''
            PreStop lifecycle hook endpoint.
            Called before the container is terminated to allow graceful shutdown.
            This endpoint returns immediately after setting shutdown flag.
            Actual cleanup is handled by the worker thread in background.
            K8s will wait terminationGracePeriodSeconds before sending SIGTERM.
            '''
            LOG.info('[DocumentProcessor] PreStop hook called, initiating graceful shutdown...')
            try:
                if not self._shutdown:
                    self._shutdown = True
                    # shutdown threads
                    if self._post_func_thread is not None and self._post_func_thread.is_alive():
                        self._post_func_thread.join(timeout=5.0)
                        if self._post_func_thread.is_alive():
                            LOG.warning('[DocumentProcessor] Post function thread did not stop within timeout')
                        else:
                            LOG.info('[DocumentProcessor] Post function thread stopped')
                    if self._workers:
                        self._workers.stop()
                        LOG.info('[DocumentProcessor] Workers stopped')
                    LOG.info('[DocumentProcessor] Shutdown initiated')
                else:
                    LOG.info('[DocumentProcessor] Shutdown already initiated')
                return BaseResponse(code=200, msg='shutdown initiated')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] PreStop hook failed: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500,
                                            detail=f'PreStop hook failed: {e}, {traceback.format_exc()}')

        @app.get('/algo/list')
        def get_algo_list(self) -> None:
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            res = []
            with self._db_manager.get_session() as session:
                AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                algorithms = session.query(AlgoInfo).all()
                for algorithm in algorithms:
                    res.append(_orm_to_dict(algorithm))
            data = []
            for algo_dict in res:
                data.append({
                    'algo_id': algo_dict.get('id'),
                    'display_name': algo_dict.get('display_name'),
                    'description': algo_dict.get('description'),
                    'created_at': algo_dict.get('created_at'),
                    'updated_at': algo_dict.get('updated_at'),
                })
            if not data:
                LOG.warning('[DocumentProcessor] No algorithm registered')
            return BaseResponse(code=200, msg='success', data=data)

        @app.get('/algo/{algo_id}/group/info')
        def get_algo_group_info(self, algo_id: str) -> None:
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            try:
                data = self._get_algo_group_info_data(algo_id)
                LOG.info(f'[DocumentProcessor] Get group info for {algo_id} success with {data}')
                return BaseResponse(code=200, msg='success', data=data)
            except fastapi.HTTPException:
                raise
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to get group info: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to get group info: {str(e)}')

        def get_group_info(self, algo_id: str) -> None:
            return self.get_algo_group_info(algo_id)

        def _get_algo_group_info_data(self, algo_id: str):
            algorithm = self._get_algo(algo_id)
            if algorithm is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            info_pickle_bytes = algorithm.get('info_pickle')
            info = load_obj(info_pickle_bytes)
            store: _DocumentStore = info['store']  # type: ignore
            node_groups = info['node_groups']

            data = []
            for group_name in store.activated_groups():
                if group_name in node_groups:
                    group_info = {'name': group_name, 'type': node_groups[group_name].get('group_type'),
                                  'display_name': node_groups[group_name].get('display_name')}
                    data.append(group_info)
            return data

        @staticmethod
        def _resolve_add_task_type(file_infos) -> str:
            has_reparse = False
            has_new_file = False
            for file_info in file_infos:
                if file_info.reparse_group is not None:
                    has_reparse = True
                else:
                    has_new_file = True
            if has_new_file and has_reparse:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail='new_file_ids and reparse_file_ids cannot be specified at the same time'
                )
            if has_reparse:
                return TaskType.DOC_REPARSE.value
            if has_new_file:
                return TaskType.DOC_ADD.value
            raise fastapi.HTTPException(status_code=400, detail='no input files or reparse group specified')

        @app.post('/doc/add')
        def add_doc(self, request: AddDocRequest):  # noqa: C901
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            LOG.info(f'[DocumentProcessor] Received add doc request (raw): {request.model_dump()}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            algorithm = self._get_algo(algo_id)
            if algorithm is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            # NOTE: No idempotency key check, should be handled by the caller!
            for file_info in file_infos:
                if self._path_prefix:
                    file_info.file_path = create_file_path(path=file_info.file_path, prefix=self._path_prefix)
            task_type = self._resolve_add_task_type(file_infos)
            payload = request.model_dump()
            resolved_callback_url = self._resolve_callback_url(payload)
            if resolved_callback_url:
                payload['callback_url'] = resolved_callback_url
            payload_json = json.dumps(payload, ensure_ascii=False)

            try:
                user_priority = request.priority if request.priority is not None else 0
                task_score = _calculate_task_score(task_type, user_priority)
                now = datetime.now()
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    status=TaskStatus.WAITING.value,
                    worker_id=None,
                    lease_expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                LOG.info(f'[DocumentProcessor] Task {task_id} (type={task_type}, user_priority={user_priority}, '
                         f'score={task_score}) submitted to database queue successfully')
                data = {
                    'task_id': task_id,
                    'task_type': task_type,
                    'created_at': datetime.now(),
                }
                return BaseResponse(code=200, msg='success', data=data)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.post('/doc/meta/update')
        def update_meta(self, request: UpdateMetaRequest):
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            LOG.info(f'[DocumentProcessor] Received update meta request (raw): {request.model_dump()}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos

            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            algorithm = self._get_algo(algo_id)
            if algorithm is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received update meta request: {payload}')
            resolved_callback_url = self._resolve_callback_url(payload)
            if resolved_callback_url:
                payload['callback_url'] = resolved_callback_url
            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                task_type = TaskType.DOC_UPDATE_META.value
                user_priority = request.priority if request.priority is not None else 0
                task_score = _calculate_task_score(task_type, user_priority)
                now = datetime.now()
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    status=TaskStatus.WAITING.value,
                    worker_id=None,
                    lease_expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                LOG.info(f'[DocumentProcessor] Update meta task {task_id} (user_priority={user_priority}, '
                         f'score={task_score}) submitted to database queue successfully')
                data = {
                    'task_id': task_id,
                    'task_type': TaskType.DOC_UPDATE_META.value,
                    'created_at': datetime.now(),
                }
                return BaseResponse(code=200, msg='success', data=data)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit update meta task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.delete('/doc/delete')
        def delete_doc(self, request: DeleteDocRequest):
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            LOG.info(f'[DocumentProcessor] Received delete doc request (raw): {request.model_dump()}')
            task_id = request.task_id
            algo_id = request.algo_id
            doc_ids = request.doc_ids
            if not doc_ids:
                raise fastapi.HTTPException(status_code=400, detail='doc_ids is required')
            algorithm = self._get_algo(algo_id)
            if algorithm is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received delete doc request: {payload}')
            resolved_callback_url = self._resolve_callback_url(payload)
            if resolved_callback_url:
                payload['callback_url'] = resolved_callback_url
            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                task_type = TaskType.DOC_DELETE.value
                user_priority = request.priority if request.priority is not None else 0
                task_score = _calculate_task_score(task_type, user_priority)
                now = datetime.now()
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    status=TaskStatus.WAITING.value,
                    worker_id=None,
                    lease_expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                LOG.info(f'[DocumentProcessor] Delete task {task_id} (user_priority={user_priority}, '
                         f'score={task_score}) submitted to database queue successfully')
                data = {
                    'task_id': task_id,
                    'task_type': TaskType.DOC_DELETE.value,
                    'created_at': datetime.now(),
                }
                return BaseResponse(code=200, msg='success', data=data)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit delete task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.post('/doc/cancel')
        def cancel(self, request: CancelTaskRequest):
            self._lazy_init()
            if self._shutdown:
                raise fastapi.HTTPException(status_code=503, detail='Server is shutting down...')
            LOG.info(f'[DocumentProcessor] Received cancel task request (raw): {request.model_dump()}')
            task_id = request.task_id
            try:
                # NOTE: only the task in waiting state can be canceled
                cancel_status = False
                deleted = self._waiting_task_queue.delete(
                    filter_by={'task_id': task_id, 'status': TaskStatus.WAITING.value}
                )
                message = ''
                if deleted:
                    cancel_status = True
                    message = 'Canceled by user'
                else:
                    message = (f'Task {task_id} not found in waiting queue,'
                               ' task may be running or already finished and cannot be canceled!')
                return BaseResponse(
                    code=200,
                    msg='success' if cancel_status else 'failed',
                    data={
                        'task_id': task_id,
                        'cancel_status': cancel_status,
                        'message': message,
                    }
                )
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to cancel task {task_id}: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to cancel task: {str(e)}')

        def _check_post_func(self) -> bool:
            '''assert post function is callable and params include task_id, task_status, error_code, error_msg'''
            if not self._post_func:
                if self._callback_url:
                    LOG.info('[DocumentProcessor] No custom post function configured,'
                             ' using built-in HTTP callback')
                else:
                    LOG.warning('[DocumentProcessor] No custom post function configured, built-in HTTP callback'
                                ' will only run when callback_url or feedback_url is provided in task request')
                return True
            if not callable(self._post_func):
                LOG.error('[DocumentProcessor] Post function is not callable')
                return False
            try:
                sig = inspect.signature(self._post_func)
            except (TypeError, ValueError):
                LOG.error('[DocumentProcessor] Failed to inspect post function signature')
                return False
            params = sig.parameters
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_keyword and not all(
                param in params for param in ['task_id', 'task_status', 'error_code', 'error_msg']
            ):
                LOG.error('[DocumentProcessor] Post function params do not include'
                          ' task_id, task_status, error_code, error_msg')
                return False
            return True

        def _callback(self, finished_task: Optional[Dict[str, Any]] = None, **legacy_kwargs):
            '''callback to service'''
            if finished_task is None:
                finished_task = legacy_kwargs
            task_id = finished_task.get('task_id')
            task_type = finished_task.get('task_type')
            task_status = finished_task.get('task_status')
            error_code = finished_task.get('error_code')
            error_msg = finished_task.get('error_msg')
            message = f'Task {task_id} callback status: {task_status}.'
            if error_msg:
                message += f' Error code: {error_code}, error_msg: {error_msg}.'
            LOG.info(f'[DocumentProcessor] {message}')

            try:
                if self._post_func:
                    if 'task_type' in self._post_func.__code__.co_varnames:
                        self._post_func(task_id, task_status, error_code, error_msg, task_type=task_type)
                    else:
                        self._post_func(task_id, task_status, error_code, error_msg)
                else:
                    self._default_post_func(finished_task)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to call post function: {e}, {traceback.format_exc()}')
                if self._post_func:
                    try:
                        self._default_post_func(finished_task)
                    except Exception:
                        raise e
                else:
                    raise e

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(self, port: int = None, url: str = None, num_workers: int = 1,
                 db_config: Optional[Dict[str, Any]] = None,
                 launcher: Optional[Launcher] = None, post_func: Optional[Callable] = None,
                 path_prefix: Optional[str] = None, callback_url: Optional[str] = None, lease_duration: float = 300.0,
                 lease_renew_interval: float = 60.0, high_priority_task_types: Optional[List[str]] = None,
                 high_priority_workers: int = 1):
        super().__init__()
        self._raw_impl = None  # save the reference of the original Impl object
        self._db_config = db_config if db_config else _get_default_db_config('doc_task_management')
        if not url:
            # create the Impl object (lazy loading, no threads created)
            self._raw_impl = DocumentProcessor._Impl(
                num_workers=num_workers,
                db_config=self._db_config,
                post_func=post_func,
                path_prefix=path_prefix,
                lease_duration=lease_duration,
                lease_renew_interval=lease_renew_interval,
                high_priority_task_types=high_priority_task_types,
                high_priority_workers=high_priority_workers,
                callback_url=callback_url
            )
            self._impl = ServerModule(self._raw_impl, port=port, launcher=launcher)
        else:
            self._impl = UrlModule(url=ensure_call_endpoint(url))

    def start(self):
        # start the server
        result = super().start()
        # ensure the initialization
        if self._raw_impl:
            LOG.info('[DocumentProcessor] Server started, triggering post-start initialization...')
            try:
                self._dispatch('_lazy_init')
                LOG.info('[DocumentProcessor] Post-start initialization triggered successfully')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Post-start initialization failed: {e}, {traceback.format_exc()}')
                raise
        return result

    def wait(self):
        impl = self._impl
        if isinstance(impl, ServerModule):
            return impl.wait()
        LOG.warning('[DocumentProcessor] wait() is no-op in UrlModule mode')

    def set_callback_url(self, callback_url: Optional[str]):
        if isinstance(self._impl, UrlModule):
            raise RuntimeError('set_callback_url is only supported in local server mode')
        return self._dispatch('set_callback_url', callback_url)

    def _dispatch(self, method: str, *args, **kwargs):
        try:
            impl = self._impl
            if isinstance(impl, ServerModule):
                return impl._call(method, *args, **kwargs)
            else:
                return getattr(impl, method)(*args, **kwargs)
        except Exception as e:
            LOG.error(f'[DocumentProcessor] Failed to dispatch method {method}: {e}, {traceback.format_exc()}')
            raise e

    def register_algorithm(self, name: str, store: _DocumentStore, reader: DirectoryReader,
                           node_groups: Dict[str, Dict], schema_extractor: Optional[SchemaExtractor] = None,
                           display_name: Optional[str] = None, description: Optional[str] = None, **kwargs):
        assert isinstance(reader, DirectoryReader), 'Only DirectoryReader can be registered to processor'
        self._dispatch('register_algorithm', name, store, reader, node_groups, schema_extractor,
                       display_name, description, **kwargs)

    def drop_algorithm(self, name: str) -> None:
        return self._dispatch('drop_algorithm', name)
