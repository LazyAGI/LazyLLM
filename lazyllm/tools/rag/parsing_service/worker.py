import json
import os
import subprocess
import time
import traceback
import threading
import cloudpickle

from datetime import datetime
from uuid import uuid4
from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule, once_wrapper
from ..utils import BaseResponse, _get_default_db_config
from .base import (
    FINISHED_TASK_QUEUE_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO,
    TaskStatus, TaskType, ALGORITHM_TABLE_INFO, AddDocRequest, UpdateMetaRequest,
    DeleteDocRequest, _calculate_task_score, _resolve_add_doc_task_type
)
from .impl import _Processor
from .queue import _SQLBasedQueue as Queue
from ...sql import SqlManager

WORKER_ERROR_RETRY_INTERVAL = 5.0


class DocumentProcessorWorker(ModuleBase):

    class _Impl():
        def __init__(self, db_config: dict = None, task_poller=None, lease_duration: float = 300.0,
                     lease_renew_interval: float = 60.0, high_priority_task_types: list[str] = None,
                     high_priority_only: bool = False):
            self._db_config = db_config if db_config else _get_default_db_config('doc_task_management')
            self._shutdown = False
            self._processors: dict[str, _Processor] = {}  # algo_id -> _Processor
            self._waiting_task_queue = None
            self._finished_task_queue = None
            self._worker_thread = None
            self._poller_thread = None
            if task_poller is not None and not callable(task_poller):
                raise TypeError('task_poller is not callable')
            self._task_poller = task_poller
            self._task_poller_impl = self._wrap_task_poller(task_poller) if task_poller else None
            self._worker_id = f'{self._get_worker_identity()}-{uuid4()}'
            self._in_progress_task = None
            self._lease_thread = None
            self._lease_stop_event = None
            self._lease_duration = lease_duration
            self._lease_renew_interval = lease_renew_interval
            self._high_priority_task_types = set(high_priority_task_types or [])
            self._high_priority_only = high_priority_only

        @once_wrapper(reset_on_pickle=True)
        def _lazy_init(self):
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
            )
            self._db_manager = SqlManager(
                **self._db_config,
                tables_info_dict={'tables': [ALGORITHM_TABLE_INFO]},
            )

            LOG.info(f'{self._log_prefix()} initialized')

        def _get_worker_identity(self) -> str:
            env_keys = ('POD_IP', 'POD_NAME', 'HOSTNAME')
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    return value
            try:
                ip = subprocess.check_output(['hostname', '-i'], text=True).strip()
                if ip:
                    return ip
            except Exception:
                pass
            return 'worker'

        def _log_prefix(self, task_id: str = None) -> str:
            if task_id:
                return f'[DocumentProcessorWorker._Impl][worker_id={self._worker_id}][task_id={task_id}]'
            return f'[DocumentProcessorWorker._Impl][worker_id={self._worker_id}]'

        def _wrap_task_poller(self, task_poller):
            def _impl():
                result = task_poller()
                if result is None:
                    return []
                return result if isinstance(result, list) else [result]
            return _impl

        def _start_lease_renewal(self, task_id: str):
            if self._lease_renew_interval <= 0:
                return
            self._lease_stop_event = threading.Event()

            def _renew():
                while not self._lease_stop_event.wait(self._lease_renew_interval):
                    try:
                        self._waiting_task_queue.extend_lease(task_id, self._worker_id, self._lease_duration)
                    except Exception as e:
                        LOG.warning(f'{self._log_prefix(task_id)} Failed to extend lease: {e}')

            self._lease_thread = threading.Thread(target=_renew, daemon=True)
            self._lease_thread.start()

        def _stop_lease_renewal(self):
            if self._lease_stop_event is not None:
                self._lease_stop_event.set()
            if self._lease_thread is not None and self._lease_thread.is_alive():
                self._lease_thread.join(timeout=2.0)
            self._lease_thread = None
            self._lease_stop_event = None

        def _fail_in_progress_task(self, reason: str):
            if not self._in_progress_task:
                return
            task_id = self._in_progress_task.get('task_id')
            task_type = self._in_progress_task.get('task_type')
            if task_id and task_type:
                self._enqueue_finished_task(
                    task_id=task_id,
                    task_type=task_type,
                    task_status=TaskStatus.FAILED,
                    error_code='PRESTOP',
                    error_msg=reason,
                )
                deleted = self._waiting_task_queue.delete(
                    filter_by={'task_id': task_id, 'worker_id': self._worker_id}
                )
                if deleted == 0:
                    LOG.warning(f'{self._log_prefix(task_id)} Failed to delete in-progress task')
            self._in_progress_task = None

        @app.get('/health')
        def get_health(self):
            self._lazy_init()
            if self._worker_thread is None:
                return BaseResponse(code=503, msg='Worker thread not started')

            if not self._worker_thread.is_alive():
                LOG.error(f'{self._log_prefix()} Worker thread is dead')
                return BaseResponse(code=503, msg='Worker thread is not alive')

            return BaseResponse(code=200, msg='success')

        @app.get('/prestop')
        def get_prestop(self):
            self._shutdown = True
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    LOG.warning(f'{self._log_prefix()} Worker thread did not stop within timeout')
                    self._fail_in_progress_task('prestop timeout')
                else:
                    LOG.info(f'{self._log_prefix()} Worker thread stopped')
            if self._poller_thread is not None and self._poller_thread.is_alive():
                self._poller_thread.join(timeout=5.0)
                if self._poller_thread.is_alive():
                    LOG.warning(f'{self._log_prefix()} Poller thread did not stop within timeout')
                else:
                    LOG.info(f'{self._log_prefix()} Poller thread stopped')
            return BaseResponse(code=200, msg='success')

        def _get_or_create_processor(self, algo_id: str) -> _Processor:
            try:
                self._lazy_init()
                if algo_id in self._processors:
                    LOG.debug(f'{self._log_prefix()} Using cached processor for {algo_id}')
                    return self._processors[algo_id]

                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == algo_id).first()
                    if algorithm is None:
                        raise ValueError(f'Algo id {algo_id} not found')
                    display_name = algorithm.display_name
                    description = algorithm.description
                    info_pickle = algorithm.info_pickle
                    info = cloudpickle.loads(info_pickle)
                    store = info['store']
                    reader = info['reader']
                    node_groups = info['node_groups']
                    schema_extractor = info['schema_extractor']
                    processor = _Processor(algo_id, store, reader, node_groups, schema_extractor,
                                           display_name, description)
                    self._processors[algo_id] = processor
                    LOG.info(f'{self._log_prefix()} Created processor for {algo_id}')
                return self._processors[algo_id]
            except Exception as e:
                LOG.warning(f'{self._log_prefix()} Failed to load algo: {e}')
                raise e

        def _exec_add_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                file_infos = payload.get('file_infos')
                kb_id = payload.get('kb_id', None)
                input_files = []
                ids = []
                metadatas = []

                for file_info in file_infos:
                    input_files.append(file_info.get('file_path'))
                    ids.append(file_info.get('doc_id'))
                    metadatas.append(file_info.get('metadata'))

                processor.add_doc(input_files=input_files, ids=ids, metadatas=metadatas, kb_id=kb_id)
            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Execute add task failed: {e}')
                raise e

        def _exec_reparse_task(
            self, processor: _Processor, task_id: str, payload: dict
        ):
            try:
                file_infos = payload.get('file_infos')
                kb_id = payload.get('kb_id', None)
                reparse_group = None
                reparse_doc_ids = []
                reparse_files = []
                reparse_metadatas = []

                first_reparse_group = None
                for file_info in file_infos:
                    current_group = file_info.get('reparse_group')
                    if first_reparse_group is None:
                        first_reparse_group = current_group
                    elif first_reparse_group != current_group:
                        raise ValueError('All files must have the same reparse_group')
                    reparse_doc_ids.append(file_info.get('doc_id'))
                    reparse_files.append(file_info.get('file_path'))
                    reparse_metadatas.append(file_info.get('metadata'))

                reparse_group = first_reparse_group
                processor.reparse(group_name=reparse_group, doc_ids=reparse_doc_ids,
                                  doc_paths=reparse_files, metadatas=reparse_metadatas,
                                  kb_id=kb_id)
            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Execute reparse task failed: {e}')
                raise e

        def _exec_transfer_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                file_infos = payload.get('file_infos')
                kb_id = payload.get('kb_id', None)
                input_files = []
                ids = []
                metadatas = []

                transfer_mode = None
                target_kb_id = None
                target_doc_ids = []

                for file_info in file_infos:
                    input_files.append(file_info.get('file_path'))
                    ids.append(file_info.get('doc_id'))
                    metadatas.append(file_info.get('metadata'))
                    if transfer_mode is None:
                        transfer_mode = file_info.get('transfer_params', {}).get('mode')
                    if target_kb_id is None:
                        target_kb_id = file_info.get('transfer_params', {}).get('target_kb_id')
                    target_doc_ids.append(file_info.get('transfer_params', {}).get('target_doc_id'))
                processor.add_doc(input_files=input_files, ids=ids, metadatas=metadatas, kb_id=kb_id,
                                  transfer_mode=transfer_mode, target_kb_id=target_kb_id, target_doc_ids=target_doc_ids)

            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Execute transfer task failed: {e}')
                raise e

        def _exec_delete_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                kb_id = payload.get('kb_id')
                doc_ids = payload.get('doc_ids')
                processor.delete_doc(doc_ids=doc_ids, kb_id=kb_id)
            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Execute delete task failed: {e}')
                raise e

        def _exec_update_meta_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                file_infos = payload.get('file_infos')
                kb_id = payload.get('kb_id', None)
                for file_info in file_infos:
                    doc_id = file_info.get('doc_id')
                    metadata = file_info.get('metadata')
                    processor.update_doc_meta(doc_id=doc_id, metadata=metadata, kb_id=kb_id)
            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Execute update meta task failed: {e}')
                raise e

        def _resolve_task_type(self, request: AddDocRequest) -> str:
            return _resolve_add_doc_task_type(request)

        def _validate_task_payload(self, task_type: str, payload: dict):
            if not isinstance(payload, dict):
                raise ValueError('payload must be a dict')
            if task_type in (
                TaskType.DOC_ADD.value,
                TaskType.DOC_REPARSE.value,
                TaskType.DOC_TRANSFER.value,
                TaskType.DOC_UPDATE_META.value,
            ):
                file_infos = payload.get('file_infos')
                if not isinstance(file_infos, list) or not file_infos:
                    raise ValueError(f'file_infos is required for task_type {task_type}')
            if task_type == TaskType.DOC_DELETE.value:
                doc_ids = payload.get('doc_ids')
                if not isinstance(doc_ids, list) or not doc_ids:
                    raise ValueError('doc_ids is required for task_type DOC_DELETE')

        def _enqueue_task_from_payload(self, task: dict):
            try:
                task_type = task.get('task_type')
                if task_type == TaskType.DOC_DELETE.value:
                    task_info = DeleteDocRequest(**task)
                elif task_type == TaskType.DOC_UPDATE_META.value:
                    task_info = UpdateMetaRequest(**task)
                else:
                    task_info = AddDocRequest(**task)
                    task_type = task_type or self._resolve_task_type(task_info)
                task_id = task_info.task_id
                payload = task_info.model_dump()
                self._validate_task_payload(task_type, payload)
                user_priority = task_info.priority if task_info.priority is not None else 0
                task_score = _calculate_task_score(task_type, user_priority)
                payload_json = json.dumps(payload, ensure_ascii=False)
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
                LOG.info(f'{self._log_prefix(task_id)} [Poller] task (type={task_type}, '
                         f'user_priority={user_priority}, score={task_score}) '
                         'submitted to database queue successfully')
            except Exception as e:
                LOG.warning(f'{self._log_prefix()} [Poller] Skip invalid task payload: {e}. '
                            f'payload={task}')

        def _poller(self):  # noqa: C901
            while not self._shutdown:
                try:
                    tasks = self._task_poller_impl()
                    if not tasks:
                        time.sleep(0.1)
                        continue
                    for task in tasks:
                        self._enqueue_task_from_payload(task)
                except Exception as e:
                    LOG.error(f'{self._log_prefix()} [Poller] fetch failed: {e}')
                    time.sleep(WORKER_ERROR_RETRY_INTERVAL)
            LOG.info(f'{self._log_prefix()} [Poller] stopped')

        def _poll_task(self):
            include_types = None
            exclude_types = None
            if self._high_priority_task_types and self._high_priority_only:
                include_types = list(self._high_priority_task_types)
            return self._waiting_task_queue.claim(
                worker_id=self._worker_id,
                lease_duration=self._lease_duration,
                status_waiting=TaskStatus.WAITING.value,
                status_working=TaskStatus.WORKING.value,
                include_task_types=include_types,
                exclude_task_types=exclude_types,
            )

        def _enqueue_finished_task(self, task_id: str, task_type: str, task_status: TaskStatus,
                                   error_code: str = None, error_msg: str = None):
            try:
                self._lazy_init()
                self._finished_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    task_status=task_status.value,
                    finished_at=datetime.now(),
                    error_code=error_code if error_code else '200',
                    error_msg=error_msg if error_msg else 'success'
                )
                if task_status == TaskStatus.FINISHED:
                    LOG.info(f'{self._log_prefix(task_id)} Task finished successfully')
                else:
                    LOG.error(f'{self._log_prefix(task_id)} Task completed with status {task_status}: {error_msg}')
            except Exception as e:
                LOG.error(f'{self._log_prefix(task_id)} Failed to enqueue finished task: {e}')

        def _worker_impl(self):  # noqa: C901
            while not self._shutdown:
                task_id = None
                task_type = None
                try:
                    task_data = self._poll_task()
                    if not task_data:
                        time.sleep(0.1)
                        continue

                    task_id = task_data['task_id']
                    task_type = task_data['task_type']
                    self._in_progress_task = {'task_id': task_id, 'task_type': task_type}
                    self._start_lease_renewal(task_id)
                    payload = json.loads(task_data.get('message'))
                    algo_id = payload.get('algo_id')
                    if not algo_id:
                        raise ValueError(f'{self._log_prefix(task_id)} task_id is missing algo_id in payload: {payload}')

                    LOG.info(f'{self._log_prefix(task_id)} Start processing task, type: {task_type}, '
                             f'algo_id: {algo_id}')

                    processor = self._get_or_create_processor(algo_id)
                    if task_type == TaskType.DOC_ADD.value:
                        self._exec_add_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_REPARSE.value:
                        self._exec_reparse_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_DELETE.value:
                        self._exec_delete_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_UPDATE_META.value:
                        self._exec_update_meta_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_TRANSFER.value:
                        self._exec_transfer_task(processor, task_id, payload)
                    else:
                        raise ValueError(f'{self._log_prefix(task_id)} Unknown task type: {task_type}')

                    self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FINISHED,
                                                error_code='200', error_msg='success')
                    deleted = self._waiting_task_queue.delete(
                        filter_by={'task_id': task_id, 'worker_id': self._worker_id}
                    )
                    if deleted == 0:
                        LOG.warning(f'{self._log_prefix(task_id)} Failed to delete finished task')
                except Exception as e:
                    LOG.error(f'{self._log_prefix(task_id)} Failed to run task: {e}, {traceback.format_exc()}')
                    if task_id and task_type:
                        self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FAILED,
                                                    error_code=type(e).__name__, error_msg=str(e))
                        deleted = self._waiting_task_queue.delete(
                            filter_by={'task_id': task_id, 'worker_id': self._worker_id}
                        )
                        if deleted == 0:
                            LOG.warning(f'{self._log_prefix(task_id)} Failed to delete failed task')
                    time.sleep(WORKER_ERROR_RETRY_INTERVAL)
                    continue
                finally:
                    self._stop_lease_renewal()
                    self._in_progress_task = None

        def start(self):
            LOG.info(f'{self._log_prefix()} Starting worker...')
            self._lazy_init()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                LOG.warning(f'{self._log_prefix()} Worker thread is already running')
                return
            self._shutdown = False
            if self._task_poller_impl is not None:
                if self._poller_thread is None or not self._poller_thread.is_alive():
                    self._poller_thread = threading.Thread(target=self._poller, daemon=True)
                    self._poller_thread.start()
            self._worker_thread = threading.Thread(target=self._worker_impl, daemon=True)
            self._worker_thread.start()
            LOG.info(f'{self._log_prefix()} Worker thread started')

        def shutdown(self):
            LOG.info(f'{self._log_prefix()} Shutting down worker...')
            self._shutdown = True
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    LOG.warning(f'{self._log_prefix()} Worker thread did not stop within timeout')
                    self._fail_in_progress_task('shutdown timeout')
                else:
                    LOG.info(f'{self._log_prefix()} Worker thread stopped')
            if self._poller_thread is not None and self._poller_thread.is_alive():
                self._poller_thread.join(timeout=5.0)
                if self._poller_thread.is_alive():
                    LOG.warning(f'{self._log_prefix()} Poller thread did not stop within timeout')
                else:
                    LOG.info(f'{self._log_prefix()} Poller thread stopped')

    def __init__(self, db_config: dict = None, num_workers: int = 1, port: int = None,
                 task_poller=None, lease_duration: float = 300.0, lease_renew_interval: float = 60.0,
                 high_priority_task_types: list[str] = None, high_priority_only: bool = False):
        super().__init__()
        self._db_config = db_config if db_config else _get_default_db_config('doc_task_management')
        self._num_workers = num_workers
        self._port = port
        worker_impl = DocumentProcessorWorker._Impl(
            db_config=self._db_config,
            task_poller=task_poller,
            lease_duration=lease_duration,
            lease_renew_interval=lease_renew_interval,
            high_priority_task_types=high_priority_task_types,
            high_priority_only=high_priority_only,
        )
        self._worker_impl = ServerModule(worker_impl, port=self._port, num_replicas=self._num_workers)
        LOG.info(f'[DocumentProcessorWorker] Worker initialized with {num_workers} workers')

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._worker_impl
        if isinstance(impl, ServerModule):
            return impl._call(method, *args, **kwargs)
        else:
            return getattr(impl, method)(*args, **kwargs)

    def start(self):
        result = super().start()
        LOG.info('[DocumentProcessorWorker] Starting worker...')
        self._dispatch('start')
        LOG.info('[DocumentProcessorWorker] Worker started')
        return result

    def stop(self):
        LOG.info('[DocumentProcessorWorker] Stopping worker...')
        self._dispatch('shutdown')
        return super().stop()
