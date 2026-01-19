import json
import time
import traceback
import threading
import cloudpickle

from datetime import datetime
from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule, once_wrapper
from ..utils import BaseResponse, _get_default_db_config
from .base import (
    FINISHED_TASK_QUEUE_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO,
    TaskStatus, TaskType, ALGORITHM_TABLE_INFO
)
from .impl import _Processor
from .queue import _SQLBasedQueue as Queue
from ...sql import SqlManager

WORKER_ERROR_RETRY_INTERVAL = 5.0


class DocumentProcessorWorker(ModuleBase):

    class _Impl():
        def __init__(self, db_config: dict = None):
            self._db_config = db_config if db_config else _get_default_db_config('doc_task_management')
            self._shutdown = False
            self._processors: dict[str, _Processor] = {}  # algo_id -> _Processor
            self._waiting_task_queue = None
            self._finished_task_queue = None
            self._worker_thread = None

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

            LOG.info('[DocumentProcessorWorker._Impl] initialized')

        @app.get('/health')
        def get_health(self):
            self._lazy_init()
            if self._worker_thread is None:
                return BaseResponse(code=503, msg='Worker thread not started')

            if not self._worker_thread.is_alive():
                LOG.error('[DocumentProcessorWorker._Impl] Worker thread is dead')
                return BaseResponse(code=503, msg='Worker thread is not alive')

            return BaseResponse(code=200, msg='success')

        @app.get('/prestop')
        def get_prestop(self):
            self._shutdown = True
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    LOG.warning('[DocumentProcessorWorker._Impl] Worker thread did not stop within timeout')
                else:
                    LOG.info('[DocumentProcessorWorker._Impl] Worker thread stopped')
            return BaseResponse(code=200, msg='success')

        def _get_or_create_processor(self, algo_id: str) -> _Processor:
            try:
                self._lazy_init()
                if algo_id in self._processors:
                    LOG.debug(f'[DocumentProcessorWorker._Impl] Using cached processor for {algo_id}')
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
                    LOG.info(f'[DocumentProcessorWorker._Impl] Created processor for {algo_id}')
                return self._processors[algo_id]
            except Exception as e:
                LOG.warning(f'[DocumentProcessorWorker._Impl] Failed to load algo: {e}')
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
                LOG.error(f'[DocumentProcessorWorker._Impl] Task-{task_id}: execute add task failed, error: {e}')
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
                LOG.error(f'[DocumentProcessorWorker._Impl] Task-{task_id}: execute reparse task failed, error: {e}')
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
                LOG.error(f'[DocumentProcessorWorker._Impl] Task-{task_id}: execute transfer task failed, error: {e}')
                raise e

        def _exec_delete_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                kb_id = payload.get('kb_id')
                doc_ids = payload.get('doc_ids')
                processor.delete_doc(doc_ids=doc_ids, kb_id=kb_id)
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker._Impl] Task-{task_id}: execute delete task failed, error: {e}')
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
                LOG.error(f'[DocumentProcessorWorker._Impl] Task-{task_id}: execute update meta task failed,'
                          f'error: {e}')
                raise e

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
                    LOG.info(f'[DocumentProcessorWorker._Impl] Task {task_id} finished successfully')
                else:
                    LOG.error(f'[DocumentProcessorWorker._Impl] Task {task_id} completed with status {task_status}:'
                              f' {error_msg}')
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker._Impl] Failed to enqueue finished task {task_id}: {e}')

        def _worker_impl(self):
            while not self._shutdown:
                task_id = None
                task_type = None
                try:
                    task_data = self._waiting_task_queue.dequeue()
                    if not task_data:
                        time.sleep(0.1)
                        continue

                    task_id = task_data['task_id']
                    task_type = task_data['task_type']
                    payload = json.loads(task_data.get('message'))
                    algo_id = payload.get('algo_id')
                    if not algo_id:
                        raise ValueError(f'[DocumentProcessorWorker._Impl] task_id {task_id} is missing algo_id in '
                                         f'payload: {payload}')

                    LOG.info(f'[DocumentProcessorWorker._Impl] Start processing task {task_id}, type: {task_type},'
                             f' algo_id: {algo_id}')

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
                        raise ValueError(f'[DocumentProcessorWorker._Impl] Unknown task type: {task_type}')

                    self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FINISHED,
                                                error_code='200', error_msg='success')
                except Exception as e:
                    LOG.error(f'[DocumentProcessorWorker._Impl] Failed to run task {task_id}: {e},'
                              f' {traceback.format_exc()}')
                    if task_id and task_type:
                        self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FAILED,
                                                    error_code=type(e).__name__, error_msg=str(e))
                    time.sleep(WORKER_ERROR_RETRY_INTERVAL)
                    continue

        def start(self):
            LOG.info('[DocumentProcessorWorker._Impl] Starting worker...')
            self._lazy_init()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                LOG.warning('[DocumentProcessorWorker._Impl] Worker thread is already running')
                return
            self._shutdown = False
            self._worker_thread = threading.Thread(target=self._worker_impl, daemon=True)
            self._worker_thread.start()
            LOG.info('[DocumentProcessorWorker._Impl] Worker thread started')

        def shutdown(self):
            LOG.info('[DocumentProcessorWorker._Impl] Shutting down worker...')
            self._shutdown = True
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    LOG.warning('[DocumentProcessorWorker._Impl] Worker thread did not stop within timeout')
                else:
                    LOG.info('[DocumentProcessorWorker._Impl] Worker thread stopped')

    def __init__(self, db_config: dict = None, num_workers: int = 1, port: int = None):
        super().__init__()
        self._db_config = db_config if db_config else _get_default_db_config('doc_task_management')
        self._num_workers = num_workers
        self._port = port
        worker_impl = DocumentProcessorWorker._Impl(db_config=self._db_config)
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
