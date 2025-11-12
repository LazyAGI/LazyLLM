import base64
import json
import time
import traceback
import threading
import cloudpickle

from datetime import datetime
from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule
from ..utils import BaseResponse
from .base import (
    FINISHED_TASK_QUEUE_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO,
    TaskStatus, TaskType
)
from .impl import _Processor, _get_default_db_config
from .queue import SQLBasedQueue as Queue

WORKER_ERROR_RETRY_INTERVAL = 5.0


class DocumentProcessorWorker(ModuleBase):

    class Impl():
        def __init__(self, db_config: dict = None, server_url: str = None):
            self._db_config = db_config if db_config else _get_default_db_config()
            self._server_url = server_url
            self._server = None
            self._shutdown = False
            self._processors: dict[str, _Processor] = {}  # algo_id -> _Processor
            self._initialized = False
            self._waiting_task_queue = None
            self._finished_task_queue = None
            self._worker_thread = None

        def _ensure_initialized(self):
            if self._initialized:
                return
            self._initialized = True
            self._waiting_task_queue = Queue(
                table_name=WAITING_TASK_QUEUE_TABLE_INFO['name'],
                columns=WAITING_TASK_QUEUE_TABLE_INFO['columns'],
                db_config=self._db_config,
            )

            self._finished_task_queue = Queue(
                table_name=FINISHED_TASK_QUEUE_TABLE_INFO['name'],
                columns=FINISHED_TASK_QUEUE_TABLE_INFO['columns'],
                db_config=self._db_config,
            )
            LOG.info('[DocumentProcessorWorker - Impl] initialized')

        @property
        def server(self):
            if self._server is None and self._server_url:
                from .server import DocumentProcessor
                self._server = DocumentProcessor(url=self._server_url)
                LOG.info('[DocumentProcessorWorker - Impl] Initialized DocumentProcessor '
                         f'client with url: {self._server_url}')
            return self._server

        @app.get('/health')
        def get_health(self):
            self._ensure_initialized()
            if self._worker_thread is None:
                return BaseResponse(code=503, msg='Worker thread not started')

            if not self._worker_thread.is_alive():
                LOG.error('[DocumentProcessorWorker - Impl] Worker thread is dead')
                return BaseResponse(code=503, msg='Worker thread is not alive')

            return BaseResponse(code=200, msg='success')

        @app.get('/prestop')
        def get_prestop(self):
            self._shutdown = True
            return BaseResponse(code=200, msg='success')

        def _get_or_create_processor(self, algo_id: str):
            try:
                self._ensure_initialized()
                if not self.server:
                    raise ValueError('Server is not initialized')
                # Always fetch from server to get latest version
                response = self.server.get_algo_info(algo_id)
                if response.code != 200:
                    LOG.error(f'[DocumentProcessorWorker - Impl] Failed to get algo info: {response.msg}')
                    raise ValueError(f'[Worker] Failed to get algo info: {response.msg}')
                data = response.data
                returned_algo_id = data.get('algo_id')
                if returned_algo_id != algo_id:
                    raise ValueError(f'Algo id {returned_algo_id} does not match the requested algo_id {algo_id}')

                version = data.get('version')
                # Check if we need to update the processor (new algo or version changed)
                if algo_id in self._processors and self._processors[algo_id].version == version:
                    LOG.debug(f'[DocumentProcessorWorker - Impl] Using cached processor for {algo_id}'
                              f'with version {version}')
                    return self._processors[algo_id]

                # Create or update processor
                LOG.info(f'[DocumentProcessorWorker - Impl] Loading algo {algo_id} from server '
                         f'with version {version}...')
                display_name = data.get('display_name')
                description = data.get('description')
                info_pickle_b64 = data.get('info_pickle')
                # Decode base64 string back to bytes
                info_pickle_bytes = base64.b64decode(info_pickle_b64)
                info = cloudpickle.loads(info_pickle_bytes)
                store = info['store']
                reader = info['reader']
                node_groups = info['node_groups']
                self._processors[algo_id] = _Processor(store, reader, node_groups, display_name, description, version)
                LOG.info(f'[DocumentProcessorWorker - Impl] Created/Updated processor for {algo_id}')
                return self._processors[algo_id]
            except Exception as e:
                LOG.warning(f'[DocumentProcessorWorker - Impl] Failed to load algo from server: {e}')
                if algo_id in self._processors:
                    LOG.warning(f'[DocumentProcessorWorker - Impl] using cached processor for {algo_id}')
                return self._processors.get(algo_id, None)

        def _exec_add_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                file_infos = payload.get('file_infos')
                input_files = []
                ids = []
                metadatas = []

                for file_info in file_infos:
                    input_files.append(file_info.get('file_path'))
                    ids.append(file_info.get('doc_id'))
                    metadatas.append(file_info.get('metadata'))

                processor.add_doc(input_files=input_files, ids=ids, metadatas=metadatas)
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker - Impl] Task-{task_id}: execute add task failed, error: {e}')
                raise e

        def _exec_reparse_task(
            self, processor: _Processor, task_id: str, payload: dict
        ):
            try:
                file_infos = payload.get('file_infos')
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
                                  doc_paths=reparse_files, metadatas=reparse_metadatas)
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker - Impl] Task-{task_id}: execute reparse task failed, error: {e}')
                raise e

        def _exec_delete_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                kb_id = payload.get('kb_id')
                doc_ids = payload.get('doc_ids')
                processor.delete_doc(doc_ids=doc_ids, kb_id=kb_id)
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker - Impl] Task-{task_id}: execute delete task failed, error: {e}')
                raise e

        def _exec_update_meta_task(self, processor: _Processor, task_id: str, payload: dict):
            try:
                file_infos = payload.get('file_infos')
                for file_info in file_infos:
                    doc_id = file_info.get('doc_id')
                    metadata = file_info.get('metadata')
                    processor.update_doc_meta(doc_id=doc_id, metadata=metadata)
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker - Impl] Task-{task_id}: execute update meta task failed,'
                          f'error: {e}')
                raise e

        def _enqueue_finished_task(self, task_id: str, task_type: str, task_status: TaskStatus,
                                   error_code: str = None, error_msg: str = None):
            try:
                self._ensure_initialized()
                self._finished_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    task_status=task_status.value,
                    finished_at=datetime.now(),
                    error_code=error_code if error_code else '200',
                    error_msg=error_msg if error_msg else 'success'
                )
                if task_status == TaskStatus.FINISHED:
                    LOG.info(f'[DocumentProcessorWorker - Impl] Task {task_id} finished successfully')
                else:
                    LOG.error(f'[DocumentProcessorWorker - Impl] Task {task_id} completed with status {task_status}:'
                              f' {error_msg}')
            except Exception as e:
                LOG.error(f'[DocumentProcessorWorker - Impl] Failed to enqueue finished task {task_id}: {e}')

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
                        raise ValueError(f'[DocumentProcessorWorker - Impl] task_id {task_id} is missing algo_id in '
                                         f'payload: {payload}')

                    LOG.info(f'[DocumentProcessorWorker - Impl] Start processing task {task_id}, type: {task_type},'
                             f' algo_id: {algo_id}')

                    processor = self._get_or_create_processor(algo_id)
                    if not processor:
                        raise ValueError(f'[DocumentProcessorWorker - Impl] Failed to get or create processor for'
                                         f' algo_id: {algo_id}')
                    if task_type == TaskType.DOC_ADD.value:
                        self._exec_add_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_REPARSE.value:
                        self._exec_reparse_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_DELETE.value:
                        self._exec_delete_task(processor, task_id, payload)
                    elif task_type == TaskType.DOC_UPDATE_META.value:
                        self._exec_update_meta_task(processor, task_id, payload)
                    else:
                        raise ValueError(f'[DocumentProcessorWorker - Impl] Unknown task type: {task_type}')

                    self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FINISHED,
                                                error_code='200', error_msg='success')
                except Exception as e:
                    LOG.error(f'[DocumentProcessorWorker - Impl] Failed to run task {task_id}: {e},'
                              f' {traceback.format_exc()}')
                    if task_id and task_type:
                        self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FAILED,
                                                    error_code=type(e).__name__, error_msg=str(e))
                    time.sleep(WORKER_ERROR_RETRY_INTERVAL)
                    continue

        def start(self):
            LOG.info('[DocumentProcessorWorker - Impl] Starting worker...')
            self._ensure_initialized()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                LOG.warning('[DocumentProcessorWorker - Impl] Worker thread is already running')
                return
            self._shutdown = False
            self._worker_thread = threading.Thread(target=self._worker_impl, daemon=True)
            self._worker_thread.start()
            LOG.info('[DocumentProcessorWorker - Impl] Worker thread started')

        def shutdown(self):
            LOG.info('[DocumentProcessorWorker - Impl] Shutting down worker...')
            self._shutdown = True
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    LOG.warning('[DocumentProcessorWorker - Impl] Worker thread did not stop within timeout')
                else:
                    LOG.info('[DocumentProcessorWorker - Impl] Worker thread stopped')

    def __init__(self, db_config: dict = None, num_workers: int = 1, port: int = None, server_url: str = None):
        super().__init__()
        self._db_config = db_config if db_config else _get_default_db_config()
        self._num_workers = num_workers
        self._port = port
        self._server_url = server_url
        worker_impl = DocumentProcessorWorker.Impl(db_config=self._db_config, server_url=self._server_url)
        self._worker_impl = ServerModule(worker_impl, num_replicas=self._num_workers, port=self._port)
        LOG.info(f'[DocumentProcessorWorker] Worker initialized with {num_workers} workers')

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._worker_impl
        if isinstance(impl, ServerModule):
            return impl._call(method, *args, **kwargs)
        else:
            return getattr(impl, method)(*args, **kwargs)

    def start(self):
        result = super().start()
        self._dispatch('start')
        return result

    def stop(self):
        self._dispatch('shutdown')
        return super().stop()
