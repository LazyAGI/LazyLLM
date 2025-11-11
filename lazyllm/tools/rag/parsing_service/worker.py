import json
import time
import traceback

import cloudpickle

from lazyllm import LOG, FastapiApp as app
from ..utils import BaseResponse
from .base import (
    FINISHED_TASK_QUEUE_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO,
    TaskStatus, TaskType
)
from .impl import _Processor, _get_default_db_config
from .queue import SQLBasedQueue as Queue

WORKER_ERROR_RETRY_INTERVAL = 5.0


class DocumentProcessorWorker:
    def __init__(self, db_config: dict = None, num_workers: int = 1, server_url: str = None):
        self._db_config = db_config if db_config else _get_default_db_config()
        self._num_workers = num_workers
        self._server_url = server_url
        self._server = None
        self._shutdown = False
        self._processors: dict[str, _Processor] = {}  # algo_id -> _Processor

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
        LOG.info(f'[DocumentProcessorWorker] Worker initialized with {num_workers} workers')

    @property
    def server(self):
        if self._server is None and self._server_url:
            from .server import DocumentProcessor
            self._server = DocumentProcessor(url=self._server_url)
            LOG.info(f'[DocumentProcessorWorker] Initialized DocumentProcessor client with url: {self._server_url}')
        return self._server

    @app.get('/health')
    def get_health(self):
        return BaseResponse(code=200, msg='success')

    @app.get('/prestop')
    def get_prestop(self):
        self._shutdown = True
        return BaseResponse(code=200, msg='success')

    def _get_or_create_processor(self, algo_id: str):
        if algo_id in self._processors:
            return self._processors[algo_id]
        try:
            if not self.server:
                raise ValueError('Server is not initialized')
            LOG.info(f'[Worker] Trying to load algo {algo_id} from server...')
            response = self.server.get_algo_info(algo_id)
            if response.code != 200:
                raise ValueError(f'Failed to get algo info: {response.msg}')
            data = response.data
            name = data.get('name')
            if name != algo_id:
                raise ValueError(f'Algo name {name} does not match the requested algo_id {algo_id}')
            display_name = data.get('display_name')
            description = data.get('description')
            hash_key = data.get('hash_key')
            info_pickle = data.get('info_pickle')
            info = cloudpickle.loads(info_pickle)
            store = info['store']
            reader = info['reader']
            node_groups = info['node_groups']
            self._processors[algo_id] = _Processor(store, reader, node_groups, display_name, description, hash_key)
            LOG.info(f'[DocumentProcessorWorker] Created processor for {algo_id} from server')
            return self._processors[algo_id]
        except Exception as e:
            LOG.warning(f'[DocumentProcessorWorker] Failed to load algo from server: {e}')
            raise e

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
            LOG.error(f'[DocumentProcessorWorker] Task-{task_id}: execute add task failed, error: {e}')
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

            for file_info in file_infos:
                reparse_group = file_info.get('reparse_group')
                reparse_doc_ids.append(file_info.get('doc_id'))
                reparse_files.append(file_info.get('file_path'))
                reparse_metadatas.append(file_info.get('metadata'))

            processor.reparse(group_name=reparse_group, doc_ids=reparse_doc_ids,
                              doc_paths=reparse_files, metadatas=reparse_metadatas)
        except Exception as e:
            LOG.error(f'[DocumentProcessorWorker] Task-{task_id}: execute reparse task failed, error: {e}')
            raise e

    def _exec_delete_task(self, processor: _Processor, task_id: str, payload: dict):
        try:
            kb_id = payload.get('kb_id')
            doc_ids = payload.get('doc_ids')
            processor.delete_doc(doc_ids=doc_ids, kb_id=kb_id)
        except Exception as e:
            LOG.error(f'[DocumentProcessorWorker] Task-{task_id}: execute delete task failed, error: {e}')
            raise e

    def _exec_update_meta_task(self, processor: _Processor, task_id: str, payload: dict):
        try:
            file_infos = payload.get('file_infos')
            for file_info in file_infos:
                doc_id = file_info.get('doc_id')
                metadata = file_info.get('metadata')
                processor.update_doc_meta(doc_id=doc_id, metadata=metadata)
        except Exception as e:
            LOG.error(f'[DocumentProcessor] Task-{task_id}: execute update meta task failed, error: {e}')
            raise e

    def _enqueue_finished_task(self, task_id: str, task_type: str, task_status: TaskStatus,
                               error_code: str = None, error_msg: str = None):
        try:
            self._finished_task_queue.enqueue(
                task_id=task_id,
                task_type=task_type,
                task_status=task_status.value,
                error_code=error_code if error_code else '200',
                error_msg=error_msg if error_msg else 'success'
            )
            if task_status == TaskStatus.FINISHED:
                LOG.info(f'[Worker] Task {task_id} finished successfully')
            else:
                LOG.error(f'[Worker] Task {task_id} completed with status {task_status}: {error_msg}')
        except Exception as e:
            LOG.error(f'[Worker] Failed to enqueue finished task {task_id}: {e}')

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
                    raise ValueError(f'[Worker] task_id {task_id} is missing algo_id in payload: {payload}')

                LOG.info(f'[Worker] Start processing task {task_id}, type: {task_type}, algo_id: {algo_id}')

                processor = self._get_or_create_processor(algo_id)
                if task_type == TaskType.DOC_ADD.value or task_type == TaskType.DOC_REPARSE.value:
                    self._exec_add_task(processor, task_id, payload)
                elif task_type == TaskType.DOC_DELETE.value:
                    self._exec_delete_task(processor, task_id, payload)
                elif task_type == TaskType.DOC_UPDATE_META.value:
                    self._exec_update_meta_task(processor, task_id, payload)
                else:
                    raise ValueError(f'[Worker] Unknown task type: {task_type}')

                self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FINISHED,
                                            error_code='200', error_msg='success')
            except Exception as e:
                LOG.error(f'[Worker] Failed to run task {task_id}: {e}, {traceback.format_exc()}')
                if task_id and task_type:
                    self._enqueue_finished_task(task_id=task_id, task_type=task_type, task_status=TaskStatus.FAILED,
                                                error_code=type(e).__name__, error_msg=str(e))
                time.sleep(WORKER_ERROR_RETRY_INTERVAL)
                continue

    def start(self):
        LOG.info('[DocumentProcessorWorker] Starting worker...')
        self._worker_impl()

    def shutdown(self):
        LOG.info('[DocumentProcessorWorker] Shutting down worker...')
        self._shutdown = True
