import threading
import time
import os
import traceback
import copy
import hashlib
import json
import requests
import subprocess
from collections import defaultdict
from datetime import datetime

from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import Future

from lazyllm import (
    LOG, ModuleBase, ServerModule, UrlModule, FastapiApp as app,
    ThreadPoolExecutor, config, LazyLLMLaunchersBase as Launcher
)
from lazyllm.thirdparty import fastapi

from .base import (
    FileInfo, AddDocRequest, UpdateMetaRequest,
    DeleteDocRequest, CancelDocRequest, TABLES_INFO,
    TaskType, TaskStatus, TaskCancelled,
)

from ..store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from ..store.store_base import DEFAULT_KB_ID
from ..store.document_store import _DocumentStore
from ..store.utils import fibonacci_backoff, create_file_path
from ..transform import (AdaptiveTransform, make_transform,)
from ..data_loaders import DirectoryReader
from ..doc_node import DocNode
from ..utils import gen_docid, ensure_call_endpoint, BaseResponse
from ..global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID

from ...sql import SqlManager

# Configuration constants
WORKER_POLL_INTERVAL = 1  # seconds
WORKER_ERROR_RETRY_INTERVAL = 5  # seconds
CALLBACK_TIMEOUT = 5  # seconds
CALLBACK_MAX_RETRIES = 3
EXECUTOR_SHUTDOWN_TIMEOUT = 45  # seconds to wait for tasks to gracefully exit


def _get_default_db_config():
    root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
    os.makedirs(root_dir, exist_ok=True)
    db_path = os.path.join(root_dir, 'lazyllm_doc_task_management.db')
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_path,
    }


class _Processor:
    def __init__(self, store: _DocumentStore, reader: DirectoryReader, node_groups: Dict[str, Dict],
                 display_name: Optional[str] = None, description: Optional[str] = None,
                 server: bool = False):
        self._store = store
        self._reader = reader
        self._node_groups = node_groups
        self._display_name = display_name
        self._description = description

    def _check_cancel(self, cancel_token: threading.Event):
        if cancel_token and not isinstance(cancel_token, threading.Event):
            raise ValueError('cancel_token must be an instance of threading.Event')
        if cancel_token and cancel_token.is_set():
            raise TaskCancelled('Task cancelled.')

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,
                metadatas: Optional[List[Dict[str, Any]]] = None, cancel_token: threading.Event = None):
        try:
            if not input_files: return
            if not ids: ids = [gen_docid(path) for path in input_files]
            if metadatas is None:
                metadatas = [{} for _ in input_files]
            for metadata, doc_id, path in zip(metadatas, ids, input_files):
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, path)
                metadata.setdefault(RAG_KB_ID, DEFAULT_KB_ID)
            self._check_cancel(cancel_token)
            root_nodes = self._reader.load_data(input_files, metadatas, split_nodes_by_type=True)
            self._check_cancel(cancel_token)
            for k, v in root_nodes.items():
                if not v: continue
                self._store.update_nodes(self._set_nodes_number(v))
                self._check_cancel(cancel_token)
                self._create_nodes_recursive(v, k, cancel_token)
            self._check_cancel(cancel_token)
            LOG.info('Add documents done!')
        except TaskCancelled as e:
            kb_id = metadatas[0].get(RAG_KB_ID, None)
            self._store.remove_nodes(doc_ids=ids, kb_id=kb_id)
            raise e
        except Exception as e:
            LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e

    def _set_nodes_number(self, nodes: List[DocNode]) -> List[DocNode]:
        doc_group_number = {}
        for node in nodes:
            doc_id = node.global_metadata.get(RAG_DOC_ID)
            group_name = node.group
            if doc_id not in doc_group_number:
                doc_group_number[doc_id] = {}
            if group_name not in doc_group_number[doc_id]:
                doc_group_number[doc_id][group_name] = 1
            node.metadata['lazyllm_store_num'] = doc_group_number[doc_id][group_name]
            doc_group_number[doc_id][group_name] += 1
        return nodes

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str, cancel_token: threading.Event = None):
        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group "{group_name}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')

            if group['parent'] == p_name:
                self._check_cancel(cancel_token)
                nodes = self._create_nodes_impl(p_nodes, group_name)
                if nodes: self._create_nodes_recursive(nodes, group_name, cancel_token)

    def _create_nodes_impl(self, p_nodes, group_name):
        # NOTE transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        t = self._node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, group_name)
        nodes = transform.batch_forward(p_nodes, group_name)
        self._store.update_nodes(self._set_nodes_number(nodes))
        return nodes

    def _get_or_create_nodes(self, group_name, uids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(uids=uids, group=group_name) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(self._node_groups[group_name]['parent'], uids)
            nodes = self._create_nodes_impl(p_nodes, group_name)
        return nodes

    def reparse(self, group_name: str, uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None,
                cancel_token: threading.Event = None, **kwargs):
        if doc_ids:
            self._reparse_docs(group_name=group_name, doc_ids=doc_ids, cancel_token=cancel_token, **kwargs)
        else:
            self._get_or_create_nodes(group_name, uids)

    def _reparse_docs(self, group_name: str, doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict],
                      cancel_token: threading.Event = None):
        kb_id = metadatas[0].get(RAG_KB_ID, None)
        if group_name == 'all':
            self._check_cancel(cancel_token)
            self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
            removed_flag = False
            for wait_time in fibonacci_backoff():
                self._check_cancel(cancel_token)
                nodes = self._store.get_nodes(group=LAZY_ROOT_NAME, kb_id=kb_id, doc_ids=doc_ids)
                if not nodes:
                    removed_flag = True
                    break
                time.sleep(wait_time)
            if not removed_flag:
                raise Exception(f'Failed to remove nodes for docs {doc_ids} from store')
            self._check_cancel(cancel_token)
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas, cancel_token=cancel_token)
            LOG.info(f'Reparse docs {doc_ids} from store done')
        else:
            self._check_cancel(cancel_token)
            p_nodes = self._store.get_nodes(group=self._node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._check_cancel(cancel_token)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name,
                                          doc_ids=doc_ids, cancel_token=cancel_token)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str, doc_ids: List[str],
                                 cancel_token: threading.Event = None):
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, None)
        self._check_cancel(cancel_token)
        self._store.remove_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)

        removed_flag = False
        for wait_time in fibonacci_backoff():
            self._check_cancel(cancel_token)
            nodes = self._store.get_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)
            if not nodes:
                removed_flag = True
                break
            time.sleep(wait_time)
        if not removed_flag:
            raise Exception(f'Failed to remove nodes for docs {doc_ids} group {cur_name} from store')
        self._check_cancel(cancel_token)
        t = self._node_groups[cur_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, cur_name)
        nodes = transform.batch_forward(p_nodes, cur_name)
        # reparse need set global_metadata
        self._check_cancel(cancel_token)
        self._store.update_nodes(self._set_nodes_number(nodes))
        self._check_cancel(cancel_token)

        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group "{group_name}" does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if group['parent'] == cur_name:
                self._check_cancel(cancel_token)
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name, doc_ids=doc_ids,
                                              cancel_token=cancel_token)

    def update_doc_meta(self, doc_id: str, metadata: dict, cancel_token: threading.Event = None):
        try:
            self._check_cancel(cancel_token)
            kb_id = metadata.get(RAG_KB_ID, None)
            segments = self._store.get_segments(doc_ids=[doc_id], kb_id=kb_id)
            if not segments:
                LOG.warning(f'No segments found for doc_id: {doc_id} in dataset: {kb_id}')
                raise ValueError(f'No segments found for doc_id: {doc_id} in dataset: {kb_id}')
            original_metadata = copy.deepcopy(segments[0].get('global_meta', {}))
            done_groups = set()
            self._check_cancel(cancel_token)
            group_segments = defaultdict(list)
            for segment in segments:
                segment['global_meta'].update(metadata)
                group_segments[segment.get('group')].append(segment)
            self._check_cancel(cancel_token)
            for group, segments in group_segments.items():
                self._check_cancel(cancel_token)
                self._store.impl.upsert(self._store._gen_collection_name(group), segments)
                done_groups.add(group)
        except TaskCancelled as e:
            # rollback the changes if task is cancelled
            if done_groups:
                for group in done_groups:
                    g_s = group_segments.get(group, [])
                    for segment in g_s:
                        segment['global_meta'].update(original_metadata)
                    self._store.impl.upsert(self._store._gen_collection_name(group), g_s)
            raise e
        except Exception as e:
            LOG.error(f'Failed to update doc meta: {e}, {traceback.format_exc()}')
            raise e

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None, cancel_token: threading.Event = None) -> None:
        try:
            self._check_cancel(cancel_token)
            self._store.remove_nodes(kb_id=kb_id, doc_ids=doc_ids)
        except TaskCancelled as e:
            LOG.warning(f'Delete doc cancelled for {doc_ids}')
            raise e
        except Exception as e:
            LOG.error(f'Failed to delete doc: {e}, {traceback.format_exc()}')
            raise e


class DocumentProcessor(ModuleBase):

    class Impl():
        def __init__(self, server: bool, db_config: Optional[Dict[str, Any]] = None, num_workers: int = 4):
            self._processors: Dict[str, _Processor] = dict()
            self._server = server
            self._inited = False
            self._db_config = db_config or _get_default_db_config()
            self._db_manager = None
            self._num_workers = num_workers
            self._executor_dict = None
            self._shutdown = False

            try:
                self._callback_url = config['doc_processor_callback_url']
                self._path_prefix = config['doc_processor_path_prefix']
            except Exception as e:
                LOG.warning(f'Failed to get config: {e}, use env variables instead')
                self._callback_url = os.getenv('DOC_PROCESSOR_CALLBACK_URL', None)
                self._path_prefix = os.getenv('DOC_PROCESSOR_PATH_PREFIX', None)

        def _init_components(self, server: bool):
            if server and not self._inited:
                self._db_manager = SqlManager(**self._db_config, tables_info_dict=TABLES_INFO)
                self._working_task_ids = set()
                # TODO(chenjiahao): A worker will be created when the service is started
                # starting independently should be supported in the future.
                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
                self._tasks_lock = threading.Lock()
            self._inited = True
            LOG.info(f'[DocumentProcessor] init done. callback {self._callback_url}, prefix {self._path_prefix}')

        def register_algorithm(self, name: str, store: _DocumentStore, reader: DirectoryReader,
                               node_groups: Dict[str, Dict], display_name: Optional[str] = None,
                               description: Optional[str] = None, force_refresh: bool = False):
            self._init_components(server=self._server)
            if name in self._processors and not force_refresh:
                LOG.warning(f'There is already a processor with the same name {name}!')
                return
            self._processors[name] = _Processor(store, reader, node_groups, display_name, description)
            LOG.info(f'Processor {name} registered!')

        def drop_algorithm(self, name: str, clean_db: bool = False) -> None:
            if name not in self._processors:
                LOG.warning(f'Processor {name} not found!')
                return
            self._processors.pop(name)

        def _generate_idempotency_key(self, algo_id: str, task_type: str, file_infos: List[FileInfo]) -> str:
            key_parts = [algo_id, task_type]
            for fi in sorted(file_infos, key=lambda x: (x.file_path or '', x.doc_id or '')):
                key_parts.append(f'{fi.file_path}:{fi.doc_id}:{fi.reparse_group or ""}')
            key_str = '|'.join(key_parts)
            return hashlib.md5(key_str.encode('utf-8')).hexdigest()

        def _orm_to_dict(self, obj):
            return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

        @app.get('/health')
        async def get_health(self) -> None:
            return BaseResponse(code=200, msg='success')

        @app.get('/prestop')
        async def get_prestop(self) -> None:
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
                    # Set shutdown flag to stop worker loop
                    # Worker thread will detect this flag and:
                    # 1. Stop accepting new tasks from database
                    # 2. Cancel all running tasks via cancel_tokens
                    # 3. Wait for tasks to exit gracefully (up to EXECUTOR_SHUTDOWN_TIMEOUT)
                    # 4. Shutdown thread pool executors in finally block
                    self._shutdown = True
                    LOG.info('[DocumentProcessor] Shutdown flag set, worker will handle graceful shutdown')

                    # Log worker thread status for monitoring
                    if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
                        with self._tasks_lock:
                            total_tasks = sum(len(tasks) for tasks in self._tasks.values())
                        LOG.info(f'[DocumentProcessor] Worker thread is active with {total_tasks} tasks, '
                                 f'will shutdown gracefully within {EXECUTOR_SHUTDOWN_TIMEOUT}s')
                    else:
                        LOG.info('[DocumentProcessor] Worker thread is not active')
                else:
                    LOG.info('[DocumentProcessor] Shutdown already initiated')

                # Return immediately - don't block the HTTP request
                # K8s terminationGracePeriodSeconds will give enough time for graceful shutdown
                LOG.info('[DocumentProcessor] PreStop hook returning, graceful shutdown in progress')
                return BaseResponse(code=200, msg='shutdown initiated')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] PreStop hook failed: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500,
                                            detail=f'PreStop hook failed: {e}, {traceback.format_exc()}')

        @app.get('/algo/list')
        async def get_algo_list(self) -> None:
            res = []
            for algo_id, processor in self._processors.items():
                res.append({'algo_id': algo_id, 'display_name': processor._display_name,
                            'description': processor._description})
            if not res:
                LOG.warning('[DocumentProcessor] No algorithm registered')
            return BaseResponse(code=200, msg='success', data=res)

        @app.get('/group/info')
        async def get_group_info(self, algo_id: str) -> None:
            processor = self._processors.get(algo_id)
            if processor is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            infos = []
            for group_name in processor._store.activated_groups():
                if group_name in processor._node_groups:
                    group_info = {'name': group_name, 'type': processor._node_groups[group_name].get('group_type'),
                                  'display_name': processor._node_groups[group_name].get('display_name')}
                    infos.append(group_info)
            LOG.info(f'Get group info for {algo_id} success with {infos}')
            return BaseResponse(code=200, msg='success', data=infos)

        @app.get('/task/{task_id}/info')
        async def get_task_info(self, task_id: str) -> None:
            t = self._get_task_by_id(task_id)
            if t is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Task {task_id} not found')
            return BaseResponse(code=200, msg='success', data=t)

        @app.get('/task/{task_id}/status')
        async def get_task_status(self, task_id: str) -> None:
            t = self._get_task_by_id(task_id)
            if t is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Task {task_id} not found')
            return BaseResponse(code=200, msg='success', data=t.get('task_status'))

        @app.post('/doc/add')
        async def async_add_doc(self, request: AddDocRequest):
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received add doc request: {payload}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            if algo_id not in self._processors:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')

            for file_info in file_infos:
                if self._path_prefix:
                    file_info.file_path = create_file_path(path=file_info.file_path, prefix=self._path_prefix)

            idempotency_key = self._generate_idempotency_key(algo_id, TaskType.DOC_ADD.value, file_infos)
            payload_json = json.dumps(payload, ensure_ascii=False)

            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskQueue = self._db_manager.get_table_orm_class('lazyllm_doc_task_queue')

                    existing_task = session.query(TaskInfo).filter(
                        TaskInfo.idempotency_key == idempotency_key
                    ).with_for_update().first()

                    if existing_task:
                        # existing task is working or cancel requested, return 409
                        if existing_task.task_status in (TaskStatus.WORKING.value, TaskStatus.CANCEL_REQUESTED.value):
                            raise fastapi.HTTPException(status_code=409,
                                                        detail=(f'Task {existing_task.task_id} with same content'
                                                                f' is {existing_task.task_status}'))

                        # static status (waiting, finished, failed, canceled), rerun it
                        LOG.info(f'[DocumentProcessor] Add doc task exists with task id {existing_task.task_id} '
                                 f'and status {existing_task.task_status}, rerun it')
                        existing_task.task_status = TaskStatus.WAITING.value
                        existing_task.payload = payload_json
                        existing_task.retries = (existing_task.retries or 0) + 1
                        # NOTE: task id will not be updated currently
                        task_id = existing_task.task_id
                        task_info = existing_task
                    else:
                        # create new task detail
                        new_task = TaskInfo(task_id=task_id, idempotency_key=idempotency_key, algo_id=algo_id,
                                            task_type=TaskType.DOC_ADD.value, task_status=TaskStatus.WAITING.value,
                                            payload=payload_json, retries=0, create_at=datetime.now())
                        session.add(new_task)
                        task_info = new_task

                    # flush to catch potential IntegrityError before querying TaskQueue
                    session.flush()

                    # add to task queue
                    queue_task = session.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()

                    if queue_task:
                        queue_task.create_at = datetime.now()
                    else:
                        new_queue_task = TaskQueue(task_id=task_id, task_type=TaskType.DOC_ADD.value,
                                                   create_at=datetime.now())
                        session.add(new_queue_task)

                LOG.info(f'[DocumentProcessor] Task {task_id} submitted to database queue successfully')
                return BaseResponse(code=200, msg='success', data=self._orm_to_dict(task_info))
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.post('/doc/meta/update')
        async def async_update_meta(self, request: UpdateMetaRequest):
            payload = request.model_dump()
            LOG.info(f'update doc meta for {payload}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos

            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            if algo_id not in self._processors:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')

            task_type = TaskType.DOC_UPDATE_META
            idempotency_key = self._generate_idempotency_key(algo_id, task_type.value, file_infos)
            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskQueue = self._db_manager.get_table_orm_class('lazyllm_doc_task_queue')

                    existing_task = session.query(TaskInfo).filter(
                        TaskInfo.idempotency_key == idempotency_key
                    ).with_for_update().first()

                    if existing_task:
                        if existing_task.task_status in (TaskStatus.WORKING.value, TaskStatus.CANCEL_REQUESTED.value):
                            raise fastapi.HTTPException(status_code=409,
                                                        detail=(f'Task {existing_task.task_id} with same content'
                                                                f' is {existing_task.task_status}'))
                        LOG.info(f'[DocumentProcessor] Update meta task exists with task id {existing_task.task_id} '
                                 f'and status {existing_task.task_status}, rerun it')
                        existing_task.task_status = TaskStatus.WAITING.value
                        existing_task.payload = payload_json
                        existing_task.retries = (existing_task.retries or 0) + 1
                        task_id = existing_task.task_id
                        task_info = existing_task
                    else:
                        new_task = TaskInfo(task_id=task_id, idempotency_key=idempotency_key, algo_id=algo_id,
                                            task_type=TaskType.DOC_UPDATE_META.value,
                                            task_status=TaskStatus.WAITING.value, payload=payload_json, retries=0,
                                            create_at=datetime.now())
                        session.add(new_task)
                        task_info = new_task
                    # flush to catch potential IntegrityError before querying TaskQueue
                    session.flush()
                    # add to task queue
                    queue_task = session.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()

                    if queue_task:
                        queue_task.create_at = datetime.now()
                    else:
                        new_queue_task = TaskQueue(task_id=task_id, task_type=TaskType.DOC_UPDATE_META.value,
                                                   create_at=datetime.now())
                        session.add(new_queue_task)
                LOG.info(f'[DocumentProcessor] Update meta task {task_id} submitted to database queue successfully')
                return BaseResponse(code=200, msg='success', data=self._orm_to_dict(task_info))
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit update meta task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.delete('/doc/delete')
        async def async_delete_doc(self, request: DeleteDocRequest) -> None:
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received delete doc request: {payload}')

            task_id = request.task_id
            algo_id = request.algo_id
            kb_id = request.kb_id
            doc_ids = request.doc_ids
            if not doc_ids:
                raise fastapi.HTTPException(status_code=400, detail='doc_ids is required')
            if algo_id not in self._processors:
                raise fastapi.HTTPException(status_code=404, detail=f'algo_id {algo_id} not found')

            # generate idempotency key
            key_parts = [algo_id, TaskType.DOC_DELETE.value, kb_id] + sorted(doc_ids)
            key_str = '|'.join(key_parts)
            idempotency_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()

            # prepare payload
            payload_json = json.dumps(payload, ensure_ascii=False)

            # use database transaction to handle idempotency and queue
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskQueue = self._db_manager.get_table_orm_class('lazyllm_doc_task_queue')

                    existing_task = session.query(TaskInfo).filter(
                        TaskInfo.idempotency_key == idempotency_key
                    ).with_for_update().first()

                    if existing_task:
                        if existing_task.task_status in (TaskStatus.WORKING.value, TaskStatus.CANCEL_REQUESTED.value):
                            raise fastapi.HTTPException(status_code=409,
                                                        detail=(f'Task {existing_task.task_id} with same content'
                                                                f' is {existing_task.task_status}'))
                        LOG.info(f'[DocumentProcessor] Delete doc task exists with task id {existing_task.task_id} '
                                 f'and status {existing_task.task_status}, rerun it')
                        existing_task.task_status = TaskStatus.WAITING.value
                        existing_task.payload = payload_json
                        existing_task.retries = (existing_task.retries or 0) + 1
                        task_id = existing_task.task_id
                        task_info = existing_task
                    else:
                        new_task = TaskInfo(task_id=task_id, idempotency_key=idempotency_key, algo_id=algo_id,
                                            task_type=TaskType.DOC_DELETE.value, task_status=TaskStatus.WAITING.value,
                                            payload=payload_json, retries=0, create_at=datetime.now())
                        session.add(new_task)
                        task_info = new_task
                    # flush to catch potential IntegrityError before querying TaskQueue
                    session.flush()
                    # add to task queue
                    queue_task = session.query(TaskQueue).filter(TaskQueue.task_id == task_id).first()

                    if queue_task:
                        queue_task.create_at = datetime.now()
                    else:
                        new_queue_task = TaskQueue(task_id=task_id, task_type=TaskType.DOC_DELETE.value,
                                                   create_at=datetime.now())
                        session.add(new_queue_task)
                LOG.info(f'[DocumentProcessor] Delete task {task_id} submitted to database queue successfully')
                return BaseResponse(code=200, msg='task submit successfully',
                                    data=self._orm_to_dict(task_info))
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to submit delete task: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to submit task: {str(e)}')

        @app.post('/doc/cancel')
        async def cancel_task(self, request: CancelDocRequest):
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received cancel task request: {payload}')
            task_id = request.task_id
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskQueue = self._db_manager.get_table_orm_class('lazyllm_doc_task_queue')

                    # query task info from db
                    task = session.query(TaskInfo).filter(
                        TaskInfo.task_id == task_id
                    ).with_for_update().first()

                    if not task:
                        raise fastapi.HTTPException(status_code=404, detail=f'Task {task_id} not found')
                    cancel_status = 0
                    if task.task_status == TaskStatus.WAITING.value:
                        # if task is waiting, directly delete queue record and update status
                        queue_task = session.query(TaskQueue).filter(
                            TaskQueue.task_id == task_id
                        ).first()
                        if queue_task:
                            session.delete(queue_task)
                        task.task_status = TaskStatus.CANCELED.value
                        task.finished_at = datetime.now()
                        session.commit()
                        cancel_status = 1
                        LOG.info(f'[DocumentProcessor] Task {task_id} canceled from WAITING state')
                    elif task.task_status == TaskStatus.WORKING.value:
                        # if task is working, change task status to cancel_requested, future will stop running later
                        task.task_status = TaskStatus.CANCEL_REQUESTED.value
                        cancel_status = 1
                        LOG.info(f'[DocumentProcessor] Task {task_id} is working, cancel request submitted')
                    elif task.task_status == TaskStatus.CANCEL_REQUESTED.value:
                        # if task is cancel requested, do nothing
                        cancel_status = 0
                        LOG.info(f'[DocumentProcessor] Task {task_id} is cancel requested, cannot cancel again')
                        raise fastapi.HTTPException(status_code=409,
                                                    detail=f'Task {task_id} is cancel requested, cannot cancel again')
                    else:
                        # task is finished or canceled
                        cancel_status = 1 if task.task_status == TaskStatus.CANCELED.value else 0
                        LOG.info(f'Task {task_id} is in {task.task_status} state, cannot cancel')
                    return BaseResponse(
                        code=200,
                        msg='success' if cancel_status else 'failed',
                        data={'task_id': task_id, 'cancel_status': cancel_status, 'task_status': task.task_status}
                    )
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to cancel task {task_id}: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to cancel task: {str(e)}')

        def _callback(self, task_id: str, task_status: TaskStatus, error_code: str = '', error_msg: str = ''):
            '''callback to service'''
            if self._callback_url:
                try:
                    payload = {'task_id': task_id, 'status': 1 if task_status == TaskStatus.FINISHED else 0,
                               'error_code': error_code, 'error_msg': error_msg}
                    headers = {'Content-Type': 'application/json'}
                    res = None
                    for wait_time in fibonacci_backoff(max_retries=CALLBACK_MAX_RETRIES):
                        try:
                            res = requests.post(self._callback_url, json=payload, headers=headers,
                                                timeout=CALLBACK_TIMEOUT)
                            if res.status_code == 200:
                                break
                            LOG.warning(
                                f'Task-{task_id}: Unexpected status {res.status_code}, retrying in {wait_time}s…')
                        except Exception as e:
                            LOG.error(f'Task-{task_id}: Request failed: {e}, retrying in {wait_time}s…')
                        time.sleep(wait_time)

                    if res is None:
                        raise RuntimeError('Failed to send callback—no response received after retries')
                    res.raise_for_status()
                except Exception as e:
                    LOG.error(f'Task-{task_id}: Failed to send callback to {self._callback_url}: {e}')

        def _attach_done(self, task_id: str, future: Future):
            '''attach done callback to future'''
            def _on_done(fut):
                ok = False
                err = None
                err_code = None
                task_status = TaskStatus.FINISHED
                try:
                    ex = fut.exception()
                    if ex:
                        if isinstance(ex, TaskCancelled):
                            err = 'canceled'
                            err_code = 'Canceled'
                            task_status = TaskStatus.CANCELED
                        else:
                            err = str(ex)
                            err_code = type(ex).__name__
                            task_status = TaskStatus.FAILED
                    else:
                        ok = True
                except Exception as e:
                    err = f'exception() failed: {e}'
                    err_code = type(e).__name__
                try:
                    if ok:
                        self._update_task_status(task_id, task_status, '200', 'success')
                        LOG.info(f'[Worker] Task {task_id} finished successfully, status: {task_status}')
                    else:
                        self._update_task_status(task_id, task_status, err_code, err)
                        LOG.error(f'[Worker] Task {task_id} failed: {err}, status: {task_status}')
                except Exception as e:
                    LOG.error(f'[DoneCB] update task status failed for {task_id}: {e}')
                self._callback(task_id, task_status, err_code, err)
            future.add_done_callback(_on_done)

        def _exec_add_task(self, algo_id: str, task_id: str, payload: dict, cancel_token: threading.Event):
            try:
                file_infos = payload.get('file_infos')
                input_files = []
                ids = []
                metadatas = []

                reparse_group = None
                reparse_doc_ids = []
                reparse_files = []
                reparse_metadatas = []

                for file_info in file_infos:
                    if file_info.get('reparse_group'):
                        reparse_group = file_info.get('reparse_group')
                        reparse_doc_ids.append(file_info.get('doc_id'))
                        reparse_files.append(file_info.get('file_path'))
                        reparse_metadatas.append(file_info.get('metadata'))
                    else:
                        input_files.append(file_info.get('file_path'))
                        ids.append(file_info.get('doc_id'))
                        metadatas.append(file_info.get('metadata'))
                if input_files and reparse_group:
                    LOG.error(f'Input files {input_files} and reparse group {reparse_group} '
                              'cannot be specified at the same time')
                    raise ValueError('Input files and reparse group cannot be specified at the same time')
                if input_files:
                    self._processors[algo_id].add_doc(input_files=input_files, ids=ids, metadatas=metadatas,
                                                      cancel_token=cancel_token)
                elif reparse_group:
                    self._processors[algo_id].reparse(group_name=reparse_group, doc_ids=reparse_doc_ids,
                                                      doc_paths=reparse_files, metadatas=reparse_metadatas,
                                                      cancel_token=cancel_token)
                else:
                    LOG.error(f'No input files {input_files} or reparse group {reparse_group}')
                    raise ValueError('No input files or reparse group specified')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Task-{task_id}: execute add task failed, error: {e}')
                raise e

        def _exec_delete_task(self, algo_id: str, task_id: str, payload: dict, cancel_token: threading.Event):
            try:
                dataset_id = payload.get('dataset_id')
                doc_ids = payload.get('doc_ids')
                self._processors[algo_id].delete_doc(kb_id=dataset_id, doc_ids=doc_ids, cancel_token=cancel_token)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Task-{task_id}: execute delete task failed, error: {e}')
                raise e

        def _exec_update_meta_task(self, algo_id: str, task_id: str, payload: dict, cancel_token: threading.Event):
            try:
                file_infos = payload.get('file_infos')
                for file_info in file_infos:
                    doc_id = file_info.get('doc_id')
                    metadata = file_info.get('metadata')
                    self._processors[algo_id].update_doc_meta(doc_id=doc_id, metadata=metadata,
                                                              cancel_token=cancel_token)
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Task-{task_id}: execute update meta task failed, error: {e}')
                raise e

        def _get_task_by_id(self, task_id: str):
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    t = session.query(TaskInfo).filter(TaskInfo.task_id == task_id).first()
                    return None if not t else self._orm_to_dict(t)
            except Exception as e:
                LOG.error(f'Failed to get task by id: {e}, {traceback.format_exc()}')
                return None

        def _fetch_task_from_db(self, task_type: str):
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskQueue = self._db_manager.get_table_orm_class('lazyllm_doc_task_queue')
                    TaskRecord = self._db_manager.get_table_orm_class('lazyllm_doc_task_record')
                    # get a task from queue (by priority and create time)
                    queue_task = session.query(TaskQueue).filter(
                        TaskQueue.task_type == task_type
                    ).order_by(
                        TaskQueue.priority.desc(),
                        TaskQueue.create_at.asc()
                    ).first()
                    if not queue_task:
                        return None
                    task_id = queue_task.task_id
                    # get task info
                    task_info = session.query(TaskInfo).filter(
                        TaskInfo.task_id == task_id
                    ).with_for_update().first()
                    if not task_info:
                        # task info not found, remove from queue
                        session.delete(queue_task)
                        LOG.warning(f'[DocumentProcessor] Task {task_id} not found in task_info, removed from queue')
                        return None
                    # check task status
                    if task_info.task_status != TaskStatus.WAITING.value:
                        # not waiting state, remove from queue
                        session.delete(queue_task)
                        LOG.info(f'Task {task_id} status is {task_info.task_status}, removed from queue')
                        return None
                    task_record = TaskRecord(task_id=task_id, from_status=task_info.task_status,
                                             to_status=TaskStatus.WORKING.value, create_at=datetime.now(),
                                             attempt_seq=task_info.retries or 0, reason='')
                    # update task status to WORKING
                    task_info.task_status = TaskStatus.WORKING.value
                    task_info.started_at = datetime.now()
                    task_info.finished_at = None
                    task_info.lease_owner = self._worker_id
                    # remove from queue
                    session.delete(queue_task)
                    # add task record
                    session.add(task_record)
                    # parse payload
                    payload = json.loads(task_info.payload)

                    return {'task_id': task_id, 'task_type': task_info.task_type,
                            'algo_id': task_info.algo_id, 'payload': payload}
            except Exception as e:
                LOG.error(f'Failed to fetch task from database: {e}, {traceback.format_exc()}')
                return None

        def _update_task_status(self, task_id: str, status: TaskStatus, error_code: str = None, error_msg: str = None):
            '''update task status'''
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    TaskRecord = self._db_manager.get_table_orm_class('lazyllm_doc_task_record')

                    # update task info
                    task = session.query(TaskInfo).filter(TaskInfo.task_id == task_id).first()

                    if task:
                        original_status = task.task_status
                        record = TaskRecord(task_id=task_id, from_status=original_status, to_status=status.value,
                                            create_at=datetime.now(), attempt_seq=task.retries, reason='')
                        task.task_status = status.value
                        if status in (TaskStatus.FINISHED, TaskStatus.FAILED, TaskStatus.CANCELED):
                            task.finished_at = datetime.now()
                            if error_code is not None:
                                task.error_code = error_code
                                record.reason += f'error_code: {error_code}\n'
                            if error_msg is not None:
                                task.error_msg = error_msg[:512]  # limit length
                                record.reason += f'error_msg: {error_msg[:512]}'
                        LOG.info(f'[DocumentProcessor] Task {task_id} status updated to {status}')
                        session.add(record)
                    else:
                        raise ValueError(f'Task {task_id} not found')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to update task status: {e}, {traceback.format_exc()}')

        def _run_task_wrapper(self, executor: ThreadPoolExecutor, algo_id: str,
                              task_id: str, payload: dict, func: Callable):
            '''run task wrapper'''
            cancel_token = threading.Event()
            with self._tasks_lock:
                self._cancel_tokens[task_id] = cancel_token
            future = executor.submit(func, algo_id, task_id, payload, cancel_token)
            self._attach_done(task_id, future)
            return future

        def _check_cancellations(self):
            '''check cancellations for tasks owned by this worker'''
            try:
                with self._db_manager.get_session() as session:
                    TaskInfo = self._db_manager.get_table_orm_class('lazyllm_doc_task_detail')
                    cancel_requested_tasks = session.query(TaskInfo).filter(
                        TaskInfo.task_status == TaskStatus.CANCEL_REQUESTED.value,
                        TaskInfo.lease_owner == self._worker_id
                    ).all()
                    for task in cancel_requested_tasks:
                        task_id = task.task_id
                        with self._tasks_lock:
                            if task_id in self._cancel_tokens and not self._cancel_tokens[task_id].is_set():
                                LOG.info(f'[Worker] Task {task_id} cancel requested, setting cancel token')
                                self._cancel_tokens[task_id].set()
            except Exception as e:
                LOG.error(f'[Worker] Failed to check cancellations: {e}, {traceback.format_exc()}')

        def _shutdown_executors(self):
            '''Gracefully shutdown all thread pool executors'''
            if self._executor_dict:
                LOG.info('[DocumentProcessor] Shutting down executors...')

                # Step 1: Set all cancel tokens to signal running tasks to stop
                LOG.info('[DocumentProcessor] Signaling all running tasks to cancel...')
                with self._tasks_lock:
                    cancel_count = 0
                    for _, cancel_token in self._cancel_tokens.items():
                        if not cancel_token.is_set():
                            cancel_token.set()
                            cancel_count += 1
                    LOG.info(f'[DocumentProcessor] Signaled {cancel_count} tasks to cancel')

                # Step 2: Wait a reasonable time for tasks to gracefully exit
                LOG.info(f'[DocumentProcessor] Waiting up to {EXECUTOR_SHUTDOWN_TIMEOUT}s for tasks to exit...')
                start_time = time.time()
                while time.time() - start_time < EXECUTOR_SHUTDOWN_TIMEOUT:
                    with self._tasks_lock:
                        # Count only tasks that are not yet done
                        total_running = sum(
                            sum(1 for future in tasks.values() if not future.done())
                            for tasks in self._tasks.values()
                        )
                    if total_running == 0:
                        LOG.info('[DocumentProcessor] All tasks exited gracefully')
                        break
                    LOG.debug(f'[DocumentProcessor] {total_running} tasks still running, waiting...')
                    time.sleep(1)
                else:
                    with self._tasks_lock:
                        # Count only tasks that are not yet done
                        total_running = sum(
                            sum(1 for future in tasks.values() if not future.done())
                            for tasks in self._tasks.values()
                        )
                    if total_running > 0:
                        LOG.warning(f'[DocumentProcessor] {total_running} tasks still running after timeout')

                # Step 3: Shutdown executors (cancel remaining futures in queue)
                for task_type, (executor, _) in self._executor_dict.items():
                    try:
                        # cancel_futures=True will cancel tasks waiting in queue
                        # wait=False will not block, as we already waited above
                        executor.shutdown(wait=False, cancel_futures=True)
                        LOG.info(f'[DocumentProcessor] Executor for {task_type} shutdown successfully')
                    except Exception as e:
                        LOG.error(f'[DocumentProcessor] Failed to shutdown executor for {task_type}: {e}')
                self._executor_dict = None

        def _worker(self):  # noqa: C901
            LOG.info(f'[DocumentProcessor] Worker started with {self._num_workers} workers')
            try:
                node = subprocess.check_output(['hostname', '-i']).decode().strip()
                if ' ' in node: node = node.split()[0]
            except Exception:
                node = subprocess.check_output(['hostname']).decode().strip()
            self._worker_id = f'worker_{node}'
            self._executor_dict = {
                TaskType.DOC_ADD.value: (ThreadPoolExecutor(max_workers=self._num_workers), self._exec_add_task),
                TaskType.DOC_DELETE.value: (ThreadPoolExecutor(max_workers=self._num_workers), self._exec_delete_task),
                TaskType.DOC_UPDATE_META.value: (ThreadPoolExecutor(max_workers=self._num_workers),
                                                 self._exec_update_meta_task),
            }
            self._tasks: Dict[str, Dict[str, Future]] = {
                TaskType.DOC_ADD.value: {},
                TaskType.DOC_DELETE.value: {},
                TaskType.DOC_UPDATE_META.value: {}
            }
            self._cancel_tokens: Dict[str, threading.Event] = {}

            try:
                while not self._shutdown:
                    try:
                        # clear finished tasks (thread-safe)
                        with self._tasks_lock:
                            for task_type, tasks in self._tasks.items():
                                done_ids = []
                                for task_id, future in list(tasks.items()):
                                    if future.done():
                                        done_ids.append(task_id)
                                for tid in done_ids:
                                    self._tasks[task_type].pop(tid, None)
                                    self._cancel_tokens.pop(tid, None)

                        # if there are free workers, fetch task from database
                        for task_type, (executor, func) in self._executor_dict.items():
                            with self._tasks_lock:
                                current_tasks = len(self._tasks[task_type])

                            if current_tasks < self._num_workers:
                                # fetch task from database and set working status
                                task_data = self._fetch_task_from_db(task_type=task_type)
                                if not task_data:
                                    continue
                                task_id = task_data['task_id']
                                algo_id = task_data['algo_id']
                                payload = task_data['payload']
                                LOG.info(f'[DocumentProcessor] Worker fetched task {task_id} ({task_type})')

                                # submit task execution and attach done callback
                                future = self._run_task_wrapper(executor, algo_id, task_id, payload, func)
                                with self._tasks_lock:
                                    self._tasks[task_type][task_id] = future

                        # check cancel task
                        self._check_cancellations()
                        # sleep for configured interval
                        time.sleep(WORKER_POLL_INTERVAL)

                    except Exception as e:
                        LOG.error(f'[Worker] Worker loop error: {e}\n{traceback.format_exc()}')
                        time.sleep(WORKER_ERROR_RETRY_INTERVAL)
            finally:
                # Cleanup on exit
                self._shutdown_executors()
                LOG.info('[DocumentProcessor] Worker thread terminated')

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(self, server: bool = True, port: int = None, url: str = None, num_workers: int = 4,
                 db_config: Optional[Dict[str, Any]] = None, launcher: Optional[Launcher] = None):
        super().__init__()
        if not url:
            self._impl = DocumentProcessor.Impl(server=server, num_workers=num_workers, db_config=db_config)
            if server:
                self._impl = ServerModule(self._impl, port=port, launcher=launcher)
        else:
            self._impl = UrlModule(url=ensure_call_endpoint(url))

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._impl
        if isinstance(impl, ServerModule):
            impl._call(method, *args, **kwargs)
        else:
            getattr(impl, method)(*args, **kwargs)

    def register_algorithm(self, name: str, store: _DocumentStore, reader: DirectoryReader,
                           node_groups: Dict[str, Dict], display_name: Optional[str] = None,
                           description: Optional[str] = None, force_refresh: bool = False, **kwargs):
        assert isinstance(reader, DirectoryReader), 'Only DirectoryReader can be registered to processor'
        self._dispatch('register_algorithm', name, store, reader, node_groups,
                       display_name, description, force_refresh, **kwargs)

    def drop_algorithm(self, name: str, clean_db: bool = False) -> None:
        return self._dispatch('drop_algorithm', name, clean_db)
