import base64
import json
import threading
import time
import traceback
import cloudpickle
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from lazyllm import (
    LOG, ModuleBase, ServerModule, UrlModule, FastapiApp as app,
    LazyLLMLaunchersBase as Launcher
)
from lazyllm.thirdparty import fastapi

from .base import (
    ALGORITHM_TABLE_INFO, WAITING_TASK_QUEUE_TABLE_INFO, FINISHED_TASK_QUEUE_TABLE_INFO, TaskStatus,
    TaskType, UpdateMetaRequest, AddDocRequest, CancelDocRequest, DeleteDocRequest, calculate_task_score
)
from .impl import _Processor, _get_default_db_config
from .worker import DocumentProcessorWorker as Worker
from .queue import SQLBasedQueue as Queue

from ..data_loaders import DirectoryReader
from ..store.document_store import _DocumentStore
from ..store.utils import create_file_path
from ..utils import BaseResponse, ensure_call_endpoint
from ...sql import SqlManager


class DocumentProcessor(ModuleBase):

    class Impl():
        def __init__(self, server: bool, db_config: Optional[Dict[str, Any]] = None, num_workers: int = 1,
                     post_func: Optional[Callable] = None, path_prefix: Optional[str] = None):
            self._processors: Dict[str, _Processor] = dict()
            self._server = server
            self._db_config = db_config
            self._num_workers = num_workers
            self._post_func = post_func
            if not self._assert_post_func():
                raise ValueError('Invalid post function!')
            self._shutdown = False
            self._path_prefix = path_prefix

            self._db_manager = None
            self._waiting_task_queue = None
            self._finished_task_queue = None
            self._post_func_thread = None
            self._refresh_algo_thread = None
            self._workers = None
            self._initialized = False

        def _ensure_initialized(self, server_url: str = None):
            if self._initialized:
                return

            LOG.info('[DocumentProcessor] Starting lazy initialization...')
            self._initialized = True

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
            )

            self._post_func_thread = threading.Thread(target=self.process_finished_task, daemon=True)
            self._post_func_thread.start()
            self._refresh_algo_thread = threading.Thread(target=self._refresh_algorithms, daemon=True)
            self._refresh_algo_thread.start()

            if self._num_workers > 0:
                self._workers = Worker(db_config=self._db_config, num_workers=self._num_workers, server_url=server_url)
                self._workers.start()
            LOG.info('[DocumentProcessor] Lazy initialization completed!')

        def __getstate__(self):
            state = self.__dict__.copy()
            state['_db_manager'] = None
            state['_waiting_task_queue'] = None
            state['_finished_task_queue'] = None
            state['_post_func_thread'] = None
            state['_refresh_algo_thread'] = None
            state['_workers'] = None
            state['_initialized'] = False
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)

        def _refresh_algorithms(self) -> None:
            while True:
                try:
                    self._refresh_algorithms_impl()
                    time.sleep(10)
                except Exception as e:
                    LOG.error(f'[DocumentProcessor] Failed to refresh algorithms: {e}, {traceback.format_exc()}')
                    time.sleep(10)

        def _refresh_algorithms_impl(self) -> None:
            self._ensure_initialized()
            try:
                active_algorithms_dict = {}
                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    # Query algorithms where instance_keys is not empty (is_active is auto-calculated)
                    query_algos = session.query(AlgoInfo).all()
                    for algo in query_algos:
                        # Check if algorithm is active by checking instance_keys
                        instance_keys_str = algo.instance_keys or '[]'
                        try:
                            instance_keys_list = json.loads(instance_keys_str)
                            is_active = len(instance_keys_list) > 0
                        except (json.JSONDecodeError, TypeError):
                            # Fallback to is_active field for backward compatibility
                            is_active = algo.is_active if hasattr(algo, 'is_active') else True

                        if is_active:
                            active_algorithms_dict[algo.id] = {
                                'display_name': algo.display_name,
                                'description': algo.description,
                                'info_pickle': algo.info_pickle,
                                'version': algo.version,
                                'instance_keys': instance_keys_list if isinstance(instance_keys_list, list) else []
                            }

                LOG.info(f'[DocumentProcessor._refresh_algorithms] Get {len(active_algorithms_dict)} active algorithms')
                for algo_id, algo_detail in active_algorithms_dict.items():
                    # check if the algorithm is already registered and content hasn't changed
                    if not self._processors.get(algo_id):
                        display_name = algo_detail['display_name']
                        description = algo_detail['description']
                        version = algo_detail['version']
                        info_dict = cloudpickle.loads(algo_detail['info_pickle'])
                        store = info_dict['store']
                        reader = info_dict['reader']
                        node_groups = info_dict['node_groups']
                        processor = _Processor(store, reader, node_groups, display_name, description, version)
                        self._processors[algo_id] = processor
                        LOG.info(f'[DocumentProcessor._refresh_algorithms] Algorithm {algo_id} {version} registered!')
                # drop inactive algorithms
                for algo_id in self._processors.keys():
                    if algo_id not in active_algorithms_dict:
                        self._processors.pop(algo_id)
                        LOG.info(f'[DocumentProcessor._refresh_algorithms] Algorithm {algo_id} dropped!')
            except Exception as e:
                LOG.error(f'Failed to refresh algorithms: {e}, {traceback.format_exc()}')
                raise e

        def process_finished_task(self):
            '''process finished task in background thread'''
            while True:
                try:
                    finished_task = self._finished_task_queue.dequeue()
                    if finished_task:
                        task_id = finished_task['task_id']
                        task_status = finished_task['task_status']
                        error_code = finished_task['error_code']
                        error_msg = finished_task['error_msg']
                        self._callback(task_id, task_status, error_code, error_msg)
                        time.sleep(0.1)
                    else:
                        time.sleep(1)
                except Exception as e:
                    LOG.error(f'[DocumentProcessor] Failed to process finished task: {e}, {traceback.format_exc()}')
                    time.sleep(10)

        def register_algorithm(self, name: str, store: _DocumentStore, reader: DirectoryReader,
                               node_groups: Dict[str, Dict], display_name: Optional[str] = None,
                               description: Optional[str] = None, version: Optional[str] = '1.0.0',
                               instance_key: Optional[str] = None):
            # NOTE: name is the algorithm id, display_name is the algorithm display name
            self._ensure_initialized()
            LOG.info((f'[DocumentProcessor] Get register algorithm request: name={name},'
                      f'display_name={display_name}, description={description}, version={version},'
                      f'instance_keys={instance_key}'))
            # write the processor to database
            try:
                info_dict = {
                    'store': store,
                    'reader': reader,
                    'node_groups': node_groups
                }
                info_pickle = cloudpickle.dumps(info_dict)
                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    existing_algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == name).first()
                    if not existing_algorithm:
                        # new algorithm
                        new_algo_info = AlgoInfo(
                            id=name,
                            display_name=display_name,
                            description=description,
                            version=version,
                            instance_keys=json.dumps([instance_key]),
                            info_pickle=info_pickle,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            is_active=True,
                        )
                        session.add(new_algo_info)
                    else:
                        # existing algorithm
                        existing_instance_keys = json.loads(existing_algorithm.instance_keys)
                        if existing_algorithm.version != version:
                            if len(existing_instance_keys) > 0:
                                # existing algorithm with different version and still serving
                                msg = (f'[DocumentProcessor] Algorithm {name} already registered with'
                                       f'different version and still serving(instance_keys={existing_instance_keys})')
                                LOG.warning(msg)
                                raise ValueError(msg)
                            else:
                                # existing algorithm with same version and no instance keys
                                existing_algorithm.version = version
                                existing_algorithm.instance_keys = json.dumps([instance_key])
                                existing_algorithm.info_pickle = info_pickle
                                existing_algorithm.display_name = display_name
                                existing_algorithm.description = description
                                existing_algorithm.updated_at = datetime.now()
                                existing_algorithm.is_active = True
                                self._processors.pop(name)
                        else:
                            # existing algorithm with same version and instance keys
                            existing_instance_keys.append(instance_key)
                            existing_algorithm.instance_keys = json.dumps(existing_instance_keys)
                            existing_algorithm.is_active = True
                            LOG.info(f'[DocumentProcessor] Algorithm {name} already registered with same version and'
                                     f'instance keys(instance_keys={existing_instance_keys})')
                if name not in self._processors:
                    processor = _Processor(store, reader, node_groups, display_name, description, version)
                    self._processors[name] = processor
                LOG.info(f'[DocumentProcessor] Algorithm {name} {version} {instance_key} registered!')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to register algorithm: {e}, {traceback.format_exc()}')
                raise e

        def drop_algorithm(self, name: str, version: Optional[str] = None, instance_key: Optional[str] = None,
                           force: bool = False) -> None:
            try:
                self._ensure_initialized()
                with self._db_manager.get_session() as session:
                    AlgoInfo = self._db_manager.get_table_orm_class('lazyllm_algorithm')
                    existing_algorithm = session.query(AlgoInfo).filter(AlgoInfo.id == name).first()
                    if existing_algorithm:
                        if force:
                            # force drop the algorithm
                            self._processors.pop(name)
                            existing_algorithm.is_active = False
                            existing_algorithm.instance_keys = json.dumps([])
                            existing_algorithm.updated_at = datetime.now()
                            LOG.info(f'[DocumentProcessor] Algorithm {name} dropped (force={force})!')
                            return

                        if existing_algorithm.version != version:
                            raise ValueError(f'[DocumentProcessor] Algorithm {name} registered with different version')
                        instance_keys = json.loads(existing_algorithm.instance_keys)
                        if instance_key not in instance_keys:
                            raise ValueError(f'[DocumentProcessor] Algorithm {name} registered '
                                             f'with different instance keys)')
                        else:
                            instance_keys.remove(instance_key)
                            existing_algorithm.instance_keys = json.dumps(instance_keys)
                            existing_algorithm.updated_at = datetime.now()
                            if len(instance_keys) == 0:
                                existing_algorithm.is_active = False
                                self._processors.pop(name)
                                LOG.info(f'[DocumentProcessor] Algorithm {name} dropped! No instance keys left')
                    else:
                        LOG.warning(f'[DocumentProcessor] Algorithm {name} not found')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to drop algorithm: {e}, {traceback.format_exc()}')
                raise e

        def _orm_to_dict(self, obj):
            return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

        @app.get('/health')
        def get_health(self) -> None:
            self._ensure_initialized()
            if self._post_func_thread is None or not self._post_func_thread.is_alive():
                return BaseResponse(code=503, msg='Post function thread not alive')
            if self._refresh_algo_thread is None or not self._refresh_algo_thread.is_alive():
                return BaseResponse(code=503, msg='Refresh algorithm thread not alive')

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
            self._refresh_algorithms_impl()
            res = []
            for algo_id, processor in self._processors.items():
                res.append({'algo_id': algo_id, 'display_name': processor._display_name,
                            'description': processor._description})
            if not res:
                LOG.warning('[DocumentProcessor] No algorithm registered')
            return BaseResponse(code=200, msg='success', data=res)

        @app.delete('/algo/{algo_id}')
        def drop_algo(self, algo_id: str) -> None:
            self.drop_algorithm(algo_id)
            return BaseResponse(code=200, msg='success')

        @app.get('/algo/{algo_id}/info')
        def get_algo_info(self, algo_id: str) -> None:
            self._refresh_algorithms_impl()
            processor = self._processors.get(algo_id)
            if processor is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            # Serialize info_pickle to base64 for JSON transmission
            info_pickle_bytes = cloudpickle.dumps({
                'store': processor._store,
                'reader': processor._reader,
                'node_groups': processor._node_groups
            })
            info_pickle_b64 = base64.b64encode(info_pickle_bytes).decode('utf-8')
            data = {
                'algo_id': algo_id,
                'display_name': processor._display_name,
                'description': processor._description,
                'version': processor._version,
                'info_pickle': info_pickle_b64,
            }
            return BaseResponse(code=200, msg='success', data=data)

        @app.get('/algo/{algo_id}/group/info')
        def get_algo_group_info(self, algo_id: str) -> None:
            self._refresh_algorithms_impl()
            processor = self._processors.get(algo_id)
            if processor is None:
                raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            data = []
            for group_name in processor._store.activated_groups():
                if group_name in processor._node_groups:
                    group_info = {'name': group_name, 'type': processor._node_groups[group_name].get('group_type'),
                                  'display_name': processor._node_groups[group_name].get('display_name')}
                    data.append(group_info)
            LOG.info(f'[DocumentProcessor] Get group info for {algo_id} success with {data}')
            return BaseResponse(code=200, msg='success', data=data)

        @app.post('/doc/add')
        def add_doc(self, request: AddDocRequest):
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received add doc request: {payload}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            if algo_id not in self._processors:
                self._refresh_algorithms_impl()
                if algo_id not in self._processors:
                    raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')
            # NOTE: No idempotency key check, should be handled by the caller!
            new_file_ids = []
            reparse_file_ids = []
            for file_info in file_infos:
                if self._path_prefix:
                    file_info.file_path = create_file_path(path=file_info.file_path, prefix=self._path_prefix)
                if file_info.reparse_group is not None:
                    reparse_file_ids.append(file_info.doc_id)
                else:
                    new_file_ids.append(file_info.doc_id)
            if new_file_ids and reparse_file_ids:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail='new_file_ids and reparse_file_ids cannot be specified at the same time'
                )
            if new_file_ids:
                task_type = TaskType.DOC_ADD.value
            elif reparse_file_ids:
                task_type = TaskType.DOC_REPARSE.value
            else:
                raise fastapi.HTTPException(status_code=400, detail='no input files or reparse group specified')
            payload_json = json.dumps(payload, ensure_ascii=False)

            try:
                user_priority = request.priority if request.priority is not None else 0
                task_score = calculate_task_score(task_type, user_priority)
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    created_at=datetime.now(),
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
            payload = request.model_dump()
            LOG.info(f'update doc meta for {payload}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos

            if not file_infos:
                raise fastapi.HTTPException(status_code=400, detail='file_infos is required')
            if algo_id not in self._processors:
                self._refresh_algorithms_impl()
                if algo_id not in self._processors:
                    raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')

            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                task_type = TaskType.DOC_UPDATE_META.value
                user_priority = request.priority if request.priority is not None else 0
                task_score = calculate_task_score(task_type, user_priority)
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    created_at=datetime.now(),
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
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received delete doc request: {payload}')

            task_id = request.task_id
            algo_id = request.algo_id
            doc_ids = request.doc_ids
            if not doc_ids:
                raise fastapi.HTTPException(status_code=400, detail='doc_ids is required')
            if algo_id not in self._processors:
                self._refresh_algorithms_impl()
                if algo_id not in self._processors:
                    raise fastapi.HTTPException(status_code=404, detail=f'Invalid algo_id {algo_id}')

            payload_json = json.dumps(payload, ensure_ascii=False)
            try:
                task_type = TaskType.DOC_DELETE.value
                user_priority = request.priority if request.priority is not None else 0
                task_score = calculate_task_score(task_type, user_priority)
                self._waiting_task_queue.enqueue(
                    task_id=task_id,
                    task_type=task_type,
                    user_priority=user_priority,
                    task_score=task_score,
                    message=payload_json,
                    created_at=datetime.now(),
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
        def cancel(self, request: CancelDocRequest):
            payload = request.model_dump()
            LOG.info(f'[DocumentProcessor] Received cancel task request: {payload}')
            task_id = request.task_id
            try:
                # NOTE: only the task in waiting state can be canceled
                cancel_status = False
                task_status = TaskStatus.WAITING.value
                waiting_task = self._waiting_task_queue.peek(filter_by={'task_id': task_id})
                message = ''
                if waiting_task:
                    self._waiting_task_queue.dequeue(filter_by={'task_id': task_id})
                    cancel_status = True
                    task_status = TaskStatus.CANCELED.value
                    message = 'Canceled by user'
                else:
                    finished_task = self._finished_task_queue.peek(filter_by={'task_id': task_id})
                    if finished_task:
                        cancel_status = False
                        task_status = finished_task['task_status']
                        if task_status == TaskStatus.FAILED.value:
                            message = (f'Finished with error: code={finished_task["error_code"]},'
                                       f' msg={finished_task["error_msg"]}')
                        else:
                            message = 'Task already finished'
                        self._finished_task_queue.dequeue(filter_by={'task_id': task_id})
                    else:
                        task_status = TaskStatus.WORKING.value
                        message = (f'Task {task_id} not found in waiting or finished queue,'
                                   ' task may be running and cannot be canceled!')
                return BaseResponse(
                    code=200,
                    msg='success' if cancel_status else 'failed',
                    data={
                        'task_id': task_id,
                        'cancel_status': cancel_status,
                        'task_status': task_status,
                        'message': message,
                    }
                )
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Failed to cancel task {task_id}: {e}, {traceback.format_exc()}')
                raise fastapi.HTTPException(status_code=500, detail=f'Failed to cancel task: {str(e)}')

        def _assert_post_func(self) -> bool:
            '''assert post function is callable and params include task_id, task_status, error_code, error_msg'''
            if not self._post_func:
                return True
            if not callable(self._post_func):
                LOG.error('[DocumentProcessor] Post function is not callable')
                return False
            if not all(
                param in self._post_func.__code__.co_varnames for param in [
                    'task_id', 'task_status', 'error_code', 'error_msg'
                ]
            ):
                LOG.error('[DocumentProcessor] Post function params do not include'
                          ' task_id, task_status, error_code, error_msg')
                return False
            return True

        def _callback(self, task_id: str, task_status: str = None, error_code: str = None, error_msg: str = None):
            '''callback to service'''
            if self._post_func:
                try:
                    self._post_func(task_id, task_status, error_code, error_msg)
                except Exception as e:
                    LOG.error(f'[DocumentProcessor] Failed to call post function: {e}, {traceback.format_exc()}')
                    raise e
            else:
                LOG.warning(f'[DocumentProcessor] No post function configured, task {task_id} status: {task_status},'
                            f'error_code: {error_code}, error_msg: {error_msg}')

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(self, server: bool = True, port: int = None, url: str = None, num_workers: int = 1,
                 db_config: Optional[Dict[str, Any]] = _get_default_db_config(),
                 launcher: Optional[Launcher] = None, post_func: Optional[Callable] = None,
                 path_prefix: Optional[str] = None):
        super().__init__()
        self._server_mode = server
        self._raw_impl = None  # 保存原始 Impl 对象的引用
        self._db_config = db_config  # 保存 db_config，即使是通过 URL 初始化也需要用于直接操作数据库
        if not url:
            # 创建 Impl 对象（此时使用懒加载，不会创建线程）
            self._raw_impl = DocumentProcessor.Impl(server=server, num_workers=num_workers, db_config=db_config,
                                                    post_func=post_func, path_prefix=path_prefix)
            if server:
                # 用 ServerModule 包装
                self._impl = ServerModule(self._raw_impl, port=port, launcher=launcher)
            else:
                self._impl = self._raw_impl
        else:
            self._impl = UrlModule(url=ensure_call_endpoint(url))

    def start(self):
        # start the server
        result = super().start()

        # ensure the initialization
        if self._server_mode and self._raw_impl:
            LOG.info('[DocumentProcessor] Server started, triggering post-start initialization...')
            try:
                self._dispatch('_ensure_initialized', server_url=self._impl._url)
                LOG.info('[DocumentProcessor] Post-start initialization triggered successfully')
            except Exception as e:
                LOG.error(f'[DocumentProcessor] Post-start initialization failed: {e}, {traceback.format_exc()}')
                raise
        return result

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
                           node_groups: Dict[str, Dict], display_name: Optional[str] = None,
                           description: Optional[str] = None, version: Optional[str] = '1.0.0',
                           instance_key: Optional[str] = None, **kwargs):
        assert isinstance(reader, DirectoryReader), 'Only DirectoryReader can be registered to processor'
        self._dispatch('register_algorithm', name, store, reader, node_groups,
                       display_name, description, version, instance_key, **kwargs)

    def drop_algorithm(self, name: str, version: Optional[str] = '1.0.0', instance_key: Optional[str] = None) -> None:
        return self._dispatch('drop_algorithm', name, version, instance_key)

    def get_algo_info(self, algo_id: str) -> BaseResponse:
        return self._dispatch('get_algo_info', algo_id)
