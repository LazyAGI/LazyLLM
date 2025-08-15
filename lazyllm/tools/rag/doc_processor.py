from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, JSON, String, TIMESTAMP, Table, MetaData, inspect, delete, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine
from lazyllm import LOG, ModuleBase, ServerModule, UrlModule, FastapiApp as app, ThreadPoolExecutor, config

from .store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .store.store_base import DEFAULT_KB_ID
from .store.document_store import _DocumentStore
from .store.utils import fibonacci_backoff, create_file_path
from .transform import (AdaptiveTransform, make_transform,)
from .readers import ReaderBase
from .doc_node import DocNode
from .utils import gen_docid, ensure_call_endpoint, BaseResponse
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID

import queue
import threading
import time
import requests
import uuid
import os
import traceback

DB_TYPES = ['mysql']
ENABLE_DB = os.getenv("RAG_ENABLE_DB", "false").lower() == "true"


class _Processor:
    def __init__(self, store: _DocumentStore, reader: ReaderBase, node_groups: Dict[str, Dict],
                 display_name: Optional[str] = None, description: Optional[str] = None,
                 server: bool = False):
        self._store = store
        self._reader = reader
        self._node_groups = node_groups
        self._display_name = display_name
        self._description = description

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,
                metadatas: Optional[List[Dict[str, Any]]] = None):
        try:
            if not input_files: return
            if not ids: ids = [gen_docid(path) for path in input_files]
            if metadatas is None:
                metadatas = [{} for _ in input_files]
            for metadata, doc_id, path in zip(metadatas, ids, input_files):
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, path)
                metadata.setdefault(RAG_KB_ID, DEFAULT_KB_ID)
            root_nodes, image_nodes = self._reader.load_data(input_files, metadatas, split_image_nodes=True)
            self._store.update_nodes(self._set_nodes_number(root_nodes))
            self._create_nodes_recursive(root_nodes, LAZY_ROOT_NAME)
            if image_nodes:
                self._store.update_nodes(self._set_nodes_number(image_nodes))
                self._create_nodes_recursive(image_nodes, LAZY_IMAGE_GROUP)
            LOG.info("Add documents done!")
        except Exception as e:
            LOG.error(f"Add documents failed: {e}, {traceback.format_exc()}")
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

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str):
        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f"Node group '{group_name}' does not exist. Please check the group name "
                                 "or add a new one through `create_node_group`.")

            if group['parent'] == p_name:
                nodes = self._create_nodes_impl(p_nodes, group_name)
                if nodes: self._create_nodes_recursive(nodes, group_name)

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

    def reparse(self, group_name: str, uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None, **kwargs):
        if doc_ids:
            self._reparse_docs(group_name=group_name, doc_ids=doc_ids, **kwargs)
        else:
            self._get_or_create_nodes(group_name, uids)

    def _reparse_docs(self, group_name: str, doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict]):
        kb_id = metadatas[0].get(RAG_KB_ID, None)
        if group_name == "all":
            self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
            removed_flag = False
            for wait_time in fibonacci_backoff():
                nodes = self._store.get_nodes(group=LAZY_ROOT_NAME, kb_id=kb_id, doc_ids=doc_ids)
                if not nodes:
                    removed_flag = True
                    break
                time.sleep(wait_time)
            if not removed_flag:
                raise Exception(f"Failed to remove nodes for docs {doc_ids} from store")
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas)
        else:
            p_nodes = self._store.get_nodes(group=self._node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name, doc_ids=doc_ids)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str, doc_ids: List[str]):
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, None)
        self._store.remove_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)

        removed_flag = False
        for wait_time in fibonacci_backoff():
            nodes = self._store.get_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)
            if not nodes:
                removed_flag = True
                break
            time.sleep(wait_time)
        if not removed_flag:
            raise Exception(f"Failed to remove nodes for docs {doc_ids} group {cur_name} from store")

        t = self._node_groups[cur_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, cur_name)
        nodes = transform.batch_forward(p_nodes, cur_name)
        # reparse need set global_metadata
        self._store.update_nodes(self._set_nodes_number(nodes))

        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f"Node group '{group_name}' does not exist. Please check the group name "
                                 "or add a new one through `create_node_group`.")
            if group['parent'] == cur_name:
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name, doc_ids=doc_ids)

    def update_doc_meta(self, doc_id: str, metadata: dict):
        self._store.update_doc_meta(doc_id=doc_id, metadata=metadata)

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None) -> None:
        LOG.info(f"delete_doc_ids: {doc_ids}")
        self._store.remove_nodes(kb_id=kb_id, doc_ids=doc_ids)


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
    task_id: str
    algo_id: Optional[str] = "__default__"
    file_infos: List[FileInfo]
    db_info: DBInfo
    feedback_url: Optional[str] = None


class UpdateMetaRequest(BaseModel):
    algo_id: Optional[str] = "__default__"
    file_infos: List[FileInfo]
    db_info: DBInfo


class DeleteDocRequest(BaseModel):
    algo_id: Optional[str] = "__default__"
    dataset_id: str
    doc_ids: List[str]
    db_info: Optional[DBInfo] = None


class CancelDocRequest(BaseModel):
    task_id: str


class DocumentProcessor(ModuleBase):

    class Impl():
        def __init__(self, server: bool):
            self._processors: Dict[str, _Processor] = dict()
            self._server = server
            self._inited = False
            try:
                self._feedback_url = config['process_feedback_service']
                self._path_prefix = config['process_path_prefix']
            except Exception as e:
                LOG.warning(f"Failed to get config: {e}, use env variables instead")
                self._feedback_url = os.getenv("PROCESS_FEEDBACK_SERVICE", None)
                self._path_prefix = os.getenv("PROCESS_PATH_PREFIX", None)

        def _init_components(self, server: bool):
            if server and not self._inited:
                self._task_queue = queue.Queue()
                self._tasks = {}    # running tasks
                self._pending_task_ids = set()  # pending tasks
                self._add_executor = ThreadPoolExecutor(max_workers=4)
                self._add_futures = {}
                self._delete_executor = ThreadPoolExecutor(max_workers=4)
                self._update_executor = ThreadPoolExecutor(max_workers=4)
                self._update_futures = {}

                self._engines: dict[str, Engine] = {}
                self._inspectors: dict[str, inspect] = {}

                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
            self._inited = True
            LOG.info(f"[DocumentProcessor] init done. feedback {self._feedback_url}, prefix {self._path_prefix}")

        def register_algorithm(self, name: str, store: _DocumentStore, reader: ReaderBase,
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

        def _get_engine(self, url) -> Engine:
            if url not in self._engines:
                engine = create_engine(url, echo=False, pool_pre_ping=True)
                self._engines[url] = engine
                self._inspectors[url] = inspect(engine)
            return self._engines[url]

        def _get_inspector(self, url):
            self._get_engine(url=url)
            return self._inspectors[url]

        def _get_url_from_db_info(self, db_info: DBInfo):
            return (f"mysql+pymysql://{db_info.user}:{db_info.password}"
                    f"@{db_info.host}:{db_info.port}/{db_info.db_name}"
                    "?charset=utf8mb4")

        def create_table(self, db_info: DBInfo):
            if db_info.db_type == "mysql":
                try:
                    url = self._get_url_from_db_info(db_info)
                    engine = self._get_engine(url=url)
                    inspector = self._get_inspector(url=url)
                    tbl = db_info.table_name
                    schema = db_info.db_name

                    if not inspector.has_table(tbl, schema=schema):
                        metadata = MetaData()
                        table = Table(tbl, metadata, Column('document_id', String(255), primary_key=True),
                                      Column('file_name', String(255), nullable=False),
                                      Column('file_path', String(255), nullable=False),
                                      Column('description', String(255), nullable=True),
                                      Column('creater', String(255), nullable=False),
                                      Column('dataset_id', String(255), nullable=False),
                                      Column('tags', JSON, nullable=True),
                                      Column('created_at', TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")))
                        metadata.create_all(engine, tables=[table])
                        LOG.info(f"Created table `{tbl}` in `{schema}`")
                except Exception as e:
                    LOG.error(f"Failed to create table `{tbl}` in `{schema}`: {e}")
                    return
            else:
                raise ValueError(f"Unsupported database type: {db_info.db_type}")

        def operate_db(self, db_info: DBInfo, operation: str,
                       file_infos: List[FileInfo] = None, params: Dict = None) -> None:
            db_type = db_info.db_type
            if db_type not in DB_TYPES:
                raise ValueError(f"Unsupported db_type: {db_type}")
            url = self._get_url_from_db_info(db_info)
            engine = self._get_engine(url=url)
            if operation == 'upsert':
                self._upsert_records(engine, db_info, file_infos)
            elif operation == 'delete':
                self._delete_records(engine, db_info, params)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        def _upsert_records(self, engine, db_info, file_infos):
            table_name = db_info['table_name']
            metadata = MetaData()
            metadata.reflect(bind=engine, only=[table_name])
            table = metadata.tables[table_name]
            with engine.begin() as conn:
                for file_info in file_infos:
                    document_id = file_info.get("doc_id")
                    file_path = file_info.get("file_path")
                    if not document_id or not file_path:
                        raise ValueError(f"Invalid file_info: {file_info}")

                    raw_infos = {"document_id": document_id, "file_name": os.path.basename(file_path),
                                 "file_path": file_path, "description": file_info["metadata"].get("description", None),
                                 "creater": file_info["metadata"].get("creater", None),
                                 "dataset_id": file_info["metadata"].get(RAG_KB_ID, None),
                                 "tags": file_info["metadata"].get("tags", []) or []}
                    infos = {}
                    for k, v in raw_infos.items():
                        if v is None:
                            continue
                        if isinstance(v, str) and not v.strip():
                            continue
                        if isinstance(v, (list, dict)) and not v:
                            continue
                        infos[k] = v
                    if "document_id" not in infos:
                        infos["document_id"] = document_id

                    stmt = mysql_insert(table).values(**infos)
                    update_dict = {k: stmt.inserted[k] for k in infos if k != 'document_id'}
                    upsert_stmt = stmt.on_duplicate_key_update(**update_dict)
                    conn.execute(upsert_stmt)

        def _delete_records(self, engine, db_info, params):
            table_name = db_info['table_name']
            metadata = MetaData()
            metadata.reflect(bind=engine, only=[table_name])
            table = metadata.tables[table_name]

            with engine.begin() as conn:  # 自动提交或回滚事务
                doc_ids = params.get("doc_ids", [])
                for document_id in doc_ids:
                    stmt = delete(table).where(table.c.document_id == document_id)
                    conn.execute(stmt)

        @app.get('/algo/list')
        async def get_algo_list(self) -> None:
            res = []
            for algo_id, processor in self._processors.items():
                res.append({"algo_id": algo_id, "display_name": processor._display_name,
                            "description": processor._description})
            return BaseResponse(code=200, msg='success', data=res)

        @app.get('/group/info')
        async def get_group_info(self, algo_id: str) -> None:
            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f"Invalid algo_id {algo_id}")
            processor = self._processors[algo_id]
            infos = []
            for group_name in processor._store.activated_groups():
                if group_name in processor._node_groups:
                    group_info = {"name": group_name, "type": processor._node_groups[group_name].get('group_type'),
                                  "display_name": processor._node_groups[group_name].get('display_name')}
                    infos.append(group_info)
            LOG.info(f"Get group info for {algo_id} success with {infos}")
            return BaseResponse(code=200, msg='success', data=infos)

        @app.post('/doc/add')
        async def async_add_doc(self, request: AddDocRequest):
            LOG.info(f"Add doc for {request.model_dump_json()}")
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            db_info = request.db_info
            feedback_url = request.feedback_url
            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f"Invalid algo_id {algo_id}")
            if task_id in self._pending_task_ids or task_id in self._tasks:
                return BaseResponse(code=400, msg=f'The task {task_id} already exists in queue', data=None)
            if self._path_prefix:
                for file_info in file_infos:
                    file_info.file_path = create_file_path(path=file_info.file_path, prefix=self._path_prefix)

            params = {"file_infos": file_infos, "db_info": db_info, "feedback_url": feedback_url}
            if ENABLE_DB:
                self.create_table(db_info=db_info)

            self._task_queue.put(('add', algo_id, task_id, params))
            self._pending_task_ids.add(task_id)
            return BaseResponse(code=200, msg='task submit successfully', data={"task_id": task_id})

        @app.post('/doc/meta/update')
        async def async_update_meta(self, request: UpdateMetaRequest):
            LOG.info(f"update doc meta for {request.model_dump_json()}")
            algo_id = request.algo_id
            file_infos = request.file_infos
            db_info = request.db_info

            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f"Invalid algo_id {algo_id}")

            for file_info in file_infos:
                doc_id = file_info.doc_id
                metadata = file_info.metadata
                old_fut = self._update_futures.get(doc_id)
                if old_fut and not old_fut.done():
                    cancelled = old_fut.cancel()
                    LOG.info(f"Canceled previous update for {doc_id}: {cancelled}")

                new_fut = self._update_executor.submit(self._processors[algo_id].update_doc_meta, doc_id=doc_id,
                                                       metadata=metadata)

                self._update_futures[doc_id] = new_fut

                def _cleanup(fut, doc_id=doc_id):
                    if self._update_futures.get(doc_id) is fut:
                        del self._update_futures[doc_id]
                new_fut.add_done_callback(_cleanup)
                if ENABLE_DB:
                    new_fut.add_done_callback(
                        lambda fut, dbi=db_info, fi=file_info: self.operate_db(dbi, 'upsert', file_infos=[fi]))

            return BaseResponse(code=200, msg='success')

        @app.delete('/doc/delete')
        async def async_delete_doc(self, request: DeleteDocRequest) -> None:
            LOG.info(f"Del doc for {request.model_dump_json()}")
            algo_id = request.algo_id
            dataset_id = request.dataset_id
            doc_ids = request.doc_ids
            db_info = request.db_info

            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f"Invalid algo_id {algo_id}")

            task_id = str(uuid.uuid4())
            self._task_queue.put(('delete', algo_id, task_id,
                                  {"dataset_id": dataset_id, "doc_ids": doc_ids, "db_info": db_info}))
            self._pending_task_ids.add(task_id)
            return BaseResponse(code=200, msg='task submit successfully', data={"task_id": task_id})

        @app.post('/doc/cancel')
        async def cancel_task(self, request: CancelDocRequest):
            task_id = request.task_id
            if task_id in self._pending_task_ids:
                self._pending_task_ids.remove(task_id)
                status = 1
            elif task_id in self._tasks:
                future = self._tasks.get(task_id)
                if future and not future.done():
                    cancelled = future.cancel()
                    status = 1 if cancelled else 0
                    if cancelled:
                        self._tasks.pop(task_id, None)
                else:
                    status = 0
            return BaseResponse(code=200, msg="success" if status else "failed",
                                data={"task_id": task_id, "status": status})

        def _send_status_message(self, task_id: str, callback_path: str, success: bool,
                                 error_code: str = "", error_msg: str = ""):
            if self._feedback_url:
                try:
                    full_url = self._feedback_url + callback_path
                    payload = {"task_id": task_id, "status": 1 if success else 0, "error_code": error_code,
                               "error_msg": error_msg}
                    headers = {"Content-Type": "application/json"}
                    res = None
                    for wait_time in fibonacci_backoff(max_retries=3):
                        try:
                            res = requests.post(full_url, json=payload, headers=headers, timeout=5)
                            if res.status_code == 200:
                                break
                            LOG.warning(
                                f"Task-{task_id}: Unexpected status {res.status_code}, retrying in {wait_time}s…")
                        except Exception as e:
                            LOG.error(f"Task-{task_id}: Request failed: {e}, retrying in {wait_time}s…")
                        time.sleep(wait_time)

                    if res is None:
                        raise RuntimeError("Failed to send feedback—no response received after retries")
                    res.raise_for_status()
                except Exception as e:
                    LOG.error(f"Task-{task_id}: Failed to send feedback to {full_url}: {e}")
            else:
                LOG.error("process_feedback_service is not set")

        def _exec_add_task(self, algo_id, task_id, params):
            try:
                file_infos: List[FileInfo] = params.get('file_infos')
                callback_path = params.get('feedback_url')
                db_info: DBInfo = params.get('db_info')

                input_files = []
                ids = []
                metadatas = []

                reparse_group = None
                reparse_doc_ids = []
                reparse_files = []
                reparse_metadatas = []

                for file_info in file_infos:
                    if file_info.reparse_group:
                        reparse_group = file_info.reparse_group
                        reparse_doc_ids.append(file_info.doc_id)
                        reparse_files.append(file_info.file_path)
                        reparse_metadatas.append(file_info.metadata)
                    else:
                        input_files.append(file_info.file_path)
                        ids.append(file_info.doc_id)
                        metadatas.append(file_info.metadata)

                if input_files:
                    future = self._add_executor.submit(self._processors[algo_id].add_doc, input_files=input_files,
                                                       ids=ids, metadatas=metadatas)
                    if ENABLE_DB:
                        future.add_done_callback(lambda fut: self.operate_db(db_info, 'upsert', file_infos=file_infos))
                elif reparse_group:
                    future = self._add_executor.submit(self._processors[algo_id].reparse, group_name=reparse_group,
                                                       doc_ids=reparse_doc_ids, doc_paths=reparse_files,
                                                       metadatas=reparse_metadatas)
                else:
                    LOG.error(
                        f"Task-{task_id}: add task error, no input files {input_files} or reparse group {reparse_group}"
                    )
                self._tasks[task_id] = (future, callback_path)
                self._pending_task_ids.remove(task_id)
            except Exception as e:
                LOG.error(f"Task-{task_id}: add task error {e}")

        def _exec_delete_task(self, algo_id, task_id, params):
            dataset_id = params.get("dataset_id")
            doc_ids = params.get("doc_ids")
            future = self._delete_executor.submit(
                self._processors[algo_id].delete_doc, dataset_id=dataset_id, doc_ids=doc_ids
            )
            if ENABLE_DB and params.get("db_info") is not None:
                db_info = params.get("db_info")
                future.add_done_callback(lambda fut: self.operate_db(db_info, 'delete', params=params))
            self._tasks[task_id] = (future, None)
            self._pending_task_ids.remove(task_id)

        def _worker(self):  # noqa: C901
            while True:
                try:
                    task_type, algo_id, task_id, params = self._task_queue.get(timeout=1)
                    if task_id not in self._pending_task_ids:
                        continue
                    if task_type == 'add':
                        self._exec_add_task(algo_id=algo_id, task_id=task_id, params=params)
                    elif task_type == 'delete':
                        self._exec_delete_task(algo_id=algo_id, task_id=task_id, params=params)
                    time.sleep(0.2)
                except queue.Empty:
                    task_need_pop = []
                    for task_id, (future, callback_path) in self._tasks.items():
                        if future.done():
                            task_need_pop.append(task_id)
                            ex = future.exception()
                            if callback_path and not ex:
                                self._send_status_message(task_id=task_id, callback_path=callback_path, success=True,
                                                          error_code="", error_msg="")
                            elif callback_path and ex:
                                self._send_status_message(task_id=task_id, callback_path=callback_path, success=False,
                                                          error_code=type(ex).__name__, error_msg=str(ex))
                                LOG.error(f"task {task_id} failed: {str(ex)}")
                            elif ex:
                                LOG.error(f"task {task_id} failed: {str(ex)}")
                    for task_id in task_need_pop:
                        self._tasks.pop(task_id)
                        LOG.info(f"task {task_id} done")
                    time.sleep(5)

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(self, server: bool = True, port: int = None, url: str = None):
        super().__init__()
        if not url:
            self._impl = DocumentProcessor.Impl(server=server)
            if server:
                self._impl = ServerModule(self._impl, port=port)
        else:
            self._impl = UrlModule(url=ensure_call_endpoint(url))

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._impl
        if isinstance(impl, ServerModule):
            impl._call(method, *args, **kwargs)
        else:
            getattr(impl, method)(*args, **kwargs)

    def register_algorithm(self, name: str, store: _DocumentStore, reader: ReaderBase, node_groups: Dict[str, Dict],
                           display_name: Optional[str] = None, description: Optional[str] = None,
                           force_refresh: bool = False, **kwargs):
        self._dispatch("register_algorithm", name, store, reader, node_groups,
                       display_name, description, force_refresh, **kwargs)

    def drop_algorithm(self, name: str, clean_db: bool = False) -> None:
        return self._dispatch("drop_algorithm", name, clean_db)
