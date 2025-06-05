from urllib.parse import urljoin
from collections import defaultdict
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from lazyllm import LOG, ServerModule, FastapiApp as app, ThreadPoolExecutor, config
from .transform import (AdaptiveTransform, make_transform,)
from .store_base import StoreBase, LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .readers import ReaderBase
from .doc_node import DocNode
from .utils import gen_docid, BaseResponse
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH
import queue
import threading
import time
import requests
import uuid

class _Processor:
    def __init__(self, store: StoreBase, reader: ReaderBase, node_groups: Dict[str, Dict], server: bool = False):
        self._store, self._reader, self._node_groups = store, reader, node_groups

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,
                metadatas: Optional[List[Dict[str, Any]]] = None):
        LOG.info("Adding documents")
        if not input_files: return
        if not ids: ids = [gen_docid(path) for path in input_files]
        if not metadatas:
            metadatas = [{RAG_DOC_ID: id, RAG_DOC_PATH: path} for id, path in zip(ids, input_files)]
        else:
            for path, id, metadata in zip(input_files, ids, metadatas):
                metadata.update({RAG_DOC_ID: id, RAG_DOC_PATH: path})
        root_nodes, image_nodes = self._reader.load_data(input_files, metadatas, split_image_nodes=True)
        self._store.update_nodes(root_nodes)
        self._create_nodes_recursive(root_nodes, LAZY_ROOT_NAME)
        if image_nodes:
            self._store.update_nodes(image_nodes)
            self._create_nodes_recursive(image_nodes, LAZY_IMAGE_GROUP)

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
        # NOTE bug: transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        t = self._node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t)
        nodes = transform.batch_forward(p_nodes, group_name)
        self._store.update_nodes(nodes)
        return nodes

    def _get_or_create_nodes(self, group_name, ids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(group_name, ids) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(self._node_groups[group_name]['parent'], ids)
            nodes = self._create_nodes_impl(p_nodes, group_name)
        return nodes

    def reparse(self, group_name: str, ids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None):
        if doc_ids:
            nodes = self._store.get_nodes(group_name=group_name, doc_ids=doc_ids)
            if nodes:
                if nodes[0].children:
                    raise ValueError(f"Cannot reparse group '{group_name}': "
                                     "the group has children. ")
                self._store.remove_nodes(group_name=group_name, uids=[node._uid for node in nodes])
                p_nodes = self._store.get_nodes(group_name=self._node_groups[group_name]['parent'])
                nodes = self._create_nodes_impl(p_nodes, group_name)
                self._store.update_nodes(p_nodes + nodes)
        else:
            self._get_or_create_nodes(group_name, ids)

    def delete_doc(self, input_files: List[str] = None, doc_ids: List[str] = None) -> None:
        if input_files:
            LOG.info(f"delete_files: {input_files}")
            root_nodes = self._store.get_index(type='file_node_map').query(input_files)
            LOG.info(f"delete_files: removing documents {input_files} and nodes {root_nodes}")
            if len(root_nodes) == 0: return

            uids_to_delete = defaultdict(list)
            uids_to_delete[LAZY_ROOT_NAME] = [node._uid for node in root_nodes]

            # Gather all nodes to be deleted including their children
            def gather_children(node: DocNode):
                for children_group, children_list in node.children.items():
                    for child in children_list:
                        uids_to_delete[children_group].append(child._uid)
                        gather_children(child)

            for node in root_nodes:
                gather_children(node)

            # Delete nodes in all groups
            for group, node_uids in uids_to_delete.items():
                self._store.remove_nodes(group, node_uids)
                LOG.debug(f"Removed nodes from group {group} for node IDs: {node_uids}")
        elif doc_ids:
            LOG.info(f"delete_doc_ids: {doc_ids}")
            self._store.remove_nodes(doc_ids=doc_ids)
        else:
            raise ValueError("Please specify either input_files or doc_ids.")


class FileInfo(BaseModel):
    file_path: str
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
    algo_id: str
    file_infos: List[FileInfo]
    db_info: DBInfo
    feedback_url: Optional[str] = None


class DeleteDocRequest(BaseModel):
    algo_id: str
    doc_ids: List[str]


class CancelDocRequest(BaseModel):
    task_id: str


class DocumentProcessor():

    class Impl():
        def __init__(self, server: bool):
            self._processors: Dict[str, _Processor] = dict()
            self._server = server
            self._inited = False

        def _init_components(self, server: bool):
            if server and not self._inited:
                self._task_queue = queue.Queue()
                self._tasks = {}    # running tasks
                self._pending_task_ids = set()  # pending tasks
                self._add_executor = ThreadPoolExecutor(max_workers=1)
                self._delete_executor = ThreadPoolExecutor(max_workers=1)
                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
            self._inited = True

        def register_algorithm(self, name: str, store: StoreBase, reader: ReaderBase,
                               node_groups: Dict[str, Dict], force_refresh: bool = False):
            if name in self._processors and not force_refresh:
                raise KeyError(f'Duplicated algo key {name} for processor!')
            self._processors[name] = _Processor(store, reader, node_groups)

        @app.get('/group/info')
        async def get_group_info(self, algo_id: str) -> None:
            processor = self._processors[algo_id]
            infos = []
            for group_name in processor._store.activated_groups():
                if group_name in processor._node_groups:
                    if processor._node_groups[group_name].get('info', {}).get('type'):
                        type = processor._node_groups[group_name].get('info', {}).get('type').value
                        display_name = processor._node_groups[group_name].get('info', {}).get('name')
                    else:
                        type = "unknown"
                        display_name = group_name
                    group_info = {
                        "name": group_name,
                        "type": type,
                        "display_name": display_name,
                    }
                    infos.append(group_info)
            return BaseResponse(code=200, msg='success', data=infos)

        @app.post('/doc/add')
        async def async_add_doc(self, request: AddDocRequest):

            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            db_info = request.db_info
            feedback_url = request.feedback_url

            if task_id in self._pending_task_ids or task_id in self._tasks:
                return BaseResponse(code=400, msg=f'The task {task_id} already exists in queue', data=None)

            params = {
                "file_infos": file_infos,
                "db_info": db_info,
                "feedback_url": feedback_url
            }

            self._task_queue.put(('add', algo_id, task_id, params))
            self._pending_task_ids.add(task_id)
            return BaseResponse(code=200, msg='task submit successfully', data={"task_id": task_id})

        @app.delete('/doc/delete')
        async def async_delete_doc(self, request: DeleteDocRequest) -> None:

            algo_id = request.algo_id
            doc_ids = request.doc_ids

            task_id = str(uuid.uuid4())
            self._task_queue.put(('delete', algo_id, task_id, doc_ids))
            self._pending_task_ids.add(task_id)
            return BaseResponse(code=200, msg='task submit successfully', data={"task_id": task_id})

        @app.post('/doc/cancel')
        async def cancel_task(self, request: CancelDocRequest):
            task_id = request.task_id
            if task_id in self._pending_task_ids:
                self._pending_task_ids.remove(task_id)
                status = 1
            elif task_id in self._tasks:
                future = self._tasks.pop(task_id, None)
                if future and not future.done():
                    future.cancel()
                    status = 1
                else:
                    status = 0
            else:
                status = 0
            return BaseResponse(
                code=200,
                msg="success" if status else "failed",
                data={"task_id": task_id, "status": status}
            )

        def _send_status_message(self, task_id: str, url: str, success: bool,
                                 error_code: str = "", error_msg: str = ""):
            if config['process_feedback_service']:
                payload = {
                    "task_id": task_id,
                    "status": 1 if success else 0,
                    "error_code": error_code,
                    "error_msg": error_msg,
                }
                headers = {"Content-Type": "application/json"}
                url = urljoin(config['process_feedback_service'], url)
                requests.post(url, json=payload, headers=headers, timeout=5)
            else:
                raise ValueError("process_feedback_service is not set")

        def _worker(self):  # noqa: C901
            while True:
                try:
                    task_type, algo_id, task_id, params = self._task_queue.get(timeout=1)
                    if task_id not in self._pending_task_ids:
                        continue
                    if task_type == 'add':
                        file_infos: List[FileInfo] = params.get('file_infos')
                        url = params.get('feedback_url')
                        input_files = []
                        ids = []
                        metadatas = []
                        reparse_group = None
                        reparse_docs = []

                        for file_info in file_infos:
                            if file_info.reparse_group:
                                reparse_group = reparse_group
                                reparse_docs.append(file_info.doc_id)
                            else:
                                input_files.append(file_info.file_path)
                                ids.append(file_info.doc_id)
                                metadatas.append(file_info.metadata)
                        if input_files:
                            future = self._add_executor.submit(
                                self._processors[algo_id].add_doc,
                                input_files=input_files,
                                ids=ids,
                                metadatas=metadatas
                            )
                        elif reparse_group:
                            future = self._add_executor.submit(
                                self._processors[algo_id].reparse,
                                group_name=reparse_group,
                                doc_ids=reparse_docs
                            )
                        self._pending_task_ids.remove(task_id)
                        self._tasks[task_id] = (future, url)
                    elif task_type == 'delete':
                        doc_ids = params
                        future = self._delete_executor.submit(self._processors[algo_id].delete_doc, doc_ids=doc_ids)
                        self._pending_task_ids.remove(task_id)
                        self._tasks[task_id] = (future, None)
                    time.sleep(0.2)
                except queue.Empty:
                    task_need_pop = []
                    for task_id, (future, url) in self._tasks.items():
                        if future.done():
                            task_need_pop.append(task_id)
                            ex = future.exception()
                            if not ex:
                                if url:
                                    self._send_status_message(
                                        task_id=task_id,
                                        url=url,
                                        success=True,
                                        error_code="",
                                        error_msg=""
                                    )
                            else:
                                if url:
                                    self._send_status_message(
                                        task_id=task_id,
                                        url=url,
                                        success=False,
                                        error_code=type(ex).__name__,
                                        error_msg=str(ex)
                                    )
                                else:
                                    LOG.error(f"task {task_id} failed: {str(ex)}")
                    for task_id in task_need_pop:
                        self._tasks.pop(task_id)
                    time.sleep(5)

    def __init__(self, server: bool = True, port: int = None):
        self._impl = DocumentProcessor.Impl(server=server)
        if server: self._impl = ServerModule(self._impl, port=port)

    def register_algorithm(self, name: str, store: StoreBase, reader: ReaderBase,
                           node_groups: Dict[str, Dict], force_refresh: bool = False):

        if isinstance(self._impl, ServerModule):
            self._impl._call('_init_components', True)
        else:
            self._impl._init_components(server=False)

        if isinstance(self._impl, ServerModule):
            self._impl._call('register_algorithm', name, store, reader, node_groups, force_refresh)
        else:
            self._impl.register_algorithm(name, store, reader, node_groups, force_refresh)
