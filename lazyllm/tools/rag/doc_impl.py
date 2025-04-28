import json
import ast
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union, Tuple, Any
from lazyllm import LOG, once_wrapper
from .transform import (NodeTransform, FuncNodeTransform, SentenceSplitter, LLMParser,
                        AdaptiveTransform, make_transform, TransformArgs, TransformArgs as TArgs)
from .index_base import IndexBase
from .store_base import StoreBase, LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .map_store import MapStore
from .chroma_store import ChromadbStore
from .milvus_store import MilvusStore
from .smart_embedding_index import SmartEmbeddingIndex
from .doc_node import DocNode
from .data_loaders import DirectoryReader
from .utils import DocListManager, gen_docid, is_sparse
from .global_metadata import GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_PATH
from .data_type import DataType
from dataclasses import dataclass
import threading
import time

_transmap = dict(function=FuncNodeTransform, sentencesplitter=SentenceSplitter, llm=LLMParser)

def embed_wrapper(func):
    if not func:
        return None

    @wraps(func)
    def wrapper(*args, **kwargs) -> List[float]:
        result = func(*args, **kwargs)
        return ast.literal_eval(result) if isinstance(result, str) else result

    return wrapper

class StorePlaceholder:
    pass

class EmbedPlaceholder:
    pass


class BuiltinGroups(object):
    @dataclass
    class Struct:
        name: str
        args: TransformArgs
        parent: str = LAZY_ROOT_NAME

        def __str__(self): return self.name

    CoarseChunk = Struct('CoarseChunk', TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=1024, chunk_overlap=100)))
    MediumChunk = Struct('MediumChunk', TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
    FineChunk = Struct('FineChunk', TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=12)))


class DocImpl:
    _builtin_node_groups: Dict[str, Dict] = {}
    _global_node_groups: Dict[str, Dict] = {}
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, embed: Dict[str, Callable], dlm: Optional[DocListManager] = None,
                 doc_files: Optional[str] = None, kb_group_name: Optional[str] = None,
                 global_metadata_desc: Dict[str, GlobalMetadataDesc] = None,
                 store_conf: Optional[Dict] = None):
        super().__init__()
        assert (dlm is None) ^ (doc_files is None), 'Only one of dataset_path or doc_files should be provided'
        self._local_file_reader: Dict[str, Callable] = {}
        self._kb_group_name = kb_group_name or DocListManager.DEFAULT_GROUP_NAME
        self._dlm, self._doc_files = dlm, doc_files
        self._reader = DirectoryReader(None, self._local_file_reader, DocImpl._registered_file_reader)
        self.node_groups: Dict[str, Dict] = {LAZY_ROOT_NAME: {}, LAZY_IMAGE_GROUP: {}}
        self.embed = {k: embed_wrapper(e) for k, e in embed.items()}
        self._global_metadata_desc = global_metadata_desc
        self.store = store_conf  # NOTE: will be initialized in _lazy_init()
        self._activated_embeddings = {}
        self.index_pending_registrations = []

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self) -> None:
        node_groups = DocImpl._builtin_node_groups.copy()
        node_groups.update(DocImpl._global_node_groups)
        node_groups.update(self.node_groups)
        self.node_groups = node_groups

        # set empty embed keys for groups that are not visited by Retriever
        for group in node_groups.keys():
            self._activated_embeddings.setdefault(group, set())

        embed_dims = {}
        embed_datatypes = {}
        for k, e in self.embed.items():
            embedding = e('a')
            if is_sparse(embedding):
                embed_datatypes[k] = DataType.SPARSE_FLOAT_VECTOR
            else:
                embed_dims[k] = len(embedding)
                embed_datatypes[k] = DataType.FLOAT_VECTOR

        if self.store is None:
            self.store = {
                'type': 'map',
            }

        if isinstance(self.store, Dict):
            self.store = self._create_store(store_conf=self.store, embed_dims=embed_dims,
                                            embed_datatypes=embed_datatypes)
        else:
            raise ValueError(f'store type [{type(self.store)}] is not a dict.')
        self._resolve_index_pending_registrations()

        if not self.store.is_group_active(LAZY_ROOT_NAME):
            ids, paths, metadatas = self._list_files()
            ids, paths, metadatas = self._delete_nonexistent_docs_on_startup(ids, paths, metadatas)
            if paths:
                if not metadatas: metadatas = [{} for _ in range(len(paths))]
                for idx, meta in enumerate(metadatas):
                    meta[RAG_DOC_ID] = ids[idx] if ids else gen_docid(paths[idx])
                    meta[RAG_DOC_PATH] = paths[idx]
                root_nodes = self._reader.load_data(paths, metadatas)
                self.store.update_nodes(root_nodes)
                if self._dlm:
                    self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                              new_status=DocListManager.Status.success)
                LOG.debug(f"building {LAZY_ROOT_NAME} nodes: {root_nodes}")
        if self._dlm:
            self._init_monitor_event = threading.Event()
            self._daemon = threading.Thread(target=self.worker)
            self._daemon.daemon = True
            self._daemon.start()
            self._init_monitor_event.wait()

    def _delete_nonexistent_docs_on_startup(self, ids, paths, metadatas):
        path_existing = [Path(path).exists() for path in paths]
        paths_need_delete = [paths[idx] for idx, exist in enumerate(path_existing) if not exist]
        rt_metadatas = [meta for meta, exist in zip(metadatas, path_existing) if exist] if metadatas else None
        rt_ids = [ids[idx] for idx, exist in enumerate(path_existing) if exist] if ids else None
        rt_paths = [path for path, exist in zip(paths, path_existing) if exist]

        if ids:
            ids_need_delete = [ids[idx] for idx, exist in enumerate(path_existing) if not exist]
        else:
            ids_need_delete = [gen_docid(path) for path in paths_need_delete]
        if ids_need_delete:
            if self._dlm is None:
                # if not using dlm, delete store directly;
                self._delete_doc_from_store(paths_need_delete)
            else:
                LOG.warning(f"Found {len(paths_need_delete)} docs that are not in store: {paths_need_delete}")
                # else dlm must turn on path monitoring to detect deleted files
                assert (self._dlm.enable_path_monitoring is True
                        ), 'DocListManager must turn on path monitoring or only use DocManager to delete files'
        return rt_ids, rt_paths, rt_metadatas

    def _resolve_index_pending_registrations(self):
        for index_type, index_cls, index_args, index_kwargs in self.index_pending_registrations:
            args = [self._resolve_index_placeholder(arg) for arg in index_args]
            kwargs = {k: self._resolve_index_placeholder(v) for k, v in index_kwargs.items()}
            self.store.register_index(index_type, index_cls(*args, **kwargs))
        self.index_pending_registrations.clear()

    def _create_store(self, store_conf: Optional[Dict], embed_dims: Optional[Dict[str, int]] = None,
                      embed_datatypes: Optional[Dict[str, DataType]] = None) -> StoreBase:
        store_type = store_conf.get('type')
        if not store_type:
            raise ValueError('store type is not specified.')

        kwargs = store_conf.get('kwargs', {})
        if not isinstance(kwargs, Dict):
            raise ValueError('`kwargs` in store conf is not a dict.')

        if store_type == "map":
            store = MapStore(node_groups=list(self._activated_embeddings.keys()), embed=self.embed, **kwargs)
        elif store_type == "chroma":
            store = ChromadbStore(group_embed_keys=self._activated_embeddings, embed=self.embed,
                                  embed_dims=embed_dims, **kwargs)
        elif store_type == "milvus":
            store = MilvusStore(group_embed_keys=self._activated_embeddings, embed=self.embed,
                                embed_dims=embed_dims, embed_datatypes=embed_datatypes,
                                global_metadata_desc=self._global_metadata_desc, **kwargs)
        else:
            raise NotImplementedError(
                f"Not implemented store type for {store_type}"
            )

        indices_conf = store_conf.get('indices', {})
        if not isinstance(indices_conf, Dict):
            raise ValueError(f"`indices`'s type [{type(indices_conf)}] is not a dict")

        for index_type, conf in indices_conf.items():
            if index_type == 'smart_embedding_index':
                backend_type = conf.get('backend')
                if not backend_type:
                    raise ValueError('`backend` is not specified in `smart_embedding_index`.')
                kwargs = conf.get('kwargs', {})
                index = SmartEmbeddingIndex(backend_type=backend_type,
                                            group_embed_keys=self._activated_embeddings,
                                            embed=self.embed,
                                            embed_dims=embed_dims,
                                            embed_datatypes=embed_datatypes,
                                            global_metadata_desc=self._global_metadata_desc,
                                            **kwargs)
            else:
                raise ValueError(f'unsupported index type [{index_type}]')

            store.register_index(type=index_type, index=index)

        return store

    @staticmethod
    def _create_node_group_impl(cls, group_name, name, transform: Union[str, Callable] = None,
                                parent: str = LAZY_ROOT_NAME, *, trans_node: bool = None,
                                num_workers: int = 0, **kwargs):
        group_name, parent = str(group_name), str(parent)
        groups = getattr(cls, group_name)

        def get_trans(t): return TransformArgs.from_dict(t) if isinstance(t, dict) else t

        if isinstance(transform, (TransformArgs, tuple, list, dict)):
            err_msg = '{} should be set in transform when transform is Dict, TransformArgs or List[TransformArgs]'
            assert trans_node is None, err_msg.format('trans_node')
            assert num_workers == 0, err_msg.format('num_workers')
            assert not kwargs, err_msg.format('kwargs')
            transforms = ([get_trans(t) for t in transform] if isinstance(transform, (list, tuple)) else
                          get_trans(transform))
        else:
            transforms = TransformArgs(f=transform, trans_node=trans_node,
                                       num_workers=num_workers, kwargs=kwargs)

        if name in groups:
            LOG.warning(f"Duplicate group name: {name}")
        for t in (transforms if isinstance(transform, list) else [transforms]):
            if isinstance(t.f, str):
                t.f = _transmap[t.f.lower()]
            if isinstance(t.f, type):
                assert t.trans_node is None, '`trans_node` is allowed only when transform is callable'
                if not issubclass(t.f, NodeTransform): LOG.warning(
                    'Please note! You are trying to use a completely custom transform class. The relationship '
                    'between nodes may become unreliable, `Document.get_parent/get_child` functions and the '
                    'target parameter of Retriever may have strange anomalies. Please use it at your own risk.')
            else:
                assert callable(t.f), f"transform should be callable, but get {t.f}"
        groups[name] = dict(transform=transforms, parent=parent)

    @classmethod
    def _create_builtin_node_group(cls, name, transform: Union[str, Callable] = None, parent: str = LAZY_ROOT_NAME,
                                   *, trans_node: bool = None, num_workers: int = 0, **kwargs) -> None:
        DocImpl._create_node_group_impl(cls, '_builtin_node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, **kwargs)

    @classmethod
    def create_global_node_group(cls, name, transform: Union[str, Callable] = None, parent: str = LAZY_ROOT_NAME,
                                 *, trans_node: bool = None, num_workers: int = 0, **kwargs) -> None:
        DocImpl._create_node_group_impl(cls, '_global_node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, **kwargs)

    def create_node_group(self, name, transform: Union[str, Callable] = None, parent: str = LAZY_ROOT_NAME,
                          *, trans_node: bool = None, num_workers: int = 0, **kwargs) -> None:
        assert not self._lazy_init.flag, 'Cannot add node group after document started'
        DocImpl._create_node_group_impl(self, 'node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, **kwargs)

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        if func is not None:
            cls._registered_file_reader[pattern] = func
            return None

        def decorator(klass):
            if callable(klass): cls._registered_file_reader[pattern] = klass
            else: raise TypeError(f"The registered object {klass} is not a callable object.")
            return klass
        return decorator

    def _resolve_index_placeholder(self, value):
        if isinstance(value, StorePlaceholder): return self.store
        elif isinstance(value, EmbedPlaceholder): return self.embed
        return value

    def register_index(self, index_type: str, index_cls: IndexBase, *args, **kwargs) -> None:
        if bool(self._lazy_init.flag):
            args = [self._resolve_index_placeholder(arg) for arg in args]
            kwargs = {k: self._resolve_index_placeholder(v) for k, v in kwargs.items()}
            self.store.register_index(index_type, index_cls(*args, **kwargs))
        else:
            self.index_pending_registrations.append((index_type, index_cls, args, kwargs))

    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        assert callable(func), 'func for reader should be callable'
        self._local_file_reader[pattern] = func

    def worker(self):
        is_first_run = True
        while True:
            # Apply meta changes
            rows = self._dlm.fetch_docs_changed_meta(self._kb_group_name)
            for row in rows:
                new_meta_dict = json.loads(row[1]) if row[1] else {}
                self.store.update_doc_meta(row[0], new_meta_dict)

            # Step 1: do doc-parsing, highest priority
            docs = self._dlm.get_docs_need_reparse(group=self._kb_group_name)
            if docs:
                filepaths = [doc.path for doc in docs]
                ids = [doc.doc_id for doc in docs]
                metadatas = [json.loads(doc.meta) if doc.meta else None for doc in docs]
                # update status and need_reparse
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.working, new_need_reparse=False)
                self._delete_doc_from_store(filepaths)
                self._add_doc_to_store(input_files=filepaths, ids=ids, metadatas=metadatas)
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.success)

            # Step 2: After doc is deleted from related kb_group, delete doc from db
            if self._kb_group_name == DocListManager.DEFAULT_GROUP_NAME:
                self._dlm.delete_unreferenced_doc()

            # Step 3: do doc-deleting
            ids, files, metadatas = self._list_files(status=DocListManager.Status.deleting)
            if files:
                self._delete_doc_from_store(files)
                self._dlm.delete_files_from_kb_group(ids, self._kb_group_name)

            # Step 4: do doc-adding
            ids, files, metadatas = self._list_files(status=DocListManager.Status.waiting,
                                                     upload_status=DocListManager.Status.success)
            if files:
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.working)
                self._add_doc_to_store(input_files=files, ids=ids, metadatas=metadatas)
                # change working to success while leaving other status unchanged
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          cond_status_list=[DocListManager.Status.working],
                                          new_status=DocListManager.Status.success)
            if is_first_run:
                self._init_monitor_event.set()
            is_first_run = False
            time.sleep(10)

    def _list_files(
            self, status: Union[str, List[str]] = DocListManager.Status.all,
            upload_status: Union[str, List[str]] = DocListManager.Status.all
    ) -> Tuple[List[str], List[str], List[Dict]]:
        if self._doc_files: return None, self._doc_files, None
        ids, paths, metadatas = [], [], []
        for row in self._dlm.list_kb_group_files(group=self._kb_group_name, status=status,
                                                 upload_status=upload_status, details=True):
            ids.append(row[0])
            paths.append(row[1])
            metadatas.append(json.loads(row[3]) if row[3] else {})
        return ids, paths, metadatas

    def _add_doc_to_store(self, input_files: List[str], ids: Optional[List[str]] = None,
                          metadatas: Optional[List[Dict[str, Any]]] = None):
        if not input_files:
            return
        root_nodes = self._reader.load_data(input_files)
        map_file_meta = {}
        if metadatas:
            for file_path, metadata in zip(input_files, metadatas):
                map_file_meta[file_path] = metadata
        for node in root_nodes:
            file_path = node.global_metadata[RAG_DOC_PATH]
            if file_path in map_file_meta:
                node.global_metadata.update(map_file_meta[file_path])
            node.global_metadata[RAG_DOC_ID] = gen_docid(node.docpath)
        temp_store = self._create_store({"type": "map"})
        temp_store.update_nodes(root_nodes)
        all_groups = self.store.all_groups()
        LOG.info(f"add_files: Trying to merge store with {all_groups}")
        for group in all_groups:
            if group != LAZY_ROOT_NAME and not self.store.is_group_active(group):
                continue
            # Duplicate group will be discarded automatically
            nodes = self._get_nodes(group, temp_store)
            self.store.update_nodes(nodes)
            LOG.debug(f"Merge {group} with {nodes}")

    def _delete_doc_from_store(self, input_files: List[str]) -> None:
        docs = self.store.get_index(type='file_node_map').query(input_files)
        LOG.info(f"delete_files: removing documents {input_files} and nodes {docs}")
        if len(docs) == 0:
            return
        self._delete_nodes_recursively(docs)

    def _delete_nodes_recursively(self, root_nodes: List[DocNode]) -> None:
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
            self.store.remove_nodes(group, node_uids)
            LOG.debug(f"Removed nodes from group {group} for node IDs: {node_uids}")

    def _dynamic_create_nodes(self, group_name: str, store: StoreBase) -> None:
        if group_name == LAZY_ROOT_NAME or store.is_group_active(group_name):
            return
        node_group = self.node_groups.get(group_name)
        if node_group is None:
            raise ValueError(f"Node group '{group_name}' does not exist. Please check the group name "
                             "or add a new one through `create_node_group`.")
        t = node_group['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t)
        parent_nodes = self._get_nodes(node_group["parent"], store)
        nodes = transform.batch_forward(parent_nodes, group_name)
        store.update_nodes(nodes)

    def _get_nodes(self, group_name: str, store: Optional[StoreBase] = None) -> List[DocNode]:
        store = store or self.store
        self._dynamic_create_nodes(group_name, store)
        return store.get_nodes(group_name)

    def retrieve(self, query: str, group_name: str, similarity: str, similarity_cut_off: Union[float, Dict[str, float]],
                 index: str, topk: int, similarity_kws: dict, embed_keys: Optional[List[str]] = None,
                 filters: Optional[Dict[str, Union[str, int, List, Set]]] = None) -> List[DocNode]:
        self._lazy_init()
        self._dynamic_create_nodes(group_name, self.store)

        if index is None or index == 'default':
            return self.store.query(query=query, group_name=group_name, similarity_name=similarity,
                                    similarity_cut_off=similarity_cut_off, topk=topk,
                                    embed_keys=embed_keys, filters=filters, **similarity_kws)

        index_instance = self.store.get_index(type=index)
        if not index_instance:
            raise NotImplementedError(f"index type '{index}' is not supported currently.")

        try:
            return index_instance.query(query=query, group_name=group_name, similarity_name=similarity,
                                        similarity_cut_off=similarity_cut_off, topk=topk,
                                        embed_keys=embed_keys, filters=filters, **similarity_kws)
        except Exception as e:
            raise RuntimeError(f'index type `{index}` of store `{type(self.store)}` query failed: {e}')

    def find(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        if len(nodes) == 0: return nodes
        self._lazy_init()
        self._dynamic_create_nodes(group, self.store)

        def get_depth(name):
            cnt = 0
            while name != LAZY_ROOT_NAME:
                cnt += 1
                name = self.node_groups[name]['parent']
            return cnt

        # 1. find lowest common ancestor
        left, right = nodes[0]._group, group
        curr_depth, target_depth = get_depth(left), get_depth(right)
        if curr_depth > target_depth:
            for i in range(curr_depth - target_depth): left = self.node_groups[left]['parent']
        elif curr_depth < target_depth:
            for i in range(target_depth - curr_depth): right = self.node_groups[right]['parent']
        while (left != right):
            left = self.node_groups[left]['parent']
            right = self.node_groups[right]['parent']
        ancestor = left

        # 2. if ancestor != current group, go to ancestor; then if ancestor != target group, go to target group
        if nodes and nodes[0]._group != ancestor:
            nodes = DocImpl.find_parent(nodes, ancestor)
        if nodes and nodes[0]._group != group:
            nodes = DocImpl.find_children(nodes, group)
        return nodes

    @staticmethod
    def find_parent(nodes: List[DocNode], group: str) -> List[DocNode]:
        def recurse_parents(node: DocNode, visited: Set[DocNode]) -> None:
            if node.parent:
                if node.parent._group == group:
                    visited.add(node.parent)
                else:
                    recurse_parents(node.parent, visited)

        result = set()
        for node in nodes:
            recurse_parents(node, result)
        if not result:
            LOG.warning(
                f"We can not find any nodes for group `{group}`, please check your input"
            )
        LOG.debug(f"Found parent node for {group}: {result}")
        return list(result)

    @staticmethod
    def find_children(nodes: List[DocNode], group: str) -> List[DocNode]:
        def recurse_children(node: DocNode, visited: Set[DocNode]) -> bool:
            if group in node.children:
                visited.update(node.children[group])
                return True

            found_in_any_child = False

            for children_list in node.children.values():
                for child in children_list:
                    if recurse_children(child, visited):
                        found_in_any_child = True
                    else:
                        break

            return found_in_any_child

        result = set()

        for node in nodes:
            if group in node.children:
                result.update(node.children[group])
            else:
                if not recurse_children(node, result):
                    LOG.warning(
                        f"Node {node} and its children do not contain any nodes with the group `{group}`. "
                        "Skipping further search in this branch."
                    )
                    break

        if not result:
            LOG.warning(
                f"We cannot find any nodes for group `{group}`, please check your input."
            )

        LOG.debug(f"Found children nodes for {group}: {result}")
        return list(result)

    def clear_cache(self, group_names: Optional[List[str]] = None):
        self.store.clear_cache(group_names)

    def __call__(self, func_name: str, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)


for k, v in BuiltinGroups.__dict__.items():
    if not k.startswith('_') and isinstance(v, BuiltinGroups.Struct):
        assert k == v.name, 'builtin group name mismatch'
        DocImpl._create_builtin_node_group(name=k, transform=v.args, parent=v.parent)
