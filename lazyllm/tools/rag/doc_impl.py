import json
import ast
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Union, Tuple, Any
from lazyllm import LOG, once_wrapper
from .transform import (NodeTransform, FuncNodeTransform, SentenceSplitter, LLMParser,
                        TransformArgs, TransformArgs as TArgs)
from .index_base import IndexBase
from .store import (MapStore, MilvusStore, ChromadbStore, SenseCoreStore, StoreBase,
                    LAZY_ROOT_NAME, LAZY_IMAGE_GROUP)
from .smart_embedding_index import SmartEmbeddingIndex
from .doc_node import DocNode
from .data_loaders import DirectoryReader
from .utils import DocListManager, is_sparse
from .global_metadata import GlobalMetadataDesc
from .data_type import DataType
from .doc_processor import _Processor, DocumentProcessor
from dataclasses import dataclass
import threading
import time
from itertools import repeat

_transmap = dict(function=FuncNodeTransform, sentencesplitter=SentenceSplitter, llm=LLMParser)

def embed_wrapper(func: Optional[Callable[..., Any]]) -> Optional[Callable[..., List[float]]]:
    if not func:
        return None

    @wraps(func)
    def wrapper(*args, **kwargs) -> List[float]:
        result = func(*args, **kwargs)
        if isinstance(result, str):
            try:
                # Use json.loads as it's generally more robust for list-like strings
                return json.loads(result)
            except json.JSONDecodeError:
                # Fallback or raise error if json.loads also fails
                # For example, if ast.literal_eval was truly necessary for some non-JSON compatible Python literal
                try:
                    LOG.warning("json.loads failed, attempting ast.literal_eval as a "
                                "fallback (might hit recursion limit).")
                    return ast.literal_eval(result)
                except Exception as e:
                    LOG.error(f"Both json.loads and ast.literal_eval failed. Error: {e}")
                    raise  # Re-raise the original or a new error
        # Explicitly check if it's already a list for dense embedding or dict for sparse embedding
        elif isinstance(result, (list, dict)):
            return result
        else:
            # Handle unexpected types by raising an error
            error_message = f"Expected List[float] or str (convertible to List[float]), but got {type(result)}"
            LOG.error(f"{error_message}")
            raise TypeError(error_message)

    return wrapper

class StorePlaceholder:
    pass

class EmbedPlaceholder:
    pass


class NodeGroupType(str, Enum):
    """An enumeration."""
    ORIGINAL = "Original Source"
    CHUNK = "Chunk"
    SUMMARY = "Summary"
    IMAGE_INFO = "Image Info"
    QUESTION_ANSWER = "Question Answer"
    OTHER = "Other"


class BuiltinGroups(object):
    @dataclass
    class Struct:
        name: str
        display_name: str
        group_type: NodeGroupType
        args: TransformArgs
        parent: str = LAZY_ROOT_NAME
        trans_node: bool = None

        def __str__(self): return self.name

    CoarseChunk = Struct('CoarseChunk', '1024 Tokens Chunk', NodeGroupType.CHUNK,
                         TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=1024, chunk_overlap=100)))
    MediumChunk = Struct('MediumChunk', '256 Tokens Chunk', NodeGroupType.CHUNK,
                         TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
    FineChunk = Struct('FineChunk', '128 Tokens Chunk', NodeGroupType.CHUNK,
                       TArgs(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=12)))
    ImgDesc = Struct('ImgDesc', 'Image Desc', NodeGroupType.OTHER,
                     lambda x: x._content, LAZY_IMAGE_GROUP, True)


class DocImpl:
    _builtin_node_groups: Dict[str, Dict] = {}
    _global_node_groups: Dict[str, Dict] = {}
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, embed: Dict[str, Callable], dlm: Optional[DocListManager] = None,
                 doc_files: Optional[str] = None, kb_group_name: Optional[str] = None,
                 global_metadata_desc: Dict[str, GlobalMetadataDesc] = None, store_conf: Optional[Dict] = None,
                 processor: Optional[DocumentProcessor] = None, algo_name: Optional[str] = None):
        super().__init__()
        self._local_file_reader: Dict[str, Callable] = {}
        self._kb_group_name = kb_group_name or DocListManager.DEFAULT_GROUP_NAME
        self._dlm, self._doc_files = dlm, doc_files
        self._reader = DirectoryReader(None, self._local_file_reader, DocImpl._registered_file_reader)
        self.node_groups: Dict[str, Dict] = {
            LAZY_ROOT_NAME: dict(parent=None, display_name="Original Source", group_type=NodeGroupType.ORIGINAL),
            LAZY_IMAGE_GROUP: dict(parent=None, display_name="Image Node", group_type=NodeGroupType.OTHER)
        }
        self.embed = {k: embed_wrapper(e) for k, e in embed.items()}
        self._global_metadata_desc = global_metadata_desc
        self.store = store_conf  # NOTE: will be initialized in _lazy_init()
        self._activated_groups = set([LAZY_ROOT_NAME, LAZY_IMAGE_GROUP])
        # activated_embeddings maintains all node_groups and active embeddings
        self._activated_embeddings = {LAZY_ROOT_NAME: set(), LAZY_IMAGE_GROUP: set()}  # {group_name: {em1, em2, ...}}
        self._index_pending_registrations = []
        self._processor = processor
        self._algo_name = algo_name

    def _init_node_groups(self):
        node_groups = DocImpl._builtin_node_groups.copy()
        node_groups.update(DocImpl._global_node_groups)
        node_groups.update(self.node_groups)
        self.node_groups = node_groups

        for group in node_groups: self._activated_embeddings.setdefault(group, set())
        self._activated_groups = set([g for g in self._activated_groups if g in node_groups])

        # use list to avoid `dictionary changed size during iteration` error
        for group in list(self._activated_groups):
            while True:
                parent_group = self.node_groups[group]['parent']
                if not parent_group or parent_group in self._activated_groups: break
                self._activated_groups.add(group := parent_group)

    def _init_store(self):
        if self.store is None: self.store = {'type': 'map'}
        embed_dims, embed_datatypes = {}, {}
        for k, e in self.embed.items():
            embedding = e('a')
            if is_sparse(embedding):
                embed_datatypes[k] = DataType.SPARSE_FLOAT_VECTOR
            else:
                embed_dims[k] = len(embedding)
                embed_datatypes[k] = DataType.FLOAT_VECTOR

        if isinstance(self.store, Dict):
            self.store = self._create_store(store_conf=self.store, embed_dims=embed_dims,
                                            embed_datatypes=embed_datatypes)
        elif not isinstance(self.store, StoreBase):
            raise ValueError(f'store type [{type(self.store)}] is not a dict.')

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self) -> None:
        self._init_node_groups()
        self._init_store()
        cloud = not (self._dlm or self._doc_files is not None)

        self._resolve_index_pending_registrations()
        if self._processor:
            assert cloud and isinstance(self._processor, DocumentProcessor)
            self._processor.register_algorithm(self._algo_name, self.store, self._reader, self.node_groups)
        else:
            self._processor = _Processor(self.store, self._reader, self.node_groups)

        # init files when `cloud` is False
        if not cloud and not self.store.is_group_active(LAZY_ROOT_NAME):
            ids, pathes, metadatas = self._list_files(upload_status=DocListManager.Status.success)
            self._processor.add_doc(pathes, ids, metadatas)
            if pathes and self._dlm:
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.success)

        if self._dlm:
            self._init_monitor_event = threading.Event()
            self._daemon = threading.Thread(target=self.worker)
            self._daemon.daemon = True
            self._daemon.start()
            self._init_monitor_event.wait()

    def _resolve_index_pending_registrations(self):
        for index_type, index_cls, index_args, index_kwargs in self._index_pending_registrations:
            args = [self._resolve_index_placeholder(arg) for arg in index_args]
            kwargs = {k: self._resolve_index_placeholder(v) for k, v in index_kwargs.items()}
            self.store.register_index(index_type, index_cls(*args, **kwargs))
        self._index_pending_registrations.clear()

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
        elif store_type == "sensecore":
            store = SenseCoreStore(group_embed_keys=self._activated_embeddings,
                                   global_metadata_desc=self._global_metadata_desc, **kwargs)
        else:
            raise NotImplementedError(
                f"Not implemented store type for {store_type}"
            )
        store.activate_group(self._activated_groups)

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
    def _create_node_group_impl(cls, group_name, name, transform: Union[str, Callable],
                                parent: str = LAZY_ROOT_NAME, *, trans_node: Optional[bool] = None,
                                num_workers: int = 0, display_name: Optional[str] = None,
                                group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs):
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
        groups[name] = dict(transform=transforms, parent=parent, display_name=display_name or name,
                            group_type=group_type)

    @classmethod
    def _create_builtin_node_group(cls, name, transform: Union[str, Callable], parent: str = LAZY_ROOT_NAME,
                                   *, trans_node: Optional[bool] = None, num_workers: int = 0,
                                   display_name: Optional[str] = None, group_type: NodeGroupType = NodeGroupType.CHUNK,
                                   **kwargs) -> None:
        DocImpl._create_node_group_impl(cls, '_builtin_node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, display_name=display_name,
                                        group_type=group_type, **kwargs)

    @classmethod
    def create_global_node_group(cls, name, transform: Union[str, Callable], parent: str = LAZY_ROOT_NAME, *,
                                 trans_node: Optional[bool] = None, num_workers: int = 0,
                                 display_name: Optional[str] = None,
                                 group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs) -> None:
        DocImpl._create_node_group_impl(cls, '_global_node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, display_name=display_name,
                                        group_type=group_type, **kwargs)

    def create_node_group(self, name, transform: Union[str, Callable], parent: str = LAZY_ROOT_NAME, *,
                          trans_node: Optional[bool] = None, num_workers: int = 0, display_name: Optional[str] = None,
                          group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs) -> None:
        assert not self._lazy_init.flag, 'Cannot add node group after document started'
        DocImpl._create_node_group_impl(self, 'node_groups', name=name, transform=transform, parent=parent,
                                        trans_node=trans_node, num_workers=num_workers, display_name=display_name,
                                        group_type=group_type, **kwargs)

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
            self._index_pending_registrations.append((index_type, index_cls, args, kwargs))

    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        assert callable(func), 'func for reader should be callable'
        self._local_file_reader[pattern] = func

    def _add_doc_to_store_with_status(self, input_files: List[str], ids: List[str], metadatas: List[Dict[str, Any]],
                                      cond_status_list: Optional[List[str]] = None):
        success_ids, failed_ids = [], []
        for filepath, doc_id, metadata in zip(input_files, ids or repeat(None), metadatas or repeat(None)):
            try:
                self._add_doc_to_store(input_files=[filepath], ids=[doc_id] if doc_id is not None else None,
                                       metadatas=[metadata] if metadata is not None else None)
                success_ids.append(doc_id)
            except Exception as e:
                LOG.error(f"Error adding document {doc_id} ({filepath}) to store: {e}")
                failed_ids.append(doc_id)

        if success_ids:
            self._dlm.update_kb_group(cond_file_ids=success_ids, cond_group=self._kb_group_name,
                                      cond_status_list=cond_status_list, new_status=DocListManager.Status.success)
        if failed_ids:
            self._dlm.update_kb_group(cond_file_ids=failed_ids, cond_group=self._kb_group_name,
                                      cond_status_list=cond_status_list, new_status=DocListManager.Status.failed)

    def worker(self):
        is_first_run = True
        while True:
            # Apply meta changes
            rows = self._dlm.fetch_docs_changed_meta(self._kb_group_name)
            for row in rows:
                new_meta_dict = json.loads(row[1]) if row[1] else {}
                self._processor.update_doc_meta(doc_id=row[0], metadata=new_meta_dict)

            # Step 1: do doc-parsing, highest priority
            docs = self._dlm.get_docs_need_reparse(group=self._kb_group_name)
            if docs:
                filepaths = [doc.path for doc in docs]
                ids = [doc.doc_id for doc in docs]
                metadatas = [json.loads(doc.meta) if doc.meta else None for doc in docs]
                # update status and need_reparse
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.working, new_need_reparse=False)
                self._delete_doc_from_store(doc_ids=ids)
                self._add_doc_to_store_with_status(filepaths, ids, metadatas)

            # Step 2: After doc is deleted from related kb_group, delete doc from db
            if self._kb_group_name == DocListManager.DEFAULT_GROUP_NAME:
                self._dlm.delete_unreferenced_doc()

            # Step 3: do doc-deleting
            ids, files, metadatas = self._list_files(status=DocListManager.Status.deleting)
            if files:
                self._delete_doc_from_store(doc_ids=ids)
                self._dlm.delete_files_from_kb_group(ids, self._kb_group_name)

            # Step 4: do doc-adding
            ids, files, metadatas = self._list_files(status=DocListManager.Status.waiting,
                                                     upload_status=DocListManager.Status.success)
            if files:
                self._dlm.update_kb_group(cond_file_ids=ids, cond_group=self._kb_group_name,
                                          new_status=DocListManager.Status.working)
                self._add_doc_to_store_with_status(files, ids, metadatas,
                                                   cond_status_list=[DocListManager.Status.working])

            if is_first_run:
                self._init_monitor_event.set()
            is_first_run = False
            time.sleep(10)

    def _list_files(
            self, status: Union[str, List[str]] = DocListManager.Status.all,
            upload_status: Union[str, List[str]] = DocListManager.Status.all
    ) -> Tuple[List[str], List[str], List[Dict]]:
        if self._doc_files is not None: return None, self._doc_files, None
        if not self._dlm: return [], [], []
        ids, paths, metadatas = [], [], []
        for row in self._dlm.list_kb_group_files(group=self._kb_group_name, status=status,
                                                 upload_status=upload_status, details=True):
            ids.append(row[0])
            paths.append(row[1])
            metadatas.append(json.loads(row[3]) if row[3] else {})
        return ids, paths, metadatas

    def _add_doc_to_store(self, input_files: List[str], ids: Optional[List[str]] = None,
                          metadatas: Optional[List[Dict[str, Any]]] = None):
        if not input_files: return
        self._processor.add_doc(input_files, ids, metadatas)

    def _delete_doc_from_store(self, doc_ids: List[str] = None) -> None:
        self._processor.delete_doc(doc_ids=doc_ids)

    def activate_group(self, group_name: str, embed_keys: List[str]):
        group_name = str(group_name)
        self._activated_groups.add(group_name)
        if embed_keys:
            activated_embeddings = self._activated_embeddings.setdefault(group_name, set())
            if len(set(embed_keys) - activated_embeddings) == 0: return
            activated_embeddings.update(embed_keys)

        if self._lazy_init.flag:
            if group_name not in self.node_groups: return
            assert not embed_keys, 'Cannot add new embed_keys for node_group when Document is inited'
            self.store.activate_group(parent := group_name)
            while True:
                parent = self.node_groups[parent]['parent']
                if parent in self._activated_groups: break
                self.store.activate_group(parent)
                self._activated_groups.add(parent)
            # BUG: when using reparse here, nodes created from recurse method will not be set children correctly
            # (For parent nodes has been upserted before creating child nodes)
            if not self.store.is_group_active(group_name): self._processor.reparse(group_name)

    def active_node_groups(self):
        return {k: v for k, v in self._activated_embeddings.items() if k in self._activated_groups}

    def retrieve(self, query: str, group_name: str, similarity: str, similarity_cut_off: Union[float, Dict[str, float]],
                 index: str, topk: int, similarity_kws: dict, embed_keys: Optional[List[str]] = None,
                 filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[DocNode]:
        self._lazy_init()
        if index is None or index == 'default':
            nodes = self.store.query(query=query, group_name=group_name, similarity_name=similarity,
                                     similarity_cut_off=similarity_cut_off, topk=topk, embed_keys=embed_keys,
                                     filters=filters, **similarity_kws, **kwargs)
        else:
            index_instance = self.store.get_index(type=index)
            if not index_instance:
                raise NotImplementedError(f"index type '{index}' is not supported currently.")

            try:
                nodes = index_instance.query(query=query, group_name=group_name, similarity_name=similarity,
                                             similarity_cut_off=similarity_cut_off, topk=topk,
                                             embed_keys=embed_keys, filters=filters, **similarity_kws, **kwargs)
            except Exception as e:
                raise RuntimeError(f'index type `{index}` of store `{type(self.store)}` query failed: {e}')
        for n in nodes:
            n._store = self.store
            n._node_groups = self.node_groups
            n._children_loaded = False
        return nodes

    def find(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        if len(nodes) == 0: return nodes
        self._lazy_init()

        def get_depth(name):
            cnt = 0
            while name != LAZY_ROOT_NAME:
                cnt += 1
                name = self.node_groups[name]['parent']
            return cnt

        for n in nodes:
            n._store = self.store
            n._node_groups = self.node_groups

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
            nodes = self.find_parent(nodes, ancestor)
        if nodes and nodes[0]._group != group:
            nodes = self.find_children(nodes, group)
        return nodes

    def find_parent(self, nodes: List[DocNode], group: str) -> List[DocNode]:
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

    def find_children(self, nodes: List[DocNode], group: str) -> List[DocNode]:  # noqa:C901
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
        DocImpl._create_builtin_node_group(name=k, display_name=v.display_name, group_type=v.group_type,
                                           transform=v.args, parent=v.parent, trans_node=v.trans_node)
