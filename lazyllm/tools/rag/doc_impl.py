import ast
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Union, Tuple
from lazyllm import LOG, config, once_wrapper
from .transform import (NodeTransform, FuncNodeTransform, SentenceSplitter, LLMParser,
                        AdaptiveTransform, make_transform, TransformArgs)
from .store import MapStore, DocNode, ChromadbStore, LAZY_ROOT_NAME, BaseStore, StoreWrapper
from .data_loaders import DirectoryReader
from .index import DefaultIndex, BaseIndex
from .utils import DocListManager
import threading
import time

_transmap = dict(function=FuncNodeTransform, sentencesplitter=SentenceSplitter, llm=LLMParser)

class FileNodeIndex(BaseIndex):
    def __init__(self):
        self._file_node_map = {}

    # override
    def update(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            if node.group != LAZY_ROOT_NAME:
                continue
            file_name = node.metadata.get("file_name")
            if file_name:
                self._file_node_map[file_name] = node

    # override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        # group_name is ignored
        left = {k: v for k, v in self._file_node_map.items() if v.uid not in uids}
        self._file_node_map = left

    # override
    def query(self, files: List[str]) -> List[DocNode]:
        ret = []
        for file in files:
            ret.append(self._file_node_map.get(file))
        return ret


def embed_wrapper(func):
    if not func:
        return None

    @wraps(func)
    def wrapper(*args, **kwargs) -> List[float]:
        result = func(*args, **kwargs)
        return ast.literal_eval(result) if isinstance(result, str) else result

    return wrapper


class DocImpl:
    _builtin_node_groups: Dict[str, Dict] = {}
    _global_node_groups: Dict[str, Dict] = {}
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, embed: Dict[str, Callable], dlm: Optional[DocListManager] = None,
                 doc_files: Optional[str] = None, kb_group_name: Optional[str] = None,
                 store: Optional[BaseStore] = None):
        super().__init__()
        assert (dlm is None) ^ (doc_files is None), 'Only one of dataset_path or doc_files should be provided'
        self._local_file_reader: Dict[str, Callable] = {}
        self._kb_group_name = kb_group_name or DocListManager.DEDAULT_GROUP_NAME
        self._dlm, self._doc_files = dlm, doc_files
        self._reader = DirectoryReader(None, self._local_file_reader, DocImpl._registered_file_reader)
        self.node_groups: Dict[str, Dict] = {LAZY_ROOT_NAME: {}}
        self.embed = {k: embed_wrapper(e) for k, e in embed.items()}
        self._embed_dim = None
        if store:
            self.store = StoreWrapper(store)
            self._create_some_indices_for_store(self.store)
        else:
            self.store = None

    @staticmethod
    def _create_file_node_index(store) -> FileNodeIndex:
        index = FileNodeIndex()
        for group in store.group_names():
            index.update(store.get_group_nodes(group))
        return index

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self) -> None:
        node_groups = DocImpl._builtin_node_groups.copy()
        node_groups.update(DocImpl._global_node_groups)
        node_groups.update(self.node_groups)
        self.node_groups = node_groups

        self._embed_dim = {k: len(e('a')) for k, e in self.embed.items()}

        if not self.store:
            self.store = self._create_store()
            self._create_some_indices_for_store(self.store)

        if not self.store.group_is_active(LAZY_ROOT_NAME):
            ids, pathes = self._list_files()
            root_nodes = self._reader.load_data(pathes)
            self.store.update_nodes(root_nodes)
            if self._dlm: self._dlm.update_kb_group_file_status(
                ids, DocListManager.Status.success, group=self._kb_group_name)
            LOG.debug(f"building {LAZY_ROOT_NAME} nodes: {root_nodes}")

        if self._dlm:
            self._daemon = threading.Thread(target=self.worker)
            self._daemon.daemon = True
            self._daemon.start()

    def _create_store(self, rag_store_type: str = None) -> BaseStore:
        if not rag_store_type:
            rag_store_type = config["rag_store_type"]
        if rag_store_type == "map":
            store = MapStore(node_groups=self.node_groups.keys())
        elif rag_store_type == "chroma":
            store = ChromadbStore(node_groups=self.node_groups.keys(), embed_dim=self._embed_dim)
        else:
            raise NotImplementedError(
                f"Not implemented store type for {rag_store_type}"
            )
        return store

    def _create_some_indices_for_store(self, store: BaseStore):
        if not store.get_index('default'):
            store.register_index(type='default', index=DefaultIndex(self.embed, store))
        if not store.get_index('file_node_map'):
            store.register_index(type='file_node_map', index=self._create_file_node_index(store))

    @staticmethod
    def _create_node_group_impl(cls, group_name, name, transform: Union[str, Callable] = None,
                                parent: str = LAZY_ROOT_NAME, *, trans_node: bool = None,
                                num_workers: int = 0, **kwargs):
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

    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        assert callable(func), 'func for reader should be callable'
        self._local_file_reader[pattern] = func

    def worker(self):
        while True:
            ids, files = self._list_files(status=DocListManager.Status.deleting)
            if files:
                self._delete_files(files)
                self._dlm.delete_files_from_kb_group(ids, self._kb_group_name)
                continue

            if self._kb_group_name == DocListManager.DEDAULT_GROUP_NAME:
                self._dlm.init_tables()
            ids, files = self._list_files(status=DocListManager.Status.waiting,
                                          upload_status=DocListManager.Status.success)
            if files:
                self._dlm.update_kb_group_file_status(ids, DocListManager.Status.working, group=self._kb_group_name)
                self._add_files(files)
                self._dlm.update_kb_group_file_status(ids, DocListManager.Status.success, group=self._kb_group_name)
                continue
            time.sleep(10)

    def _list_files(self, status: str = DocListManager.Status.all, upload_status: str = DocListManager.Status.all
                    ) -> Tuple[List[str], List[str]]:
        if self._doc_files: return None, self._doc_files
        ids, paths = [], []
        for row in self._dlm.list_kb_group_files(group=self._kb_group_name, status=status,
                                                 upload_status=upload_status, details=True):
            ids.append(row[0])
            paths.append(row[1])
        return ids, paths

    def _add_files(self, input_files: List[str]):
        if len(input_files) == 0:
            return
        self._lazy_init()
        root_nodes = self._reader.load_data(input_files)
        temp_store = self._create_store("map")
        temp_store.update_nodes(root_nodes)
        group_names = self.store.group_names()
        LOG.info(f"add_files: Trying to merge store with {group_names}")
        for group in group_names:
            if not self.store.group_is_active(group):
                continue
            # Duplicate group will be discarded automatically
            nodes = self._get_nodes(group, temp_store)
            self.store.update_nodes(nodes)
            LOG.debug(f"Merge {group} with {nodes}")

    def _delete_files(self, input_files: List[str]) -> None:
        self._lazy_init()
        docs = self.store.get_index('file_node_map').query(input_files)
        LOG.info(f"delete_files: removing documents {input_files} and nodes {docs}")
        if len(docs) == 0:
            return
        self._delete_nodes_recursively(docs)

    def _delete_nodes_recursively(self, root_nodes: List[DocNode]) -> None:
        uids_to_delete = defaultdict(list)
        uids_to_delete[LAZY_ROOT_NAME] = [node.uid for node in root_nodes]

        # Gather all nodes to be deleted including their children
        def gather_children(node: DocNode):
            for children_group, children_list in node.children.items():
                for child in children_list:
                    uids_to_delete[children_group].append(child.uid)
                    gather_children(child)

        for node in root_nodes:
            gather_children(node)

        # Delete nodes in all groups
        for group, node_uids in uids_to_delete.items():
            self.store.remove_group_nodes(group, node_uids)
            LOG.debug(f"Removed nodes from group {group} for node IDs: {node_uids}")

    def _dynamic_create_nodes(self, group_name: str, store: BaseStore) -> None:
        if store.group_is_active(group_name):
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
        LOG.debug(f"building {group_name} nodes: {nodes}")

    def _get_nodes(self, group_name: str, store: Optional[BaseStore] = None) -> List[DocNode]:
        store = store or self.store
        self._dynamic_create_nodes(group_name, store)
        return store.get_group_nodes(group_name)

    def retrieve(self, query: str, group_name: str, similarity: str, similarity_cut_off: Union[float, Dict[str, float]],
                 index: str, topk: int, similarity_kws: dict, embed_keys: Optional[List[str]] = None) -> List[DocNode]:
        self._lazy_init()

        index_instance = self.store.get_index(index)
        if not index_instance:
            raise NotImplementedError(f"index type '{index}' is not supported currently.")

        self._dynamic_create_nodes(group_name, self.store)
        return index_instance.query(query=query, group_name=group_name, similarity_name=similarity,
                                    similarity_cut_off=similarity_cut_off, topk=topk,
                                    embed_keys=embed_keys, **similarity_kws)

    @staticmethod
    def find_parent(nodes: List[DocNode], group: str) -> List[DocNode]:
        def recurse_parents(node: DocNode, visited: Set[DocNode]) -> None:
            if node.parent:
                if node.parent.group == group:
                    visited.add(node.parent)
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
                LOG.log_once(
                    f"Fetching children that are not in direct relationship might be slower. "
                    f"We recommend first fetching through direct children {list(node.children.keys())}, "
                    f"then using `find_children()` again for deeper levels.",
                    level="warning",
                )
                # Note: the input nodes are the same type
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


DocImpl._create_builtin_node_group(name="CoarseChunk", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
DocImpl._create_builtin_node_group(name="MediumChunk", transform=SentenceSplitter, chunk_size=256, chunk_overlap=25)
DocImpl._create_builtin_node_group(name="FineChunk", transform=SentenceSplitter, chunk_size=128, chunk_overlap=12)
