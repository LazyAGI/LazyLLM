import ast
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Union
from lazyllm import LOG, config, once_wrapper
from .transform import (NodeTransform, FuncNodeTransform, SentenceSplitter, LLMParser,
                        AdaptiveTransform, make_transform, TransformArgs)
from .store import MapStore, DocNode, ChromadbStore, LAZY_ROOT_NAME, BaseStore
from .data_loaders import DirectoryReader
from .index import DefaultIndex

_transmap = dict(function=FuncNodeTransform, sentencesplitter=SentenceSplitter, llm=LLMParser)


def embed_wrapper(func):
    if not func:
        return None

    @wraps(func)
    def wrapper(*args, **kwargs) -> List[float]:
        result = func(*args, **kwargs)
        return ast.literal_eval(result) if isinstance(result, str) else result

    return wrapper


class DocImpl:
    _builtin_node_groups = {}
    _global_node_groups = {}

    def __init__(self, embed: Dict[str, Callable], doc_files=Optional[List[str]],
                 local_readers: Optional[Dict] = None, global_readers: Optional[Dict] = None, **kwargs):
        super().__init__()
        self.directory_reader = DirectoryReader(doc_files, local_readers=local_readers, global_readers=global_readers)
        self.node_groups: Dict[str, Dict] = {LAZY_ROOT_NAME: {}}
        self.embed = {k: embed_wrapper(e) for k, e in embed.items()}
        self.store = None

    @once_wrapper(reset_on_pickle=True)
    def _lazy_init(self) -> None:
        node_groups = DocImpl._builtin_node_groups.copy()
        node_groups.update(DocImpl._global_node_groups)
        node_groups.update(self.node_groups)
        self.node_groups = node_groups

        self.store = self._get_store()
        self.index = DefaultIndex(self.embed, self.store)
        if not self.store.has_nodes(LAZY_ROOT_NAME):
            root_nodes = self.directory_reader.load_data()
            self.store.add_nodes(root_nodes)
            LOG.debug(f"building {LAZY_ROOT_NAME} nodes: {root_nodes}")

    def _get_store(self) -> BaseStore:
        rag_store_type = config["rag_store_type"]
        if rag_store_type == "map":
            store = MapStore(node_groups=self.node_groups.keys())
        elif rag_store_type == "chroma":
            store = ChromadbStore(node_groups=self.node_groups.keys(), embed=self.embed)
            store.try_load_store()
        else:
            raise NotImplementedError(
                f"Not implemented store type for {rag_store_type}"
            )
        return store

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

    def add_files(self, input_files: List[str]) -> None:
        if len(input_files) == 0:
            return
        self._lazy_init()
        root_nodes = self.directory_reader.load_data(input_files)
        temp_store = self._get_store()
        temp_store.add_nodes(root_nodes)
        active_groups = self.store.active_groups()
        LOG.info(f"add_files: Trying to merge store with {active_groups}")
        for group in active_groups:
            # Duplicate group will be discarded automatically
            nodes = self._get_nodes(group, temp_store)
            self.store.add_nodes(nodes)
            LOG.debug(f"Merge {group} with {nodes}")

    def delete_files(self, input_files: List[str]) -> None:
        self._lazy_init()
        docs = self.store.get_nodes_by_files(input_files)
        LOG.info(f"delete_files: removing documents {input_files} and nodes {docs}")
        if len(docs) == 0:
            return
        self._delete_nodes_recursively(docs)

    def _delete_nodes_recursively(self, root_nodes: List[DocNode]) -> None:
        nodes_to_delete = defaultdict(list)
        nodes_to_delete[LAZY_ROOT_NAME] = root_nodes

        # Gather all nodes to be deleted including their children
        def gather_children(node: DocNode):
            for children_group, children_list in node.children.items():
                for child in children_list:
                    nodes_to_delete[children_group].append(child)
                    gather_children(child)

        for node in root_nodes:
            gather_children(node)

        # Delete nodes in all groups
        for group, node_uids in nodes_to_delete.items():
            self.store.remove_nodes(node_uids)
            LOG.debug(f"Removed nodes from group {group} for node IDs: {node_uids}")

    def _dynamic_create_nodes(self, group_name: str, store: BaseStore) -> None:
        if store.has_nodes(group_name):
            return
        node_group = self.node_groups.get(group_name)
        if node_group is None:
            raise ValueError(f"Node group '{group_name}' does not exist. Please check the group name "
                             "or add a new one through `create_node_group`.")
        t = node_group['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t)
        parent_nodes = self._get_nodes(node_group["parent"], store)
        nodes = transform.batch_forward(parent_nodes, group_name)
        store.add_nodes(nodes)
        LOG.debug(f"building {group_name} nodes: {nodes}")

    def _get_nodes(self, group_name: str, store: Optional[BaseStore] = None) -> List[DocNode]:
        store = store or self.store
        self._dynamic_create_nodes(group_name, store)
        return store.traverse_nodes(group_name)

    def retrieve(self, query: str, group_name: str, similarity: str, similarity_cut_off: Union[float, Dict[str, float]],
                 index: str, topk: int, similarity_kws: dict, embed_keys: Optional[List[str]] = None) -> List[DocNode]:
        self._lazy_init()
        if index:
            assert index == "default", "we only support default index currently"
        nodes = self._get_nodes(group_name)
        return self.index.query(
            query, nodes, similarity, similarity_cut_off, topk, embed_keys, **similarity_kws
        )

    def find_parent(self, nodes: List[DocNode], group: str) -> List[DocNode]:
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

    def find_children(self, nodes: List[DocNode], group: str) -> List[DocNode]:
        active_groups = self.store.active_groups()
        if group not in active_groups:
            raise ValueError(
                f"group {group} not found in active groups {active_groups}, please retrieve the group first."
            )

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
