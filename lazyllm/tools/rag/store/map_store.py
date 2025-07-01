from typing import Dict, List, Optional, Callable, Union, Set

from .store_base import StoreBase, LAZY_ROOT_NAME

from ..index_base import IndexBase
from ..doc_node import DocNode
from ..default_index import DefaultIndex
from ..global_metadata import RAG_SYSTEM_META_KEYS, RAG_DOC_ID
from ..utils import _FileNodeIndex

from lazyllm.common import override


class MapStore(StoreBase):
    def __init__(self, node_groups: Union[List[str], Set[str]], embed: Dict[str, Callable], **kwargs):
        self._group2uids: Dict[str, Set[str]] = {
            group: set() for group in node_groups
        }
        self._uid2node: Dict[str, DocNode] = {}
        self._name2index = {
            'default': DefaultIndex(embed, self),
            'file_node_map': _FileNodeIndex(),
        }
        self._activated_groups = set()

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._uid2node[node._uid] = node
            self._group2uids[node._group].add(node._uid)

        for index in self._name2index.values():
            index.update(nodes)

    @override
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        doc_nodes: List[DocNode] = self.get_nodes(group_name=LAZY_ROOT_NAME, doc_ids=[doc_id])
        if not doc_nodes:
            return
        root_node = doc_nodes[0].root_node
        keys_to_delete = []
        for k in root_node.global_metadata:
            if not (k in RAG_SYSTEM_META_KEYS or k in metadata):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            root_node.global_metadata.pop(k)
        root_node.global_metadata.update(metadata)

    @override
    def remove_nodes(self, group_name: Optional[str] = None, doc_ids: Optional[Set[str]] = None,
                     uids: Optional[List[str]] = None) -> None:
        if uids:
            need_delete = uids
        elif doc_ids:
            doc_id_set = set(doc_ids)
            if group_name:
                candidates = self._group2uids.get(group_name, [])
                need_delete = [uid for uid, node in candidates if node.global_metadata.get(RAG_DOC_ID) in doc_id_set]
            else:
                need_delete = [uid for uid, node in self._uid2node.items()
                               if node.global_metadata.get(RAG_DOC_ID) in doc_id_set]
        else:
            return

        for index in self._name2index.values():
            index.remove(need_delete)

        for uid in need_delete:
            node = self._uid2node.pop(uid, None)
            if node:
                self._group2uids.get(node._group, set()).discard(uid)

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,
                  doc_ids: Optional[Set] = None, **kwargs) -> List[DocNode]:
        if uids:
            return [self._uid2node[uid] for uid in uids]
        elif group_name:
            uids = self._group2uids.get(group_name, set())
            if not doc_ids:
                return [self._uid2node[uid] for uid in uids]
            else:
                return [self._uid2node[uid] for uid in uids
                        if self._uid2node[uid].global_metadata.get(RAG_DOC_ID) in doc_ids]

    @override
    def is_group_active(self, name: str) -> bool:
        uids = self._group2uids.get(name)
        return True if uids else False

    @override
    def all_groups(self) -> List[str]:
        return list(self._group2uids.keys())

    @override
    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        if isinstance(group_names, str): group_names = [group_names]
        self._activated_groups.update(group_names)

    @override
    def activated_groups(self):
        return list(self._activated_groups)

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        return self.get_index('default').query(*args, **kwargs)

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        if type is None:
            type = 'default'
        return self._name2index.get(type)

    @override
    def clear_cache(self, group_names: Optional[List[str]]) -> None:
        if group_names is None:
            group_names = self.all_groups()
        elif isinstance(group_names, str):
            group_names = [group_names]
        elif isinstance(group_names, (tuple, list, set)):
            group_names = list(group_names)
        else:
            raise TypeError(f"Invalid type {type(group_names)} for group_names, expected list of str")
        for group_name in group_names:
            uids = list(self._group2uids.get(group_name))
            self.remove_nodes(uids=uids)
