from collections import defaultdict
from typing import Dict, List, Optional, Callable, Union, Set

from .store_base import StoreBase, LAZY_ROOT_NAME

from ..index_base import IndexBase
from ..doc_node import DocNode
from ..default_index import DefaultIndex
from ..global_metadata import RAG_SYSTEM_META_KEYS, RAG_DOC_ID

from lazyllm.common import override


class MapStore(StoreBase):
    def __init__(self, node_groups: Union[List[str], Set[str]], embed: Dict[str, Callable], **kwargs):
        self._uid2node: Dict[str, DocNode] = {}
        self._group2uids: Dict[str, Set[str]] = {group: set() for group in node_groups}
        self._docid2uids: Dict[str, Set[str]] = defaultdict(set)
        self._group_doc_uids: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._name2index = {
            'default': DefaultIndex(embed, self),
        }
        self._activated_groups = set()

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            group = node._group
            doc_id = node.global_metadata.get(RAG_DOC_ID)
            uid = node._uid
            self._group2uids[group].add(uid)
            self._docid2uids[doc_id].add(uid)
            self._group_doc_uids[group][doc_id].add(uid)
            self._uid2node[node._uid] = node

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
    def remove_nodes(self, doc_ids: List[str], group_name: Optional[str] = None,
                     uids: Optional[List[str]] = None) -> None:
        if uids:
            need_delete = uids
        elif group_name:
            need_delete = [uid for doc_id in doc_ids
                           for uid in self._group_doc_uids.get(group_name, {}).get(doc_id, ())]
        else:
            need_delete = [uid for doc_id in doc_ids for uid in self._docid2uids.get(doc_id, ())]

        for index in self._name2index.values():
            index.remove(need_delete)

        for uid in need_delete:
            node = self._uid2node.pop(uid, None)
            if not node:
                continue
            group = node._group
            doc_id = node.global_metadata.get(RAG_DOC_ID)
            self._group2uids.get(group, set()).discard(uid)
            self._docid2uids.get(doc_id, set()).discard(uid)
            self._group_doc_uids.get(group, {}).get(doc_id, set()).discard(uid)
            if group in self._group_doc_uids and not self._group_doc_uids[group]:
                self._group_doc_uids.pop(group)

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,
                  doc_ids: Optional[Set] = None, **kwargs) -> List[DocNode]:
        if uids:
            return [self._uid2node[uid] for uid in uids if uid in self._uid2node]
        elif doc_ids and group_name:
            uids = [uid for doc_id in doc_ids
                    for uid in self._group_doc_uids.get(group_name, {}).get(doc_id, ())]
        elif group_name:
            uids = self._group2uids.get(group_name, set())
        elif doc_ids:
            uids = [uid for doc_id in doc_ids for uid in self._docid2uids.get(doc_id, ())]
        else:
            return []
        return [self._uid2node[uid] for uid in uids if uid in self._uid2node]

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
    def clear_cache(self, group_names: Optional[List[str]] = None) -> None:
        if group_names is None:
            self._docid2uids.clear()
            self._group_doc_uids.clear()
            self._uid2node.clear()
            for group in self._group2uids.keys():
                self._group2uids[group].clear()
            return
        elif isinstance(group_names, str):
            group_names = [group_names]
        elif isinstance(group_names, (tuple, list, set)):
            group_names = list(group_names)
        else:
            raise TypeError(f"Invalid type {type(group_names)} for group_names, expected list of str")
        for group_name in group_names:
            uids = self._group2uids.get(group_name, set())
            self.remove_nodes(doc_ids=[], uids=uids)
