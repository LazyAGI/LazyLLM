from typing import Dict, List, Optional, Callable, Union, Set
from .index_base import IndexBase
from .store_base import StoreBase
from .doc_node import DocNode
from .utils import _FileNodeIndex
from .default_index import DefaultIndex
from lazyllm.common import override

def _update_indices(name2index: Dict[str, IndexBase], nodes: List[DocNode]) -> None:
    for index in name2index.values():
        index.update(nodes)

def _remove_from_indices(name2index: Dict[str, IndexBase], uids: List[str],
                         group_name: Optional[str] = None) -> None:
    for index in name2index.values():
        index.remove(uids, group_name)

class MapStore(StoreBase):
    def __init__(self, node_groups: Union[List[str], Set[str]], embed: Dict[str, Callable], **kwargs):
        # Dict[group_name, Dict[uuid, DocNode]]
        self._group2docs: Dict[str, Dict[str, DocNode]] = {
            group: {} for group in node_groups
        }

        self._name2index = {
            'default': DefaultIndex(embed, self),
            'file_node_map': _FileNodeIndex(),
        }

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._group2docs[node.group][node.uid] = node
        _update_indices(self._name2index, nodes)

    @override
    def remove_nodes(self, group_name: str, uids: List[str] = None) -> None:
        if uids:
            docs = self._group2docs.get(group_name)
            if docs:
                _remove_from_indices(self._name2index, uids)
                for uid in uids:
                    docs.pop(uid, None)
        else:
            docs = self._group2docs.pop(group_name, None)
            if docs:
                _remove_from_indices(self._name2index, list(docs.keys()))

    @override
    def get_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        docs = self._group2docs.get(group_name)
        if not docs:
            return []

        if uids is None:
            return list(docs.values())

        if len(uids) == 0:
            return []

        ret = []
        for uid in uids:
            doc = docs.get(uid)
            if doc:
                ret.append(doc)
        return ret

    @override
    def is_group_active(self, name: str) -> bool:
        docs = self._group2docs.get(name)
        return True if docs else False

    @override
    def all_groups(self) -> List[str]:
        return self._group2docs.keys()

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
