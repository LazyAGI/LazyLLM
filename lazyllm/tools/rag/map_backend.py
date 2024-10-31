from typing import Dict, List, Optional
from .index_base import IndexBase
from .store_base import StoreBase
from .doc_node import DocNode
from lazyllm.common import override

def _update_indices(name2index: Dict[str, IndexBase], nodes: List[DocNode]) -> None:
    for _, index in name2index.items():
        index.update(nodes)

def _remove_from_indices(name2index: Dict[str, IndexBase], uids: List[str],
                         group_name: Optional[str] = None) -> None:
    for _, index in name2index.items():
        index.remove(uids, group_name)

class MapBackend:
    def __init__(self, node_groups: List[str]):
        super().__init__()
        # Dict[group_name, Dict[uuid, DocNode]]
        self._group2docs: Dict[str, Dict[str, DocNode]] = {
            group: {} for group in node_groups
        }
        self._name2index = {}

    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._group2docs[node.group][node.uid] = node
        _update_indices(self._name2index, nodes)

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
                _remove_from_indices(self._name2index, [doc.uid for doc in docs])

    def get_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        docs = self._group2docs.get(group_name)
        if not docs:
            return []

        if not uids:
            return list(docs.values())

        ret = []
        for uid in uids:
            doc = docs.get(uid)
            if doc:
                ret.append(doc)
        return ret

    def is_group_active(self, name: str) -> bool:
        docs = self._group2docs.get(name)
        return True if docs else False

    def all_groups(self) -> List[str]:
        return self._group2docs.keys()

    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    def get_index(self, type: str = 'default') -> Optional[IndexBase]:
        if type != 'default':
            return self._name2index.get(type)
        return self

    def update(self, nodes: List[DocNode]) -> None:
        self.update_nodes(nodes)

    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        if group_name:
            self.remove_nodes(group_name, uids)
        else:
            for _, docs in self._group2docs.items():
                for uid in uids:
                    docs.pop(uid, None)
        _remove_from_indices(self._name2index, uids)

    def query(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self.get_nodes(group_name, uids)

    def find_node_by_uid(self, uid: str) -> Optional[DocNode]:
        for docs in self._group2docs.values():
            doc = docs.get(uid)
            if doc:
                return doc
        return None


class _MapIndex(IndexBase):
    def __init__(self, backend: MapBackend):
        self._backend = backend

    @override
    def update(self, nodes: List[DocNode]) -> None:
        self._backend.update(nodes)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        self._backend.remove(uids, group_name)

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        return self._backend.query(*args, **kwargs)


class MapStore(StoreBase):
    def __init__(self, node_groups: List[str]):
        self._backend = MapBackend(node_groups)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._backend.update_nodes(nodes)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        self._backend.remove_nodes(group_name, uids)

    @override
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self._backend.get_nodes(group_name, uids)

    @override
    def is_group_active(self, name: str) -> bool:
        return self._backend.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        return self._backend.all_groups()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._backend.register_index(type, index)

    @override
    def get_index(self, type: str = 'default') -> Optional[IndexBase]:
        if type == 'default':
            return _MapIndex(self._backend)
        return self._backend.get_index(type)
