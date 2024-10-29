from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .doc_node import DocNode
from .index_base import IndexBase

class StoreBase(ABC):
    def __init__(self):
        self._name2index = {}

    def register_index(self, type_name: str, index: IndexBase) -> None:
        self._name2index[type_name] = index

    def get_index(self, type_name: str) -> Optional[IndexBase]:
        return self._name2index.get(type_name)

    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._update_nodes(nodes)
        for _, index in self._name2index.items():
            index.update(nodes)

    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        self._remove_nodes(group_name, uids)
        for _, index in self._name2index.items():
            index.remove(uids, group_name)

    @abstractmethod
    def _update_nodes(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def _remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        pass

    @abstractmethod
    def group_is_active(self, group_name: str) -> bool:
        pass

    @abstractmethod
    def all_groups(self) -> List[str]:
        pass
