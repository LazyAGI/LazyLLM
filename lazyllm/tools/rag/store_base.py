from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .doc_node import DocNode
from .index_base import IndexBase

class StoreBase(ABC):
    def __init__(self):
        self._name2index = {}

    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    def get_index(self, type: str) -> Optional[IndexBase]:
        return self._name2index.get(type)

    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._update_nodes(nodes)
        self._update_indices(self._name2index, nodes)

    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        self._remove_nodes(group_name, uids)
        self._remove_from_indices(self._name2index, uids, group_name)

    @abstractmethod
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        pass

    @abstractmethod
    def is_group_active(self, name: str) -> bool:
        pass

    @abstractmethod
    def all_groups(self) -> List[str]:
        pass

    @abstractmethod
    def _update_nodes(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def _remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        pass

    @staticmethod
    def _update_indices(name2index: Dict[str, BaseIndex], nodes: List[DocNode]) -> None:
        for _, index in name2index.items():
            index.update(nodes)

    @staticmethod
    def _remove_from_indices(name2index: Dict[str, BaseIndex], uids: List[str],
                             group_name: Optional[str] = None) -> None:
        for _, index in name2index.items():
            index.remove(uids, group_name)
