from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .doc_node import DocNode
from .index_base import IndexBase

class StoreBase(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        pass

    @abstractmethod
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def group_is_active(self, group_name: str) -> bool:
        pass

    @abstractmethod
    def group_names(self) -> List[str]:
        pass

    @abstractmethod
    def register_index(self, type_name: str, index: IndexBase) -> None:
        pass

    @abstractmethod
    def remove_index(self, type_name: str) -> None:
        pass

    @abstractmethod
    def get_index(self, type_name: str) -> Optional[IndexBase]:
        pass

    # ----- helper functions ----- #

    @staticmethod
    def _update_indices(name2index: Dict[str, IndexBase], nodes: List[DocNode]) -> None:
        for _, index in name2index.items():
            index.update(nodes)

    @staticmethod
    def _remove_from_indices(name2index: Dict[str, IndexBase], uids: List[str],
                             group_name: Optional[str] = None) -> None:
        for _, index in name2index.items():
            index.remove(uids, group_name)
