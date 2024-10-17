from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .doc_node import DocNode
from .base_index import BaseIndex

class BaseStore(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_node(self, group_name: str, node_id: str) -> Optional[DocNode]:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_nodes(self, group_name: str) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove_nodes(self, uids: List[str]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def has_nodes(self, group_name: str) -> bool:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def all_groups(self) -> List[str]:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def register_index(self, type: str, index: BaseIndex) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove_index(self, type: str) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_index(self, type: str) -> Optional[BaseIndex]:
        raise NotImplementedError("not implemented yet.")

    # ----- helper functions ----- #

    @staticmethod
    def _update_indices(name2index: Dict[str, BaseIndex], nodes: List[DocNode]) -> None:
        for _, index in name2index.items():
            index.update(nodes)

    @staticmethod
    def _remove_from_indices(name2index: Dict[str, BaseIndex], uids: List[str]) -> None:
        for _, index in name2index.items():
            index.remove(uids)
