from abc import ABC, abstractmethod
from typing import Optional, List
from .doc_node import DocNode
from .index_base import IndexBase

# ---------------------------------------------------------------------------- #

LAZY_ROOT_NAME = "lazyllm_root"
EMBED_DEFAULT_KEY = '__default__'

# ---------------------------------------------------------------------------- #

class StoreBase(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        pass

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
    def query(self, *args, **kwargs) -> List[DocNode]:
        pass

    @abstractmethod
    def register_index(self, type: str, index: IndexBase) -> None:
        pass

    @abstractmethod
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        pass
