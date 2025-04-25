from abc import ABC, abstractmethod
from typing import Optional, List
from .doc_node import DocNode
from .index_base import IndexBase

# ---------------------------------------------------------------------------- #

LAZY_ROOT_NAME = "lazyllm_root"
LAZY_IMAGE_GROUP = "Image"
EMBED_DEFAULT_KEY = '__default__'

# ---------------------------------------------------------------------------- #

class StoreBase(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
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
            self.remove_nodes(group_name, None)
