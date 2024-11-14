from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List, Optional

class IndexBase(ABC):
    @abstractmethod
    def update(self, nodes: List[DocNode]) -> None:
        pass

    @abstractmethod
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        pass
