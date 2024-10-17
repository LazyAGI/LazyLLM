from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List

class BaseIndex(ABC):
    @abstractmethod
    def update(nodes: List[DocNode]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove(uids: List[str]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")
