from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List

class BaseIndex(ABC):
    @abstractmethod
    def update(nodes: List[DocNode]) -> None:
        '''
        Inserts or updates a list of `DocNode` to this index.

        Args:
            nodes (List[DocNode]): nodes to be inserted or updated.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove(uids: List[str]) -> None:
        '''
        Removes `DocNode`s sepcified by `uids` in the group named `group_name`.

        Args:
            uids (List[str]): a list of doc ids.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")
