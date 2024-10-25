from .doc_node import DocNode
from abc import ABC, abstractmethod
from typing import List, Optional

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
    def remove(uids: List[str], group_name: Optional[str] = None) -> None:
        '''
        Removes `DocNode`s sepcified by `uids`. If `group_name` is not None,
        just remove uids from that group.

        Args:
            uids (List[str]): a list of doc ids.
            group_name (Optional[str]): name of the group.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")
