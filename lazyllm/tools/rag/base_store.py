from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from .doc_node import DocNode
from .base_index import BaseIndex

class BaseStore(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        '''
        Inserts or updates a list of `DocNode` to this store.

        Args:
            nodes (List[DocNode]): nodes to be inserted or updated.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_group_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        '''
        Returns a list of `DocNode` specified by `uids` in the group named `group_name`.
        All `DocNode`s in the group `group_name` will be returned if `uids` is `None` or `[]`.

        Args:
            group_name (str): the name of group.
            uids (List[str]): a list of doc ids.

        Returns:
            List[DocNode]: the result.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove_group_nodes(self, group_name: str, uids: List[str] = None) -> None:
        '''
        Removes sepcified `DocNode`s in the group named `group_name`.
        Group `group_name` will be removed if `uids` is `None` or `[]`.

        Args:
            group_name (str): the name of group.
            uids (List[str]): a list of doc ids.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def group_is_active(self, group_name: str) -> bool:
        '''
        Returns `True` if a group named `group_name` exists or has at least one `DocNode`.

        Args:
            group_name (str): the name of group.

        Returns:
            bool: whether the group `group_name` is active.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def group_names(self) -> List[str]:
        '''
        Returns group names in this store.

        Returns:
            List[str]: the result.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def register_index(self, type: str, index: BaseIndex) -> None:
        '''
        Registers `index` with type `type` to this store.

        Args:
            type (str): type of the index to be registered.
            index (BaseIndex): the index to be registered.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove_index(self, type: str) -> None:
        '''
        Removes index with type `type` in this store.

        Args:
            type (str): type of the index to be removed.
        '''
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_index(self, type: str) -> Optional[BaseIndex]:
        '''
        Returns index with the specified type `type` in this store.

        Args:
            type (str): type of the index to be removed.

        Returns:
            Optional[BaseIndex]: the index of specified type, or `None`.
        '''
        raise NotImplementedError("not implemented yet.")

    # ----- helper functions ----- #

    @staticmethod
    def _update_indices(name2index: Dict[str, BaseIndex], nodes: List[DocNode]) -> None:
        for _, index in name2index.items():
            index.update(nodes)

    @staticmethod
    def _remove_from_indices(name2index: Dict[str, BaseIndex], uids: List[str],
                             group_name: Optional[str] = None) -> None:
        for _, index in name2index.items():
            index.remove(uids, group_name)
