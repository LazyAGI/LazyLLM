from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Set

from lazyllm import LOG
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.index_base import IndexBase
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import (
    GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_PATH, RAG_DOC_FILE_NAME,
    RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE,
    RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE
)

LAZY_ROOT_NAME = "lazyllm_root"
LAZY_IMAGE_GROUP = "image"
EMBED_DEFAULT_KEY = '__default__'
BUILDIN_GLOBAL_META_DESC = {
    RAG_DOC_ID: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=512
    ),
    RAG_DOC_PATH: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=65535
    ),
    RAG_DOC_FILE_NAME: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=128
    ),
    RAG_DOC_FILE_TYPE: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=128
    ),
    RAG_DOC_FILE_SIZE: GlobalMetadataDesc(
        data_type=DataType.INT32,
        default_value=0
    ),
    RAG_DOC_CREATION_DATE: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=10
    ),
    RAG_DOC_LAST_MODIFIED_DATE: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=10
    ),
    RAG_DOC_LAST_ACCESSED_DATE: GlobalMetadataDesc(
        data_type=DataType.VARCHAR,
        default_value=' ', max_size=10
    )
}


class StoreBase(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """ update nodes to the store """
        raise NotImplementedError

    @abstractmethod
    def remove_nodes(
        self,
        doc_ids: Optional[List[str]] = None,
        uids: Optional[List[str]] = None
    ) -> None:
        """ remove nodes from the store by doc_ids or uids """
        raise NotImplementedError

    @abstractmethod
    def get_nodes(
        self,
        group_name: Optional[str] = None,
        uids: Optional[List[str]] = None,
        doc_ids: Optional[Set] = None
    ) -> List[DocNode]:
        """ get nodes from the store """
        raise NotImplementedError

    @abstractmethod
    def register_index(self, type: str, index: IndexBase) -> None:
        """ register index to the store (for store that support hook only)"""
        raise NotImplementedError

    @abstractmethod
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """ get registered index from the store """
        raise NotImplementedError

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        """ search nodes from the store """
        raise NotImplementedError

    @abstractmethod
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
        """ update doc meta """
        raise NotImplementedError

    @abstractmethod
    def all_groups(self) -> List[str]:
        """ get all node groups for Document """
        raise NotImplementedError

    @abstractmethod
    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def activated_groups(self):
        raise NotImplementedError

    @abstractmethod
    def is_group_active(self, name: str) -> bool:
        """ check if a group has nodes (active) """
        raise NotImplementedError

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


class StoreBaseMixin:
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """ update nodes to the store """
        raise NotImplementedError

    @abstractmethod
    def remove_nodes(
        self,
        doc_ids: Optional[List[str]] = None,
        uids: Optional[List[str]] = None
    ) -> None:
        """ remove nodes from the store by doc_ids or uids """
        raise NotImplementedError

    @abstractmethod
    def _serialize_node_for_store(self, node: DocNode):
        """ serialize node to a dict that can be stored in vector store """
        raise NotImplementedError

    @abstractmethod
    def register_index(self, type: str, index: IndexBase) -> None:
        """ register index to the store (for store that support hook only)"""
        raise NotImplementedError

    @abstractmethod
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """ get registered index from the store """
        raise NotImplementedError

    @abstractmethod
    def clear_cache(self, group_names: Optional[List[str]]) -> bool:
        raise NotImplementedError


class SegmentStoreBase(StoreBaseMixin, ABC):
    @abstractmethod
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None) -> List[DocNode]:
        """ get nodes from the store """
        # NOTE vector db不一定需要get nodes
        raise NotImplementedError

    @abstractmethod
    def _deserialize_node(self, segment: Dict) -> DocNode:
        raise NotImplementedError


class VectorStoreBase(StoreBaseMixin, ABC):
    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        """ search nodes from the store """
        raise NotImplementedError


class DocStoreBase(ABC):
    def __init__(
        self,
        kb_id: str = "__default__",
        segment_store: SegmentStoreBase = None,
        vector_store: VectorStoreBase = None,
        uri: str = "",
    ):
        # uri or (segment_store + vector_store)
        if uri:
            self._uri = uri
            if self._connect_store(uri):
                LOG.info(f"Connected to doc store {self._uri}")
            else:
                raise ConnectionError(f"Failed to connect to doc store {self._uri}")
        elif segment_store and vector_store:
            self._segment_store = segment_store
            self._vector_store = vector_store
        else:
            raise ValueError("Either uri or (segment_store, vector_store) must be provided")
        self._kb_id = kb_id
        self._activated_groups: Set = {LAZY_ROOT_NAME, LAZY_IMAGE_GROUP}

    @abstractmethod
    def _connect_store(self, uri: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """ update nodes to the store """
        raise NotImplementedError

    @abstractmethod
    def remove_nodes(
        self,
        group_name: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        uids: Optional[List[str]] = None
    ) -> None:
        """ remove nodes from the store by doc_ids or uids """
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None) -> List[DocNode]:
        """ get nodes from the store """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query: str,
        group_name: str,
        topk: int = 10,
        embed_keys: Optional[List[str]] = None,
        **kwargs
    ) -> List[DocNode]:
        """ search nodes from the store """
        raise NotImplementedError

    @abstractmethod
    def register_index(self, type: str, index: IndexBase) -> None:
        """ register index to the store (for store that support hook only)"""
        raise NotImplementedError

    @abstractmethod
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """ get registered index from the store """
        raise NotImplementedError

    @abstractmethod
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
        """ update doc meta """
        raise NotImplementedError

    @property
    def all_groups(self) -> List[str]:
        """ get all node groups for Document """
        return list(self._activated_groups)

    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        if isinstance(group_names, str): group_names = [group_names]
        self._activated_groups.update(group_names)

    def activated_groups(self):
        return list(self._activated_groups)

    @abstractmethod
    def is_group_active(self, name: str) -> bool:
        """ check if a group has nodes (active) """
        raise NotImplementedError
