import re

from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Optional, List, Union, Set, Dict
from lazyllm import LazyLLMRegisterMetaClass, once_wrapper
from pydantic import BaseModel, Field

from ..doc_node import DocNode
from ..index_base import IndexBase
from ..data_type import DataType
from ..global_metadata import (
    GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_PATH, RAG_DOC_FILE_NAME,
    RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE,
    RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE
)

LAZY_ROOT_NAME = "lazyllm_root"
LAZY_IMAGE_GROUP = "image"
EMBED_DEFAULT_KEY = '__default__'
EMBED_PREFIX = "embedding_"
DEFAULT_KB_ID = "default"
GLOBAL_META_KEY_PREFIX = "global_meta_"

BUILDIN_GLOBAL_META_DESC = {
    RAG_DOC_ID: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=512),
    RAG_DOC_PATH: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=65535),
    RAG_DOC_FILE_NAME: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=65535),
    RAG_DOC_FILE_TYPE: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=65535),
    RAG_DOC_FILE_SIZE: GlobalMetadataDesc(data_type=DataType.INT32, default_value=0),
    RAG_DOC_CREATION_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=10),
    RAG_DOC_LAST_MODIFIED_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=10),
    RAG_DOC_LAST_ACCESSED_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=10)
}
INSERT_BATCH_SIZE = 3000
IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


class SegmentType(IntFlag):
    TEXT = auto()
    IMAGE = auto()
    HYBRID = auto()
    TABLE = auto()
    CODE = auto()
    QA = auto()


class Segment(BaseModel):
    uid: str
    doc_id: str
    group: str
    content: str
    meta: str   # json string
    global_meta: str   # json string
    embedding: Optional[Dict[str, List[float]]] = Field(default_factory=dict)
    type: Optional[SegmentType] = SegmentType.TEXT
    number: Optional[int] = 0
    kb_id: Optional[str] = "__default__"
    excluded_embed_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    excluded_llm_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    parent: Optional[str] = None    # uid of parent node
    answer: Optional[str] = None
    image_keys: Optional[List[str]] = Field(default_factory=list)


class StoreCapability(IntFlag):
    SEGMENT = auto()
    VECTOR = auto()
    BOTH = SEGMENT | VECTOR
    ALL = SEGMENT | VECTOR | auto()


class StoreBase(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """ update nodes to the store """
        raise NotImplementedError

    @abstractmethod
    def remove_nodes(self, doc_ids: List[str], group_name: Optional[str] = None,
                     uids: Optional[List[str]] = None) -> None:
        """ remove nodes from the store by doc_ids or uids """
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
    def clear_cache(self, group_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,
                  doc_ids: Optional[Set] = None, **kwargs) -> List[DocNode]:
        """ get nodes from the store """
        raise NotImplementedError

    @abstractmethod
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        """ update doc meta """
        raise NotImplementedError

    @abstractmethod
    def query(self, *args, **kwargs) -> List[DocNode]:
        """ search nodes from the store """
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


class LazyLLMStoreBase(ABC, metaclass=LazyLLMRegisterMetaClass):
    capability: StoreCapability

    def __init_subclass__(cls, *, capability: StoreCapability, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.capability = capability

    @once_wrapper(reset_on_pickle=True)
    @abstractmethod
    def lazy_init(self, *args, **kwargs):
        """ lazy init """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """ upsert data to the store """
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        """ delete data from the store """
        raise NotImplementedError

    @abstractmethod
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        """ get data from the store """
        raise NotImplementedError

    @abstractmethod
    def search(self, collection_name: str, query: Union[str, dict, List[float]], topk: int,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        """ search data from the store, search result: [{"uid": str, "score": float}] """
        raise NotImplementedError
