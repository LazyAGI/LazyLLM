import re

from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Optional, List, Union, Set, Dict, Any
from lazyllm import LazyLLMRegisterMetaABCClass
from pydantic import BaseModel, Field

from ..data_type import DataType
from ..global_metadata import (
    GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_PATH, RAG_DOC_FILE_NAME,
    RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE,
    RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE, RAG_KB_ID
)

LAZY_ROOT_NAME = 'lazyllm_root'
LAZY_IMAGE_GROUP = 'image'
EMBED_DEFAULT_KEY = '__default__'
EMBED_PREFIX = 'embedding_'
DEFAULT_KB_ID = 'default'
GLOBAL_META_KEY_PREFIX = 'global_meta_'

BUILDIN_GLOBAL_META_DESC = {
    RAG_DOC_ID: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=512),
    RAG_KB_ID: GlobalMetadataDesc(data_type=DataType.VARCHAR, default_value=' ', max_size=512),
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
    """An enumeration."""
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
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)
    global_meta: Optional[Dict[str, Any]] = Field(default_factory=dict)
    embedding: Optional[Dict[str, List[float]]] = Field(default_factory=dict)
    type: Optional[int] = SegmentType.TEXT.value
    number: Optional[int] = 0
    kb_id: Optional[str] = '__default__'
    excluded_embed_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    excluded_llm_metadata_keys: Optional[List[str]] = Field(default_factory=list)
    parent: Optional[str] = None    # uid of parent node
    answer: Optional[str] = ''
    image_keys: Optional[List[str]] = Field(default_factory=list)


class StoreCapability(IntFlag):
    """An enumeration."""
    SEGMENT = auto()
    VECTOR = auto()
    ALL = SEGMENT | VECTOR


class LazyLLMStoreBase(ABC, metaclass=LazyLLMRegisterMetaABCClass):
    """Base class for vector storage, defining the common interface specification. 
All concrete storage implementations (e.g., ChromaDB, Milvus) must inherit and implement this class.
"""
    capability: StoreCapability
    need_embedding: bool = True
    supports_index_registration: bool = False

    @property
    def dir(self):
        raise NotImplementedError

    @abstractmethod
    def connect(self, *args, **kwargs):
        """Establish connection to the storage backend.

Args:
    *args: Variable positional arguments.
    **kwargs: Variable keyword arguments.
"""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """Insert or update data in a collection.

Args:
    collection_name (str): The collection name.
    data (List[dict]): List of records to upsert.
"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        """Delete data from a collection.

Args:
    collection_name (str): The collection name.
    criteria (dict): Conditions for deletion.
    **kwargs: Additional parameters.
"""
        raise NotImplementedError

    @abstractmethod
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        """Retrieve data from a collection by criteria.

Args:
    collection_name (str): The collection name.
    criteria (dict): Filter conditions.
    **kwargs: Additional parameters.
"""
        raise NotImplementedError

    @abstractmethod
    def search(self, collection_name: str, query: Optional[str] = None,
               query_embedding: Optional[Union[dict, List[float]]] = None, topk: int = 10,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        """Perform a search operation, supporting both text and vector queries.

Args:
    collection_name (str): The collection name.
    query (Optional[str]): Text query string.
    query_embedding (Optional[Union[dict, List[float]]]): Query vector.
    topk (int): Number of results to return. Defaults to 10.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Metadata filter conditions.
    embed_key (Optional[str]): Embedding key.
    **kwargs: Additional parameters.
"""
        raise NotImplementedError
