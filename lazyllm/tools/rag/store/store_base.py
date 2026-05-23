import re

from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Optional, List, Union, Set, Dict, Any, Tuple
from lazyllm.common import LazyLLMRegisterMetaABCClass
from pydantic import BaseModel, Field

from ..data_type import DataType
from ..utils import is_sparse
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
    TEXT = auto()
    IMAGE = auto()
    HYBRID = auto()
    TABLE = auto()
    CODE = auto()
    QA = auto()
    JSON = auto()
    RICH = auto()


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
    copy_source: Optional[Dict[str, str]] = Field(default_factory=dict)


class StoreCapability(IntFlag):
    SEGMENT = auto()
    VECTOR = auto()
    ALL = SEGMENT | VECTOR


class EmbedResolveMixin:
    '''Mixin for stores that lazily resolve embed dims/datatypes when creating a new collection.
    Requires the host class to have _embed, _embed_dims, _embed_datatypes attributes.'''

    def _resolve_missing_embed_specs(self, embed_keys: Set[str]) -> None:
        # Must be called inside _ddl_lock to be thread-safe.
        for k in embed_keys:
            if k in self._embed_datatypes:
                continue
            if not self._embed or k not in self._embed:
                raise ValueError(f'Cannot resolve embed specs for key {k!r}: embed function not available')
            embedding = self._embed[k]('a')
            if is_sparse(embedding):
                self._embed_datatypes[k] = DataType.SPARSE_FLOAT_VECTOR
            else:
                self._embed_dims[k] = len(embedding)
                self._embed_datatypes[k] = DataType.FLOAT_VECTOR


class LazyLLMStoreBase(ABC, metaclass=LazyLLMRegisterMetaABCClass):
    capability: StoreCapability
    need_embedding: bool = True
    supports_index_registration: bool = False

    @property
    def dir(self):
        raise NotImplementedError

    @abstractmethod
    def connect(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def search(self, collection_name: str, query: Optional[str] = None,
               query_embedding: Optional[Union[dict, List[float]]] = None, topk: int = 10,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        raise NotImplementedError

    def seg_connect(self, *args, **kwargs):
        # For pure SEGMENT stores: seg_connect == connect.
        # For pure VECTOR stores: seg_connect is a no-op.
        if self.capability & StoreCapability.SEGMENT:
            self.connect(*args, **kwargs)

    def vec_connect(self, *args, **kwargs):
        # For pure VECTOR stores: vec_connect == connect.
        # For pure SEGMENT stores: vec_connect is a no-op.
        if self.capability & StoreCapability.VECTOR:
            self.connect(*args, **kwargs)

    def try_read_dims_from_schema(self, collections: List[str]) -> Tuple[Dict[str, int], Dict[str, DataType]]:
        '''Try to read embed_dims and embed_datatypes from existing backend schema.

        Default: no introspection supported. Subclasses override when the vector backend
        can describe collections before connect().
        '''
        return {}, {}
