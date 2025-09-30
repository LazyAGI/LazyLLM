import lazyllm
from .store_base import (
    LazyLLMStoreBase,
    LAZY_IMAGE_GROUP,
    LAZY_ROOT_NAME,
    EMBED_DEFAULT_KEY,
    BUILDIN_GLOBAL_META_DESC
)
from .hybrid import HybridStore, MapStore, SenseCoreStore
from .segment import OpenSearchStore, ElasticSearchStore
from .vector import ChromaStore, MilvusStore

_STORE_REGISTRY = {
    "chroma": ChromaStore,
    "chromadb": ChromaStore,
    "milvus": MilvusStore,
    "opensearch": OpenSearchStore,
    "hybrid": HybridStore,
    "map": MapStore,
    "sensecore": SenseCoreStore,
    "elasticsearch": ElasticSearchStore,
}

def get_store_class(name: str):
    cls = _STORE_REGISTRY.get(name.lower())
    if cls:
        return cls

    if hasattr(lazyllm, 'store') and hasattr(lazyllm.store, name):
        return getattr(lazyllm.store, name)

    return None

__all__ = [
    'LazyLLMStoreBase',
    'HybridStore',
    'MapStore',
    'OpenSearchStore',
    'ElasticSearchStore',
    'ChromaStore',
    'MilvusStore',
    'SenseCoreStore',
    'LAZY_IMAGE_GROUP',
    'LAZY_ROOT_NAME',
    'EMBED_DEFAULT_KEY',
    'BUILDIN_GLOBAL_META_DESC'
]
