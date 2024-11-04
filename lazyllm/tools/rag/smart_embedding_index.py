from typing import List, Optional
from .doc_node import DocNode
from .index_base import IndexBase
from lazyllm.common import override
from .map_store import MapStore
from .milvus_store import MilvusStore

class SmartEmbeddingIndex(IndexBase):
    def __init__(self, backend_type: str, **kwargs):
        if backend_type == 'milvus':
            self._store = MilvusStore(**kwargs)
        elif backend_type == 'map':
            self._store = MapStore(**kwargs)
        else:
            raise ValueError(f'unsupported backend [{backend_type}]')

    @override
    def update(self, nodes: List[DocNode]) -> None:
        self._store.update_nodes(nodes)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        self._store.remove_nodes(group_name, uids)

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        return self._store.query(*args, **kwargs)
