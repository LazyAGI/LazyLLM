from typing import List, Optional
from .doc_node import DocNode
from .store_base import StoreBase
from .index_base import IndexBase
from lazyllm.common import override

class WrapStoreToIndex(IndexBase):
    def __init__(self, store: StoreBase):
        self._store = store

    @override
    def update(self, nodes: List[DocNode]) -> None:
        self._store.update_nodes(nodes)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        self._store.remove_nodes(group_name, uids)

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        return self._store.query(*args, **kwargs)
