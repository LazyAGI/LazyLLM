from collections import defaultdict
from typing import Dict, List, Optional, Union, Set

from .store_base import LazyLLMStoreBase, StoreCapability, DEFAULT_KB_ID

from lazyllm.common import override


class MapStore(LazyLLMStoreBase, capability=StoreCapability.ALL):

    def __init__(self, uri: Optional[str] = None):
        self._uri = uri  # for persistence storage using sqlite

    def _load_from_uri(self, uri: str):
        pass

    def _save_to_uri(self, uri: str):
        pass

    @override
    def lazy_init(self, *args, **kwargs):
        self._uid2data: Dict[str, dict] = {}
        self._collection2uids: Dict[str, Set[str]] = defaultdict(set)
        self._docid2uids: Dict[str, Set[str]] = defaultdict(set)
        self._col_kb_doc_uids: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))

        if self._uri:
            self._load_from_uri(self._uri)
        return

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> None:
        for item in data:
            uid = item.get('uid')
            doc_id = item.get('doc_id')
            assert uid and doc_id, "[MapStore - upsert] uid and doc_id are required"
            self._uid2data[uid] = item
            self._collection2uids[collection_name].add(uid)
            self._col_kb_doc_uids[collection_name][item.get('kb_id', DEFAULT_KB_ID)][doc_id].add(uid)
            self._docid2uids[doc_id].add(uid)

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        need_delete = self._get_uids_by_criteria(collection_name, criteria)
        if not need_delete:
            return False
        for uid in need_delete:
            data = self._uid2data.pop(uid, None)
            if not data:
                continue
            self._collection2uids[collection_name].remove(uid)
            self._col_kb_doc_uids[collection_name][data.get('kb_id', DEFAULT_KB_ID)][data.get('doc_id')].remove(uid)
            self._docid2uids[data.get('doc_id')].remove(uid)
        return True

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        uids = self._get_uids_by_criteria(collection_name, criteria)
        return [self._uid2data[uid] for uid in uids]

    def _get_uids_by_criteria(self, collection_name: str, criteria: dict) -> List[str]:
        if not criteria:
            return list(self._collection2uids.get(collection_name, set()))
        else:
            uids = criteria.get('uids', [])
            kb_id = criteria.get('kb_id')
            doc_ids = criteria.get('doc_ids', [])
            if uids:
                return uids
            elif kb_id:
                return [uid for doc_id in doc_ids
                        for uid in self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).get(doc_id, ())]
            else:
                return [uid for doc_id in doc_ids for uid in self._docid2uids.get(doc_id, ())]

    @override
    def search(self, collection_name: str, query: str, topk: int,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[dict]:
        raise NotImplementedError(
            "[MapStore - search] Not implemented, please use default index to search data in map store...")
