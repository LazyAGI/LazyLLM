from typing import Dict, List, Optional, Union, Set

from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability


class HybridStore(LazyLLMStoreBase):
    capability = StoreCapability.ALL
    need_embedding = True
    supports_index_registration = False

    def __init__(self, segment_store: LazyLLMStoreBase, vector_store: LazyLLMStoreBase):
        self.segment_store: LazyLLMStoreBase = segment_store
        self.vector_store: LazyLLMStoreBase = vector_store

    @property
    def dir(self):
        return self.segment_store.dir

    @override
    def connect(self, *args, **kwargs):
        self.segment_store.connect(*args, **kwargs)
        self.vector_store.connect(*args, **kwargs)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        segments = [{k: v for k, v in segment.items() if k != 'embedding'} for segment in data]
        return self.segment_store.upsert(collection_name=collection_name, data=segments) and \
            self.vector_store.upsert(collection_name=collection_name, data=data)

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        return self.segment_store.delete(collection_name=collection_name, criteria=criteria, **kwargs) and \
            self.vector_store.delete(collection_name=collection_name, criteria=criteria, **kwargs)

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        res_segments = self.segment_store.get(collection_name=collection_name, criteria=criteria, **kwargs)
        if not res_segments: return []
        uids = [item.get('uid') for item in res_segments]
        res_vectors = self.vector_store.get(collection_name=collection_name, criteria={'uid': uids})

        data = {}
        for item in res_segments:
            data[item.get('uid')] = item
        for item in res_vectors:
            if item.get('uid') in data:
                data[item.get('uid')]['embedding'] = item.get('embedding')
            else:
                raise ValueError(f"[HybridStore - get] uid {item['uid']} in vector store"
                                 " but not found in segment store")
        return list(data.values())

    @override
    def search(self, collection_name: str, query: str, query_embedding: Optional[Union[dict, List[float]]] = None,
               topk: int = 10, filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        if embed_key:
            # vector store only give uid and score
            res = self.vector_store.search(collection_name=collection_name, query=query, query_embedding=query_embedding,
                                           topk=topk, filters=filters, embed_key=embed_key, **kwargs)
            if not res: return []
            uid2score = {item['uid']: item['score'] for item in res}
            uids = list(uid2score.keys())
            segments = self.segment_store.get(collection_name=collection_name, criteria={'uid': uids})
            for segment in segments:
                segment['score'] = uid2score.get(segment['uid'], 0)
            return segments
        else:
            res = self.segment_store.search(collection_name=collection_name, query=query,
                                            topk=topk, filters=filters, **kwargs)
            return res
