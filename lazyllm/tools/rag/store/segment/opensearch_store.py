import json

from typing import Dict, List, Union, Optional

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import opensearchpy

from ..store_base import LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE

DEFAULT_INDEX_BODY = {
    'settings': {
        'index': {
            'number_of_shards': 4,
        }
    },
    'mappings': {
        'properties': {
            'uid': {'type': 'keyword', 'index': True},
            'doc_id': {'type': 'keyword', 'index': True},
            'group': {'type': 'keyword', 'index': False},
            'content': {'type': 'text', 'analyzer': 'ik_max_word', 'index': False},
            'meta': {'type': 'text', 'analyzer': 'ik_max_word', 'index': False},
            'global_meta': {'type': 'text', 'analyzer': 'ik_max_word', 'index': False},
            'type': {'type': 'keyword', 'index': False},
            'number': {'type': 'integer', 'index': False},
            'kb_id': {'type': 'keyword', 'index': True},
            'excluded_embed_metadata_keys': {'type': 'keyword', 'index': False},
            'excluded_llm_metadata_keys': {'type': 'keyword', 'index': False},
            'parent': {'type': 'keyword', 'index': False},
            'answer': {'type': 'text', 'analyzer': 'ik_max_word', 'index': False},
            'image_keys': {'type': 'keyword', 'index': False},
        }
    }
}

class OpenSearchStore(LazyLLMStoreBase, capability=StoreCapability.SEGMENT):
    def __init__(self, uris: List[str], client_kwargs: Optional[Dict] = {},
                 index_kwargs: Optional[Union[Dict, List]] = None, **kwargs):
        if isinstance(uris, str):
            uris = [uris]
        self._uris = uris
        self._client_kwargs = client_kwargs
        self._index_kwargs = index_kwargs or DEFAULT_INDEX_BODY
        self._primary_key = "uid"

    @override
    def lazy_init(self) -> None:
        """ load the store """
        if self._client_kwargs.get('user') and self._client_kwargs.get('password'):
            self._client_kwargs['http_auth'] = (self._client_kwargs.pop('user'), self._client_kwargs.pop('password'))
        self._client = opensearchpy.OpenSearch(hosts=self._uris, **self._client_kwargs)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """ upsert data to the store """
        if not data: return
        try:
            if not self._client.indices.exists(index=collection_name):
                self._client.indices.create(index=collection_name, body=self._index_kwargs)
            for i in range(0, len(data), INSERT_BATCH_SIZE):
                bulk_data = []
                batch_data = data[i:i + INSERT_BATCH_SIZE]
                for segment in batch_data:
                    segment = self._serialize_node(segment)
                    bulk_data.append({"index": {"_index": collection_name, "_id": segment.get(self._primary_key)}})
                    bulk_data.append(segment)
                response = self._client.bulk(bulk_data)
                if response["errors"]:
                    raise ValueError(f"Error upserting data to OpenSearch: {response['errors']}")
            self._client.indices.refresh(index=collection_name)
            return True
        except Exception as e:
            LOG.error(f"[OpenSearchStore - upsert] Error upserting data to OpenSearch: {e}")
            return False

    @override
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        """ delete data from the store """
        try:
            if not self._client.indices.exists(index=collection_name):
                raise ValueError(f"Index {collection_name} does not exist")
            resp = self._client.delete_by_query(index=collection_name,
                                                body=self._construct_criteria(criteria), refresh=True)
            if resp.get("failures"):
                raise ValueError(f"Error deleting data from OpenSearch: {resp['failures']}")
            return True
        except Exception as e:
            LOG.error(f"[OpenSearchStore - delete] Error deleting data from OpenSearch: {e}")
            return False

    @override
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        """ get data from the store """
        try:
            if not self._client.indices.exists(index=collection_name):
                raise ValueError(f"Index {collection_name} does not exist")
            results: List[dict] = []
            if self._primary_key in criteria:
                vals = criteria.pop(self._primary_key)
                if not isinstance(vals, list):
                    vals = [vals]
                body = {"ids": vals}
                resp = self._client.mget(index=collection_name, body=body)
                for doc in resp["docs"]:
                    if doc.get("found", False):
                        src = doc["_source"]
                        src["uid"] = doc["_id"]
                        results.append(self._deserialize_node(src))
            else:
                query = self._construct_criteria(criteria)
                for hit in opensearchpy.helpers.scan(client=self._client, index=collection_name, query=query,
                                                     scroll="2m", size=500, preserve_order=True):
                    src = hit["_source"]
                    src["uid"] = hit["_id"]
                    results.append(self._deserialize_node(src))
            return results
        except Exception as e:
            LOG.error(f"[OpenSearchStore - get] Error getting data from OpenSearch: {e}")
            return []

    @override
    def search(self, collection_name: str, query: str, topk: int, **kwargs) -> List[dict]:
        """ search data from the store """
        raise NotImplementedError("[OpenSearchStore - search] Not implemented yet")

    def _serialize_node(self, segment: dict):
        """ serialize node to a dict that can be stored in OpenSearch """
        segment.pop("embedding", None)
        segment["global_meta"] = json.dumps(segment.get("global_meta", {}), ensure_ascii=False)
        segment["meta"] = json.dumps(segment.get("meta", {}), ensure_ascii=False)
        segment["image_keys"] = json.dumps(segment.get("image_keys", []), ensure_ascii=False)
        return segment

    def _deserialize_node(self, segment: dict) -> dict:
        """ deserialize node from dict """
        segment["meta"] = json.loads(segment.get("meta", "{}"))
        segment["global_meta"] = json.loads(segment.get("global_meta", "{}"))
        segment["image_keys"] = json.loads(segment.get("image_keys", "[]"))
        return segment

    def _construct_criteria(self, criteria: dict) -> dict:
        """ construct criteria for OpenSearch """
        if self._primary_key in criteria:
            vals = criteria.pop(self._primary_key)
            if not isinstance(vals, list):
                vals = [vals]
            return {"query": {"ids": {"values": vals}}}
        else:
            must_clauses = []
            for key, value in criteria.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {key: value}})
                else:
                    must_clauses.append({"term": {key: value}})
            return {"query": {"bool": {"must": must_clauses}}}
