import json
import urllib3

from typing import Dict, List, Union, Optional

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import opensearchpy

from ..store_base import LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_INDEX_BODY = {
    'settings': {
        'index': {
            'number_of_shards': 4,
            'number_of_replicas': 1,
            'refresh_interval': '1s',
        }
    },
    'mappings': {
        'dynamic': 'strict',
        'properties': {
            'uid': {'type': 'keyword'},
            'doc_id': {'type': 'keyword'},
            'group': {'type': 'keyword'},
            'kb_id': {'type': 'keyword'},
            'content': {'type': 'text', 'index': False, 'store': True},
            'answer': {'type': 'text', 'index': False, 'store': True},
            'meta': {'type': 'text', 'index': False, 'store': True},
            'global_meta': {'type': 'text', 'index': False, 'store': True},
            'type': {'type': 'keyword', 'store': True},
            'number': {'type': 'integer', 'store': True},
            'excluded_embed_metadata_keys': {'type': 'keyword', 'store': True},
            'excluded_llm_metadata_keys': {'type': 'keyword', 'store': True},
            'parent': {'type': 'keyword', 'store': True},
            'image_keys': {'type': 'keyword', 'store': True},
        }
    }
}


class OpenSearchStore(LazyLLMStoreBase):
    capability = StoreCapability.SEGMENT
    need_embedding = False
    supports_index_registration = False

    def __init__(self, uris: List[str], client_kwargs: Optional[Dict] = None,
                 index_kwargs: Optional[Union[Dict, List]] = None, **kwargs):
        if isinstance(uris, str):
            uris = [uris]
        self._uris = uris
        self._client_kwargs = client_kwargs or {}
        self._index_kwargs = index_kwargs or DEFAULT_INDEX_BODY
        self._primary_key = 'uid'

    @property
    def dir(self):
        return None

    @override
    def connect(self, *args, **kwargs) -> None:
        if self._client_kwargs.get('user') and self._client_kwargs.get('password'):
            self._client_kwargs['http_auth'] = (self._client_kwargs.pop('user'), self._client_kwargs.pop('password'))
        self._client = opensearchpy.OpenSearch(hosts=self._uris, **self._client_kwargs)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        if not data: return
        try:
            if not self._client.indices.exists(index=collection_name):
                self._client.indices.create(index=collection_name, body=self._index_kwargs)
            for i in range(0, len(data), INSERT_BATCH_SIZE):
                bulk_data = []
                batch_data = data[i:i + INSERT_BATCH_SIZE]
                for segment in batch_data:
                    segment = self._serialize_node(segment)
                    bulk_data.append({'index': {'_index': collection_name, '_id': segment.get(self._primary_key)}})
                    bulk_data.append(segment)
                response = self._client.bulk(index=collection_name, body=bulk_data)
                if response['errors']:
                    raise ValueError(f"Error upserting data to OpenSearch: {response['items']}")
            self._client.indices.refresh(index=collection_name)
            return True
        except Exception as e:
            LOG.error(f"[OpenSearchStore - upsert] Error upserting data to OpenSearch: {e}")
            return False

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f"[OpenSearchStore - delete] Index {collection_name} does not exist")
                return True
            if not criteria:
                self._client.indices.delete(index=collection_name)
                return True
            else:
                resp = self._client.delete_by_query(index=collection_name,
                                                    body=self._construct_criteria(criteria), refresh=True)
                if resp.get('failures'):
                    raise ValueError(f"Error deleting data from OpenSearch: {resp['failures']}")
                return True
        except Exception as e:
            LOG.error(f"[OpenSearchStore - delete] Error deleting data from OpenSearch: {e}")
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f"[OpenSearchStore - get] Index {collection_name} does not exist")
                return []
            results: List[dict] = []
            criteria = dict(criteria) if criteria else {}
            if criteria and self._primary_key in criteria:
                vals = criteria.pop(self._primary_key)
                if not isinstance(vals, list):
                    vals = [vals]
                body = {'ids': vals}
                resp = self._client.mget(index=collection_name, body=body)
                for doc in resp['docs']:
                    if doc.get('found', False):
                        src = doc['_source']
                        src['uid'] = doc['_id']
                        results.append(self._deserialize_node(src))
            else:
                query = self._construct_criteria(criteria)
                for hit in opensearchpy.helpers.scan(client=self._client, index=collection_name, query=query,
                                                     scroll='2m', size=500, preserve_order=True):
                    src = hit['_source']
                    src['uid'] = hit['_id']
                    results.append(self._deserialize_node(src))
            return results
        except Exception as e:
            LOG.error(f"[OpenSearchStore - get] Error getting data from OpenSearch: {e}")
            return []

    @override
    def search(self, collection_name: str, query: str, topk: int, **kwargs) -> List[dict]:
        raise NotImplementedError("[OpenSearchStore - search] Not implemented yet")

    def _serialize_node(self, segment: dict):
        seg = dict(segment)
        seg.pop('embedding', None)
        seg['global_meta'] = json.dumps(seg.get('global_meta', {}), ensure_ascii=False)
        seg['meta'] = json.dumps(seg.get('meta', {}), ensure_ascii=False)
        seg['image_keys'] = json.dumps(seg.get('image_keys', []), ensure_ascii=False)
        return seg

    def _deserialize_node(self, segment: dict) -> dict:
        segment['meta'] = json.loads(segment.get('meta', "{}"))
        segment['global_meta'] = json.loads(segment.get('global_meta', "{}"))
        segment['image_keys'] = json.loads(segment.get('image_keys', "[]"))
        return segment

    def _construct_criteria(self, criteria: Optional[dict] = None) -> dict:
        criteria = dict(criteria) if criteria else {}
        if not criteria:
            return {}
        if self._primary_key in criteria:
            vals = criteria.pop(self._primary_key)
            if not isinstance(vals, list):
                vals = [vals]
            return {'query': {'ids': {'values': vals}}}
        else:
            must_clauses = []
            if RAG_DOC_ID in criteria:
                must_clauses.append({'terms': {'doc_id': criteria.pop(RAG_DOC_ID)}})
            if RAG_KB_ID in criteria:
                must_clauses.append({'term': {'kb_id': criteria.pop(RAG_KB_ID)}})
            if 'parent' in criteria:
                must_clauses.append({'terms': {'parent': criteria.pop('parent')}})
            return {'query': {'bool': {'must': must_clauses}}}
