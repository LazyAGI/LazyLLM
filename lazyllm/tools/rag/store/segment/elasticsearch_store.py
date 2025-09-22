import json
import urllib3
import threading
import importlib.util

from typing import Dict, Union, List, Optional

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import elasticsearch

from ..store_base import (LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE)
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_MAPPING_BODY = {
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
        },
    },
}


class ElasticSearchStore(LazyLLMStoreBase):
    capability: StoreCapability.SEGMENT
    need_embedding: False
    supports_index_registration: False

    def __init__(
        self,
        uris: List[str] = '',
        client_kwargs: Optional[Dict] = None,
        index_kwargs: Optional[Union[Dict, List]] = None,
        **kwargs,
    ):
        if isinstance(uris, str):
            uris = [uris]

        self._uris = uris
        self._client_kwargs = client_kwargs or {}
        self._index_kwargs = index_kwargs or DEFAULT_MAPPING_BODY
        self._primary_key = 'uid'

    @property
    def dir(self):
        return None

    @override
    def connect(self, *args, **kwargs) -> bool:
        try:
            self._ddl_lock = threading.Lock()
            # Elastic Cloud
            if self._client_kwargs.get('cloud_id') and self._client_kwargs.get('api_key'):
                cloud_id = self._client_kwargs.get('cloud_id')
                api_key = self._client_kwargs.get('api_key')

                self._client = elasticsearch.Elasticsearch(cloud_id=cloud_id, api_key=api_key)
                if not self._client.ping():
                    raise ConnectionError(f'Failed to ping ES {self._uris}')
            # local or remote url Elasticsearch
            else:
                self._client = elasticsearch.Elasticsearch(hosts=self._uris, **self._client_kwargs)

            return True
        # connection failed exception handling
        except elasticsearch.NotFoundError as e:
            LOG.error(f'ElasticSearch sever with cloud id {cloud_id} and api key {api_key} does not exist')
            raise e
        except elasticsearch.AuthenticationException as e:
            LOG.error('ElasticSearch needs Authentication')
            raise e
        except elasticsearch.AuthorizationException as e:
            LOG.error('Unauthorized to access')
            raise e
        except Exception as e:
            LOG.error(f'Fail to connect ElasticSearch sever with cloud id {cloud_id} and api key {api_key}')
            raise e

    @override
    def _ensure_index(self, index: str = None) -> bool:
        if not index or self._client.indices.exists(index=index):
            return False
        try:
            self._client.indices.create(index=index, body=self._index_kwargs)
            return True
        except elasticsearch.TransportError as e:
            if getattr(e, 'error', '') != 'resource_already_exists_exception':
                raise e
        except Exception as e:
            LOG.error(f'[ElasticSearch - _ensure_index] Error creating index {index}: {e}')
            raise e

    @override
    def upsert(self, collection_name: str = None, data: List[Dict] = None) -> bool:
        if not data:
            return False
        try:
            self._ensure_index(collection_name)
            for i in range(0, len(data), INSERT_BATCH_SIZE):
                bulk_data = []
                batch_data = data[i: i + INSERT_BATCH_SIZE]
                for segment in batch_data:
                    segment = self._serialize_node(segment)
                    _id = segment.pop(self._primary_key, None)
                    bulk_data.append({'index': {'_index': collection_name, '_id': _id}})
                    bulk_data.append(segment)

                response = self._client.bulk(index=collection_name, body=bulk_data, refresh='wait_for')
                if response.get('errors'):
                    raise ValueError(
                        f"Error upserting data to Elasticsearch: {response.get('errors')}"
                    )
            return True

        except Exception as e:
            LOG.error(f'[ElasticSearchStore - upsert] Error upserting documents to {collection_name}: {e}')
            raise e

    @override
    def delete(self, collection_name: str = None, criteria: Optional[Dict] = None, **kwargs) -> bool:
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f'[ElasticSearchStore - delete] Index {collection_name} does not exist')
                return True
            if not criteria:
                with self._ddl_lock:
                    if self._client.indices.exists(index=collection_name):
                        self._client.indices.delete(index=collection_name)
                return True
            else:
                resp = self._client.delete_by_query(
                    index=collection_name,
                    body=self._construct_criteria(criteria),
                    refresh=True,
                    conflicts='proceed',
                )

                if resp.get('version_conflicts', 0) > 0:
                    LOG.warning(f"[ElasticsearchStore - delete] Version conflicts: {resp.get('version_conflicts')}")
                if resp.get('failures'):
                    raise ValueError(
                        f"Error deleting data from Elasticsearch: {resp['failures']}"
                    )
                return True

        except Exception as e:
            LOG.error(f'[ElasticSearchStore - delete] Error deleting from {collection_name}: {e}')
            raise e

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:  # noqa: C901
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f'[ElasticsearchStore - get] Index {collection_name} does not exist')
                return []

            results: List[dict] = []
            criteria = dict(criteria) if criteria else {}

            # Since criteria is empty, return all data
            if not criteria:
                resp = self._client.search(index=collection_name, body={'query': {'match_all': {}}})
                for hit in resp['hits']['hits']:
                    seg = self._transform_segment(hit)
                    if seg:
                        results.append(seg)

            # Query by primary key(mget)
            elif criteria and self._primary_key in criteria:
                vals = criteria.pop(self._primary_key)
                if not isinstance(vals, list):
                    vals = [vals]

                resp = self._client.mget(index=collection_name, body={'ids': vals})

                for doc in resp['docs']:
                    if doc.get('found', False):
                        seg = self._transform_segment(doc)
                        if seg:
                            results.append(seg)

            else:
                spec = importlib.util.find_spec('elasticsearch.helpers')
                helpers = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helpers)
                query = self._construct_criteria(criteria)
                for hit in helpers.scan(
                    client=self._client,
                    index=collection_name,
                    query=query,  # 8.x need to wrap a query
                    scroll='2m',
                    size=500,
                    preserve_order=False,
                ):
                    seg = self._transform_segment(hit)
                    if seg:
                        results.append(seg)

            return results

        except Exception as e:
            LOG.error(f'[ElasticsearchStore - get] Error getting data from Elasticsearch: {e}')
            return []

    @override
    def search(self, collection_name: str = None, query: str = None, topk: int = 10, **kwargs) -> List[Dict]:
        raise NotImplementedError('[ElasticSearchStore - search] Not implemented yet')

    def _serialize_node(self, segment: Dict) -> Dict:
        seg = dict(segment)
        seg.pop('embedding', None)
        seg['global_meta'] = json.dumps(seg.get('global_meta', {}), ensure_ascii=False)
        seg['meta'] = json.dumps(seg.get('meta', {}), ensure_ascii=False)
        seg['image_keys'] = json.dumps(seg.get('image_keys', []), ensure_ascii=False)
        return seg

    def _deserialize_node(self, segment: Dict) -> Dict:
        seg = dict(segment)
        seg['global_meta'] = json.loads(seg.pop('global_meta', '{}'))
        seg['meta'] = json.loads(seg.pop('meta', '{}'))
        seg['image_keys'] = json.loads(seg.pop('image_keys', '[]'))
        return seg

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
                if isinstance(criteria.get(RAG_DOC_ID), list):
                    must_clauses.append({'terms': {'doc_id': criteria.get(RAG_DOC_ID)}})
            if RAG_KB_ID in criteria:
                must_clauses.append({'term': {'kb_id': criteria.get(RAG_KB_ID)}})
            if 'parent' in criteria:
                must_clauses.append({'term': {'parent': criteria.pop('parent')}})

            return {'query': {'bool': {'must': must_clauses}}}

    def _transform_segment(self, record: dict) -> dict:
        '''Convert ES into normalized Python dict with uid. Return None if invalid (not found).'''
        src = record['_source']
        src['uid'] = record['_id']
        return self._deserialize_node(src)
