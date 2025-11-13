import json
import urllib3
import threading
import importlib.util
import copy

from typing import Dict, Union, List, Optional

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import elasticsearch

from ..store_base import (LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE)
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID, GlobalMetadataDesc
from ..store_base import BUILDIN_GLOBAL_META_DESC
from ...data_type import DataType

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_MAPPING_BODY = {
    'settings': {
        'index': {
            'number_of_shards': 4,
            'number_of_replicas': 1,
            'refresh_interval': '1s',
        },
        'analysis': {
            'tokenizer': {
                'ngram_tokenizer': {
                    'type': 'ngram',
                    'min_gram': 2,
                    'max_gram': 3,
                }
            },
            'analyzer': {
                'ngram_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'ngram_tokenizer',
                }
            }
        }
    },
    'mappings': {
        'dynamic': 'strict',
        'properties': {
            'uid': {'type': 'keyword'},
            'doc_id': {'type': 'keyword'},
            'group': {'type': 'keyword'},
            'kb_id': {'type': 'keyword'},
            'content': {
                'type': 'text',
                'index': True,
                'store': True,
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_smart',
            },
            'answer': {'type': 'text', 'index': True, 'store': True},
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
    capability = StoreCapability.SEGMENT
    need_embedding = False
    supports_index_registration = False

    def __init__(
        self,
        uris: List[str],
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
    def connect(self, global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs) -> bool:
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

            self._global_metadata_desc = global_metadata_desc

            self._index_kwargs = self._adapt_mapping_for_global_metadata()

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
                        f'Error upserting data to Elasticsearch: {response.get("errors")}'
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
                    LOG.warning(f'[ElasticsearchStore - delete] Version conflicts: {resp.get("version_conflicts")}')
                if resp.get('failures'):
                    raise ValueError(
                        f'Error deleting data from Elasticsearch: {resp["failures"]}'
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
            # Query by primary key(mget)
            if criteria and self._primary_key in criteria:
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
    def search(self, collection_name: str, query: Optional[str] = None,
               topk: Optional[int] = 10, query_fields: Optional[List[str]] = None,
               filters: Optional[dict] = None, **kwargs) -> List[Dict]:  # noqa: C901
        if not query_fields:
            # For custom global_metadata_desc, search all text fields
            if self._global_metadata_desc and self._global_metadata_desc != BUILDIN_GLOBAL_META_DESC:
                query_fields = [
                    field_name for field_name, desc in self._global_metadata_desc.items()
                    if desc.data_type in (DataType.VARCHAR, DataType.STRING)
                ]
                if not query_fields:
                    query_fields = ['content', 'answer']
            else:
                query_fields = ['content', 'answer']
        try:
            self._ensure_index(collection_name)
            must_clauses = []
            es_query = {}
            if query:
                text_query = {
                    'multi_match': {
                        'query': query,
                        'fields': query_fields,
                    }
                }
                must_clauses.append(text_query)

            filter_query = self._construct_criteria(filters) if filters else {}

            if must_clauses and filter_query:
                # combine filter_query and must_clauses
                filter_must = filter_query['query']['bool']['must']
                es_query = {'query': {'bool': {'must': must_clauses + filter_must}}}
            elif must_clauses:
                es_query = {'query': {'bool': {'must': must_clauses}}}
            elif filter_query:
                es_query = filter_query
            else:
                es_query = {'query': {'match_all': {}}}

            es_query['size'] = topk

            resp = self._client.search(index=collection_name, body=es_query)

            res = []
            for hit in resp['hits']['hits']:
                seg = self._transform_segment(hit)
                if seg:
                    seg['score'] = hit.get('_score', 0.0)
                    res.append(seg)
            return res

        except Exception as e:
            LOG.error(f'[ElasticSearchStore - search] Error searching {collection_name}: {e}')
            return []

    def _serialize_node(self, segment: Dict) -> Dict:
        seg = dict(segment)
        seg.pop('embedding', None)
        if self._global_metadata_desc and self._global_metadata_desc == BUILDIN_GLOBAL_META_DESC:
            seg['global_meta'] = json.dumps(seg.get('global_meta', {}), ensure_ascii=False)
            seg['meta'] = json.dumps(seg.get('meta', {}), ensure_ascii=False)
            seg['image_keys'] = json.dumps(seg.get('image_keys', []), ensure_ascii=False)
        return seg

    def _deserialize_node(self, segment: Dict) -> Dict:
        seg = dict(segment)
        if self._global_metadata_desc and self._global_metadata_desc == BUILDIN_GLOBAL_META_DESC:
            seg['meta'] = json.loads(seg.get('meta', '{}'))
            seg['global_meta'] = json.loads(seg.get('global_meta', '{}'))
            seg['image_keys'] = json.loads(seg.get('image_keys', '[]'))
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

        def _add_clause(key, val):
            if isinstance(val, list):
                must_clauses.append({'terms': {key: val}})
            else:
                must_clauses.append({'term': {key: val}})
        must_clauses = []
        if RAG_DOC_ID in criteria:
            val = criteria.pop(RAG_DOC_ID)
            _add_clause('doc_id', val)
        if RAG_KB_ID in criteria:
            _add_clause('kb_id', criteria.pop(RAG_KB_ID))
        if 'parent' in criteria:
            must_clauses.append({'term': {'parent': criteria.pop('parent')}})

        for k, v in criteria.items():
            field_key = k
            # For custom text fields, use .keyword subfield for exact matching
            if (self._global_metadata_desc
                and self._global_metadata_desc != BUILDIN_GLOBAL_META_DESC
                and k in self._global_metadata_desc.keys()
            ):
                field_desc = self._global_metadata_desc[k]

                if field_desc.data_type in (DataType.VARCHAR, DataType.STRING):
                    field_key = f"{k}.keyword"
            _add_clause(field_key, v)

        return {'query': {'bool': {'must': must_clauses}}} if must_clauses else {}

    def _transform_segment(self, record: dict) -> dict:
        '''Convert ES into normalized Python dict with uid. Return None if invalid (not found).'''
        src = record['_source']
        src['uid'] = record['_id']
        return self._deserialize_node(src)

    def _check_ik_plugin(self):
        try:
            plugins = self._client.cat.plugins(format='json')
            if any('analysis-ik' in p.get('component', '') for p in plugins):
                return True
            try:
                self._client.indices.analyze(
                    body={
                        'analyzer': 'ik_max_word',
                        'text': 'machine learning'
                    }
                )
                return True
            except Exception as e:
                LOG.warning(f'IK plugin is not installed: {str(e)}')
                return False
        except Exception as e:
            LOG.warning(f'check IK plugin failed: {e}')
            return False

    def _adapt_mapping_for_global_metadata(self) -> dict:
        if not self._global_metadata_desc or self._global_metadata_desc == BUILDIN_GLOBAL_META_DESC:
            return DEFAULT_MAPPING_BODY
        mapping = copy.deepcopy(DEFAULT_MAPPING_BODY)
        mapping['mappings']['dynamic'] = 'true'
        props = {'uid': {'type': 'keyword'}}
        self._type2es = {
            DataType.VARCHAR: 'text',
            DataType.ARRAY: 'array',
            DataType.INT32: 'integer',
            DataType.BOOLEAN: 'boolean',
            DataType.FLOAT: 'float',
            DataType.INT64: 'long',
            DataType.STRING: 'text',
        }

        check_ik = self._check_ik_plugin()
        if check_ik:
            LOG.info('IK plugin is installed')
        else:
            LOG.warning('IK plugin is not installed, ElasticSearch will \
                use ngram analyzer which is English Only Analyzer')
        for field_name, desc in self._global_metadata_desc.items():
            field_type = self._type2es[desc.data_type]
            field_def = {'type': field_type, 'store': True, 'index': True}
            if field_type == 'text':
                # Add keyword subfield for exact matching
                field_def['fields'] = {
                    'keyword': {
                        'type': 'keyword',
                        'ignore_above': 256
                    }
                }
                if check_ik:
                    field_def['analyzer'] = 'ik_max_word'
                    field_def['search_analyzer'] = 'ik_smart'
                else:
                    field_def['analyzer'] = 'ngram_analyzer'
                    field_def['search_analyzer'] = 'ngram_analyzer'
                check_ik = False
            props[field_name] = field_def
        mapping['mappings']['properties'] = props

        return mapping
