import json
import urllib3
import threading
import importlib.util
import copy

from typing import Dict, List, Union, Optional

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import opensearchpy

from ..store_base import LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID, GlobalMetadataDesc
from ...data_type import DataType
from ..store_base import BUILDIN_GLOBAL_META_DESC

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
                'edge_ngram_tokenizer': {
                    'type': 'edge_ngram',
                    'min_gram': 1,
                    'max_gram': 20,
                    'token_chars': ['letter', 'digit']
                },
            },
            'analyzer': {
                'edge_ngram_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'edge_ngram_tokenizer',
                    'filter': ['lowercase']
                },
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
        self._index_kwargs = index_kwargs or DEFAULT_MAPPING_BODY
        self._primary_key = 'uid'

    @property
    def dir(self):
        return None

    @override
    def connect(self, global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs) -> None:
        if self._client_kwargs.get('user') and self._client_kwargs.get('password'):
            self._client_kwargs['http_auth'] = (self._client_kwargs.pop('user'), self._client_kwargs.pop('password'))
        self._ddl_lock = threading.Lock()
        self._client = opensearchpy.OpenSearch(hosts=self._uris, **self._client_kwargs)
        self._global_metadata_desc = global_metadata_desc or {}
        self._index_kwargs = self._adapt_mapping_for_global_metadata()

    def _ensure_index(self, name: str):
        if self._client.indices.exists(index=name):
            return
        with self._ddl_lock:
            if self._client.indices.exists(index=name):
                return
            try:
                self._client.indices.create(index=name, body=self._index_kwargs)
            except opensearchpy.TransportError as e:
                if getattr(e, 'error', '') != 'resource_already_exists_exception':
                    raise e
            except Exception as e:
                LOG.error(f'[OpenSearchStore - _ensure_index] Error creating index {name}: {e}')
                raise e

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        if not data: return
        try:
            self._ensure_index(collection_name)
            for i in range(0, len(data), INSERT_BATCH_SIZE):
                bulk_data = []
                batch_data = data[i:i + INSERT_BATCH_SIZE]
                for segment in batch_data:
                    segment = self._serialize_node(segment)
                    bulk_data.append({'index': {'_index': collection_name, '_id': segment.get(self._primary_key)}})
                    bulk_data.append(segment)
                response = self._client.bulk(index=collection_name, body=bulk_data, refresh='wait_for')
                if response.get('errors'):
                    raise ValueError(f'Error upserting data to OpenSearch: {response.get("errors")}')
            return True
        except Exception as e:
            LOG.error(f'[OpenSearchStore - upsert] Error upserting data to OpenSearch: {e}')
            return False

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f'[OpenSearchStore - delete] Index {collection_name} does not exist')
                return True
            if not criteria:
                with self._ddl_lock:
                    if self._client.indices.exists(index=collection_name):
                        self._client.indices.delete(index=collection_name)
                return True
            else:
                resp = self._client.delete_by_query(index=collection_name, body=self._construct_criteria(criteria),
                                                    refresh=True, conflicts='proceed')
                if resp.get('version_conflicts', 0) > 0:
                    LOG.warning(f'[OpenSearchStore - delete] Version conflicts: {resp.get("version_conflicts")}')
                if resp.get('failures'):
                    raise ValueError(f'Error deleting data from OpenSearch: {resp["failures"]}')
                return True
        except Exception as e:
            LOG.error(f'[OpenSearchStore - delete] Error deleting data from OpenSearch: {e}')
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        try:
            if not self._client.indices.exists(index=collection_name):
                LOG.warning(f'[OpenSearchStore - get] Index {collection_name} does not exist')
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
                        results.append(self._transform_segment(doc))
            else:
                spec = importlib.util.find_spec('opensearchpy.helpers')
                helpers = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helpers)
                query = self._construct_criteria(criteria)
                for hit in helpers.scan(
                    client=self._client,
                    index=collection_name,
                    query=query,
                    scroll='2m',
                    size=500,
                    preserve_order=False
                ):
                    seg = self._transform_segment(hit)
                    if seg:
                        results.append(seg)
            return results
        except Exception as e:
            LOG.error(f'[OpenSearchStore - get] Error getting data from OpenSearch: {e}')
            return []

    @override
    def search(self, collection_name: str, query: Optional[str] = None,
               topk: Optional[int] = 10, filters: Optional[dict] = None, **kwargs) -> List[dict]:  # noqa: C901
        query_fields = ['*']
        try:
            self._ensure_index(collection_name)
            must_clauses = []
            os_query = {}
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
                os_query = {'query': {'bool': {'must': must_clauses + filter_must}}}
            elif must_clauses:
                os_query = {'query': {'bool': {'must': must_clauses}}}
            elif filter_query:
                os_query = filter_query
            else:
                os_query = {'query': {'match_all': {}}}

            os_query['size'] = topk

            resp = self._client.search(index=collection_name, body=os_query)
            res = []
            for hit in resp['hits']['hits']:
                seg = self._transform_segment(hit)
                if seg:
                    seg['score'] = hit.get('_score', 0.0)
                    res.append(seg)
            return res

        except Exception as e:
            LOG.error(f'[OpenSearchStore - search] Error searching data from OpenSearch: {e}')
            return []

    def _serialize_node(self, segment: dict):
        seg = dict(segment)
        seg.pop('embedding', None)
        if self._global_metadata_desc and self._global_metadata_desc == BUILDIN_GLOBAL_META_DESC:
            seg['global_meta'] = json.dumps(seg.get('global_meta', {}), ensure_ascii=False)
            seg['meta'] = json.dumps(seg.get('meta', {}), ensure_ascii=False)
            seg['image_keys'] = json.dumps(seg.get('image_keys', []), ensure_ascii=False)
        return seg

    def _deserialize_node(self, segment: dict) -> dict:
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
            val = criteria.pop(RAG_KB_ID)
            _add_clause('kb_id', val)
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
                    field_key = f'{k}.keyword'
            _add_clause(field_key, v)

        return {'query': {'bool': {'must': must_clauses}}} if must_clauses else {}

    def _transform_segment(self, record: dict) -> dict:
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
        self._type2os = {
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
            LOG.warning('IK plugin is not installed, OpenSearch will \
                use ngram analyzer which is English Only Analyzer')
        for field_name, desc in self._global_metadata_desc.items():
            field_type = self._type2os.get(desc.data_type, None)
            if not field_type:
                LOG.warning(f'Unsupported data type: {desc.data_type}')
                continue
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
                    field_def['analyzer'] = 'edge_ngram_analyzer'
                    field_def['search_analyzer'] = 'standard'
                check_ik = False
            props[field_name] = field_def
        mapping['mappings']['properties'] = props

        return mapping
