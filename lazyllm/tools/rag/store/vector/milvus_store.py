""" Milvus Vector Store (For Vector Store Only)"""
import copy
import json

from packaging import version
from urllib import parse
from typing import Dict, List, Union, Optional, Set

from lazyllm import LOG
from lazyllm.thirdparty import pymilvus
from lazyllm.common import override

from ..store_base import (LazyLLMStoreBase, BUILDIN_GLOBAL_META_DESC, StoreCapability,
                          GLOBAL_META_KEY_PREFIX, EMBED_PREFIX)
from ..data_type import DataType
from ..global_metadata import GlobalMetadataDesc

MILVUS_UPSERT_BATCH_SIZE = 500
MILVUS_PAGINATION_OFFSET = 1000
TYPE2MILVUS = {
    DataType.VARCHAR: pymilvus.DataType.VARCHAR,
    DataType.ARRAY: pymilvus.DataType.ARRAY,
    DataType.FLOAT_VECTOR: pymilvus.DataType.FLOAT_VECTOR,
    DataType.INT32: pymilvus.DataType.INT32,
    DataType.SPARSE_FLOAT_VECTOR: pymilvus.DataType.SPARSE_FLOAT_VECTOR,
}
BUILTIN_KEYS = {
    'uid': {'dtype': pymilvus.DataType.VARCHAR, 'max_length': 256, 'is_primary': True}
}


class MilvusStore(LazyLLMStoreBase, capability=StoreCapability.VECTOR):
    def __init__(self, uri: str = "", db_name: str = 'lazyllm', index_kwargs: Optional[Union[Dict, List]] = None,
                 client_kwargs: Optional[Dict] = {}, embed_dims: Optional[Dict[str, int]] = {},
                 embed_datatypes: Optional[Dict[str, DataType]] = {},
                 global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None):
        # one database, different collection for each group (for standalone, add prefix to collection name)
        # when there's data need upsert, collection creation happen.
        self._uri = uri
        self._db_name = db_name
        self._index_kwargs = index_kwargs
        self._client_kwargs = client_kwargs
        self._embed_dims = embed_dims
        self._embed_datatypes = embed_datatypes
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self._primary_key = 'uid'

    @override
    def lazy_init(self):
        self._client = pymilvus.MilvusClient(uri=self._uri, **self._client_kwargs)
        if self._uri and parse.urlparse(self._uri).scheme.lower() not in ["unix", "http", "https", "tcp", "grpc"]:
            self._type = 'local'
        else:
            self._type = 'remote'
            try:
                if self._db_name:
                    existing_dbs = self._client.list_databases()
                    if self._db_name not in existing_dbs:
                        self._client.create_database(self._db_name)
                    self._client.using_database(self._db_name)
            except Exception as e:
                LOG.error(f'milvus-standalone database error {e}')
        self._constant_fields = self._get_constant_fields()
        LOG.info("[Milvus Vector Store] init success!")

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """ upsert data to the store """
        try:
            if not data: return
            data_embeddings = data[0].get("embedding", {})
            if not data_embeddings: return

            if not self._client.has_collection(collection_name):
                embed_kwargs = {}
                for embed_key, embedding in data_embeddings.items():
                    assert self._embed_datatypes.get(embed_key), \
                        f'cannot find embedding params for embed [{embed_key}]'
                    if embed_key not in embed_kwargs:
                        embed_kwargs[embed_key] = {"dtype": TYPE2MILVUS[self._embed_datatypes[embed_key]]}
                    if self._embed_dims.get(embed_key): embed_kwargs[embed_key]["dim"] = self._embed_dims[embed_key]
                self._create_collection(collection_name, embed_kwargs)

            self._check_connection()
            for i in range(0, len(data), MILVUS_UPSERT_BATCH_SIZE):
                self._client.upsert(collection_name=collection_name,
                                    data=[self._serialize_data(d) for d in data[i:i + MILVUS_UPSERT_BATCH_SIZE]])
            return True
        except Exception as e:
            LOG.error(f'[Milvus Store - upsert] error: {e}')
            return False

    @override
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        """ delete data from the store """
        try:
            self._check_connection()
            self._client.delete(collection_name=collection_name, **self._construct_criteria(criteria))
            return True
        except Exception as e:
            LOG.error(f'[Milvus Store - delete] error: {e}')
            return False

    @override
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        """ get data from the store """
        try:
            self._check_connection()
            col_desc = self._client.describe_collection(collection_name=collection_name)
            field_names = [field.get("name") for field in col_desc.get('fields', [])
                           if field.get("name").startswith(EMBED_PREFIX)]
            if self._primary_key in criteria:
                res = self._client.get(collection_name=collection_name, ids=criteria[self._primary_key])
            else:
                filters = self._construct_criteria(criteria)
                if version.parse(pymilvus.__version__) >= version.parse('2.4.11'):
                    iterator = self._client.query_iterator(collection_name=collection_name,
                                                           batch_size=MILVUS_PAGINATION_OFFSET,
                                                           output_fields=field_names, **filters)
                    res = []
                    while True:
                        result = iterator.next()
                        if not result:
                            iterator.close()
                            break
                        res += result
                else:
                    res = self._client.query(collection_name=collection_name, output_fields=field_names, **filters)
            return [self._deserialize_data(r) for r in res]
        except Exception as e:
            LOG.error(f'[Milvus Store - get] error: {e}')
            return []

    def _check_connection(self):
        if not pymilvus.connections.has_connection(alias=self._client._using):
            LOG.info("[Milvus Store] try to reconnect...")
            if self._type == 'local':
                pymilvus.connections.connect(alias=self._client._using, uri=self._uri)
            else:
                pymilvus.connections.connect(alias=self._client._using, db_name=self._db_name, uri=self._uri)

    def _get_constant_fields(self) -> List[pymilvus.FieldSchema]:
        """ get constant field schema for collection """
        field_list = []
        for k, kws in BUILTIN_KEYS.items():
            field_list.append(pymilvus.FieldSchema(name=k, **kws))
        for k, desc in self._global_metadata_desc.items():
            if desc.data_type == DataType.ARRAY:
                if not desc.element_type:
                    raise ValueError(f'Milvus field [{k}]: `element_type` is required when `data_type` is ARRAY.')
                field_args = {'element_type': TYPE2MILVUS[desc.element_type], 'max_capacity': desc.max_size}
                if desc.element_type == DataType.VARCHAR: field_args['max_length'] = 65535
            elif desc.data_type == DataType.VARCHAR:
                field_args = {'max_length': desc.max_size}
            else:
                field_args = {}
            field_list.append(pymilvus.FieldSchema(name=k, dtype=TYPE2MILVUS[desc.data_type],
                                                   default_value=desc.default_value, **field_args))
        return field_list

    def _create_collection(self, collection_name: str, embed_kwargs: Dict[str, Dict]):  # noqa: C901
        field_list = copy.deepcopy(self._constant_fields)
        self._check_connection()
        index_params = self._client.prepare_index_params()
        for k, kws in embed_kwargs.items():
            embed_field_name = self._gen_embed_key(k)
            field_list.append(pymilvus.FieldSchema(name=embed_field_name, **kws))
            index_params.add_index(field_name=embed_field_name, **kws)
            if isinstance(self._index_kwargs, list):
                for item in self._index_kwargs:
                    embed_key = item.get('embed_key', None)
                    if not embed_key:
                        raise ValueError(f'cannot find `embed_key` in `index_kwargs` of `{item}`')
                    if embed_key == k:
                        index_kwarg = item.copy()
                        index_kwarg.pop('embed_key', None)
                        index_params.add_index(field_name=k, **index_kwarg)
                        break
            elif isinstance(self._index_kwargs, dict):
                index_params.add_index(field_name=k, **self._index_kwargs)
        schema = pymilvus.CollectionSchema(fields=field_list, auto_id=False, enable_dynamic_field=False)
        self._client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    def _serialize_data(self, d: dict) -> dict:
        """ prepare data for upsert """
        # only keep primary_key, embedding and global_meta
        res = {
            self._primary_key: d.get(self._primary_key, '')
        }
        for embed_key, value in d.get('embedding', {}).items():
            res[self._gen_embed_key(embed_key)] = value
        global_meta = json.loads(d.get('global_meta'))
        for name, desc in self._global_metadata_desc.items():
            value = global_meta.get(name, desc.default_value)
            if value is not None:
                res[self._gen_global_meta_key(name)] = value
        return res

    def _deserialize_data(self, d: dict) -> dict:
        """ deserialize data from vector store """
        res = {
            self._primary_key: d.get(self._primary_key, '')
        }
        for k, v in d.items():
            if k.startswith(EMBED_PREFIX):
                res['embedding'][k[len(EMBED_PREFIX):]] = v
        return res

    def _gen_embed_key(self, k: str) -> str:
        return EMBED_PREFIX + k

    def _gen_global_meta_key(self, k: str) -> str:
        return GLOBAL_META_KEY_PREFIX + k

    def _construct_criteria(self, criteria: dict) -> dict:
        """ construct criteria for delete """
        res = {}
        if self._primary_key in criteria:
            res["ids"] = criteria[self._primary_key]
        else:
            filter_str = ""
            for key, vaule in criteria.items():
                if len(filter_str) > 0:
                    filter_str += ' AND '
                if isinstance(vaule, list):
                    filter_str += f'{key} in {vaule}'
                elif isinstance(vaule, str):
                    filter_str += f'{key} == {vaule}'
                else:
                    raise ValueError(f'invalid criteria type: {type(vaule)}')
            res["filter"] = filter_str
        return res

    @override
    def search(self, collection_name: str, query: Union[dict, List[float]], topk: int,
               filters: Optional[Dict[str, Union[List, set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        self._check_connection()
        if not embed_key or embed_key not in self._embed_datatypes:
            raise ValueError(f'[Milvus Store - search] Not supported or None `embed_key`: {embed_key}')
        res = []
        results = self._client.search(collection_name=collection_name, data=[query], limit=topk,
                                      anns_field=self._gen_embed_key(embed_key),
                                      filter=self._construct_filter_expr(filters))
        if len(results) != 1:
            raise ValueError(f'number of results [{len(results)}] != expected [1]')
        for result in results[0]:
            score = result.get('distance', 0)
            uid = result.get('id', result.get(self._primary_key, ''))
            if not uid:
                continue
            res.append({'uid': uid, 'score': score})
        return res

    def _construct_filter_expr(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret_str = ""
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            key = self._gen_global_meta_key(name)
            if isinstance(candidates, str):
                candidates = [candidates]
            elif (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '
        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '
        return ret_str
