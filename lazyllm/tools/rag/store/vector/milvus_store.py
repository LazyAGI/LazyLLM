import copy
import traceback
import threading

from contextlib import contextmanager
from queue import Queue, Empty, Full
from packaging import version
from urllib import parse
from pathlib import Path
from typing import Dict, List, Union, Optional, Set

from lazyllm import LOG
from lazyllm.thirdparty import pymilvus
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability, GLOBAL_META_KEY_PREFIX, EMBED_PREFIX
from ...data_type import DataType
from ...global_metadata import GlobalMetadataDesc

MILVUS_UPSERT_BATCH_SIZE = 500
MILVUS_PAGINATION_OFFSET = 1000
MILVUS_INDEX_MAX_RETRY = 3
MILVUS_INDEX_TYPE_DEFAULTS = {
    'HNSW': {'metric_type': 'COSINE', 'params': {'M': 16, 'efConstruction': 200}},
    'IVF_FLAT': {'metric_type': 'L2', 'params': {'nlist': 1024}},
    'IVF_SQ8': {'metric_type': 'L2', 'params': {'nlist': 1024}},
    'IVF_PQ': {'metric_type': 'L2', 'params': {'nlist': 1024, 'm': 8, 'nbits': 8}},
    'FLAT': {'metric_type': 'L2', 'params': {}},
    'GPU_IVF_FLAT': {'metric_type': 'L2', 'params': {'nlist': 1024}},
    'GPU_IVF_SQ8': {'metric_type': 'L2', 'params': {'nlist': 1024}},
    'GPU_IVF_PQ': {'metric_type': 'L2', 'params': {'nlist': 1024, 'm': 8, 'nbits': 8}},
    'DISKANN': {'metric_type': 'L2', 'params': {'nlist': 1024}},
    'BIN_FLAT': {'metric_type': 'HAMMING', 'params': {}},
    'BIN_IVF_FLAT': {'metric_type': 'HAMMING', 'params': {'nlist': 1024}},
    'SPARSE_INVERTED_INDEX': {'metric_type': 'IP', 'params': {'inverted_index_algo': 'DAAT_MAXSCORE'}},
    'AUTOINDEX': {'metric_type': 'COSINE', 'params': {'nlist': 128}},
}
MILVUS_SPARSE_FLOAT_VECTOR_DEFAULT_VALUE = {0: 0.0}

class _ClientPool:
    def __init__(self, maker, max_size: int = 8):
        self._q = Queue(maxsize=max_size)
        self._maker = maker

    def acquire(self):
        try:
            return self._q.get_nowait()
        except Empty:
            return self._maker()

    def release(self, c):
        try:
            self._q.put_nowait(c)
        except Full:
            c.close()


class MilvusStore(LazyLLMStoreBase):
    capability = StoreCapability.VECTOR
    need_embedding = True
    supports_index_registration = False

    def __init__(self, uri: str = '', db_name: str = 'lazyllm', index_kwargs: Optional[Union[Dict, List]] = None,
                 client_kwargs: Optional[Dict] = None):
        # one database, different collection for each group (for standalone, add prefix to collection name)
        # when there's data need upsert, collection creation happen.
        self._uri = uri
        self._db_name = db_name
        self._index_kwargs = index_kwargs
        self._client_kwargs = client_kwargs or {}
        self._primary_key = 'uid'
        if self._uri and parse.urlparse(self._uri).scheme.lower() in ['unix', 'http', 'https', 'tcp', 'grpc']:
            self._is_remote = True
        else:
            self._is_remote = False

    @property
    def dir(self):
        if self._is_remote: return None
        p = Path(self._uri)
        p = p if p.suffix else (p / 'milvus.db')
        return str(p.resolve(strict=False))

    @override
    def connect(self, embed_dims: Optional[Dict[str, int]] = None,
                embed_datatypes: Optional[Dict[str, DataType]] = None,
                global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs):
        self._embed_dims = embed_dims or {}
        self._embed_datatypes = embed_datatypes or {}
        self._global_metadata_desc = global_metadata_desc or {}
        self._set_constants()

        self._ddl_lock = threading.Lock()
        self._db_ready = False
        self._ensure_database()
        self._index_kwargs = self.validate_milvus_embed_keys(self._index_kwargs)

        max_pool_size = int(self._client_kwargs.pop('max_pool_size', 8))
        self._client_pool = _ClientPool(self._new_client, max_size=max_pool_size)
        LOG.info('[Milvus Vector Store] init success!')

    def _new_client(self):
        kwargs = dict(self._client_kwargs)
        try:
            c = pymilvus.MilvusClient(uri=self._uri, **kwargs)
            if self._is_remote and self._db_name:
                c.using_database(self._db_name)
            return c
        except Exception as e:
            LOG.error(f'[Milvus Store - _new_client] error: {e}')
            raise e

    def _ensure_database(self):
        if not (self._is_remote and self._db_name) or self._db_ready:
            return
        tmp = pymilvus.MilvusClient(uri=self._uri, **self._client_kwargs)
        try:
            with self._ddl_lock:
                if self._db_ready:
                    return
                need_create = True
                try:
                    db_list = tmp.list_databases()
                    need_create = self._db_name not in db_list
                except Exception:
                    pass
                if need_create:
                    try:
                        tmp.create_database(self._db_name)
                    except Exception as e:
                        if 'already exist' not in str(e).lower():
                            raise
                self._db_ready = True
        finally:
            tmp.close()

    @contextmanager
    def _client_context(self):
        c = self._client_pool.acquire()
        try:
            yield c
        finally:
            self._client_pool.release(c)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        try:
            if not data: return True
            data_embeddings = data[0].get('embedding', {})
            if not data_embeddings: return True
            with self._client_context() as client:
                if not client.has_collection(collection_name):
                    embed_kwargs = {}
                    for embed_key in data_embeddings.keys():
                        assert self._embed_datatypes.get(embed_key), \
                            f'cannot find embedding params for embed [{embed_key}]'
                        if embed_key not in embed_kwargs:
                            embed_kwargs[embed_key] = {'dtype': self._type2milvus[self._embed_datatypes[embed_key]]}
                        if self._embed_dims.get(embed_key): embed_kwargs[embed_key]['dim'] = self._embed_dims[embed_key]
                    with self._ddl_lock:
                        if not client.has_collection(collection_name):
                            self._create_collection(client, collection_name, embed_kwargs)

                for i in range(0, len(data), MILVUS_UPSERT_BATCH_SIZE):
                    client.upsert(collection_name=collection_name,
                                  data=[self._serialize_data(d) for d in data[i:i + MILVUS_UPSERT_BATCH_SIZE]])
            return True
        except Exception as e:
            LOG.error(f'[Milvus Store - upsert] error: {e}')
            LOG.error(traceback.format_exc())
            return False

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            with self._client_context() as client:
                if not client.has_collection(collection_name):
                    return True
                client.load_collection(collection_name)
                if not criteria:
                    with self._ddl_lock:
                        if client.has_collection(collection_name):
                            client.drop_collection(collection_name=collection_name)
                else:
                    client.delete(collection_name=collection_name, **self._construct_criteria(criteria))
            return True
        except Exception as e:
            LOG.error(f'[Milvus Store - delete] error: {e}')
            LOG.error(traceback.format_exc())
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:  # noqa: C901
        try:
            with self._client_context() as client:
                if not client.has_collection(collection_name):
                    return []
                client.load_collection(collection_name)
                col_desc = client.describe_collection(collection_name=collection_name)
                field_names = [field.get('name') for field in col_desc.get('fields', [])
                               if field.get('name').startswith(EMBED_PREFIX)]
                query_kwargs = self._construct_criteria(criteria) if criteria else {}
                if version.parse(pymilvus.__version__) < version.parse('2.4.11'):
                    # For older versions, batch query manually
                    res = self._batch_query_legacy(client, collection_name, field_names, query_kwargs)
                else:
                    if criteria and self._primary_key in criteria:
                        ids = criteria[self._primary_key]
                        if isinstance(ids, str):
                            ids = [ids]
                        query_kwargs = {'filter': f'{self._primary_key} in {ids}'}
                        # return all fields
                        field_names = None
                    else:
                        query_kwargs.update(**kwargs)

                    iterator = client.query_iterator(collection_name=collection_name,
                                                     batch_size=MILVUS_PAGINATION_OFFSET,
                                                     output_fields=field_names, **query_kwargs)
                    res = []
                    while True:
                        result = iterator.next()
                        if not result:
                            iterator.close()
                            break
                        res += result
            return [self._deserialize_data(r) for r in res]
        except Exception as e:
            LOG.error(f'[Milvus Store - get] error: {e}')
            LOG.error(traceback.format_exc())
            return []

    def _batch_query_legacy(self, client, collection_name: str, field_names: List[str], kwargs: dict) -> List[dict]:
        res = []
        offset = 0
        batch_size = MILVUS_PAGINATION_OFFSET

        while True:
            try:
                # Add offset and limit to filters for pagination
                batch_kwargs = dict(kwargs)
                batch_kwargs['offset'] = offset
                batch_kwargs['limit'] = batch_size

                batch_res = client.query(collection_name=collection_name, output_fields=field_names, **batch_kwargs)
                if not batch_res:
                    break

                res.extend(batch_res)
                if len(batch_res) < batch_size:
                    break
                offset += batch_size
            except Exception as e:
                LOG.error(f'[Milvus Store - _batch_query_legacy] error: {e}')
                raise
        return res

    def _set_constants(self):
        self._type2milvus = {
            DataType.VARCHAR: pymilvus.DataType.VARCHAR,
            DataType.ARRAY: pymilvus.DataType.ARRAY,
            DataType.FLOAT_VECTOR: pymilvus.DataType.FLOAT_VECTOR,
            DataType.INT32: pymilvus.DataType.INT32,
            DataType.INT64: pymilvus.DataType.INT64,
            DataType.SPARSE_FLOAT_VECTOR: pymilvus.DataType.SPARSE_FLOAT_VECTOR,
            DataType.STRING: pymilvus.DataType.STRING,
        }
        self._builtin_keys = {
            'uid': {'dtype': pymilvus.DataType.VARCHAR, 'max_length': 256, 'is_primary': True}
        }
        self._constant_fields = self._get_constant_fields()

    def _get_constant_fields(self) -> list:
        field_list = []
        for k, kws in self._builtin_keys.items():
            field_list.append(pymilvus.FieldSchema(name=k, **kws))
        for k, desc in self._global_metadata_desc.items():
            field_name = self._gen_global_meta_key(k)
            if desc.data_type == DataType.ARRAY:
                if desc.element_type is None:
                    raise ValueError(f'Milvus field [{field_name}]: '
                                     '`element_type` is required when `data_type` is ARRAY.')
                field_args = {'element_type': self._type2milvus[desc.element_type], 'max_capacity': desc.max_size}
                if desc.element_type == DataType.VARCHAR: field_args['max_length'] = 65535
            elif desc.data_type == DataType.VARCHAR:
                field_args = {'max_length': desc.max_size}
            else:
                field_args = {}
            field_list.append(pymilvus.FieldSchema(name=field_name, dtype=self._type2milvus[desc.data_type],
                                                   default_value=desc.default_value, **field_args))
        return field_list

    def _create_collection(self, client, collection_name: str, embed_kwargs: Dict[str, Dict],  # noqa: C901
                           retry: int = 0):
        field_list = copy.deepcopy(self._constant_fields)
        index_params = client.prepare_index_params()
        original_index_kwargs = copy.deepcopy(self._index_kwargs)

        # Pre-process index_kwargs to create a lookup dictionary for O(1) access
        index_kwargs_lookup = {}
        if isinstance(original_index_kwargs, dict):
            original_index_kwargs = [original_index_kwargs]
        for item in original_index_kwargs:
            embed_key = item.get('embed_key', None)
            if not embed_key:
                raise ValueError(f'cannot find `embed_key` in `index_kwargs` of `{item}`')
            # add default values to the params of each index item with no overrides
            self._ensure_params_defaults(item)
            index_kwargs_lookup[embed_key] = item.copy()
            index_kwargs_lookup[embed_key].pop('embed_key', None)
        for k, kws in embed_kwargs.items():
            embed_field_name = self._gen_embed_key(k)
            field_list.append(pymilvus.FieldSchema(name=embed_field_name, **kws))

            if k in index_kwargs_lookup:
                index_params.add_index(field_name=embed_field_name, **index_kwargs_lookup[k])

        schema = pymilvus.CollectionSchema(fields=field_list, auto_id=False, enable_dynamic_field=False)
        try:
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        except pymilvus.MilvusException as e:
            msg = getattr(e, 'message', str(e))
            if 'invalid index type' in msg.lower():
                if retry >= MILVUS_INDEX_MAX_RETRY:
                    LOG.error(f'[Milvus Store] index fallback exceeded max retries ({MILVUS_INDEX_MAX_RETRY}),'
                              f' last error: {msg}')
                    raise
                try:
                    wrong_index_type = msg.split('invalid index type: ')[1]
                    if ',' in wrong_index_type:
                        wrong_index_type = wrong_index_type.split(',')[0].strip()
                except Exception:
                    LOG.error(f'[Milvus Store] failed to parse invalid index type from error: {msg}')
                    raise
                self._ensure_valid_index(self._index_kwargs)
                LOG.warning(f'[Milvus Store] Unsupported index type: {wrong_index_type}. '
                            f'Fallback to AUTOINDEX and retry (try #{retry + 1}).')
                self._create_collection(client, collection_name, embed_kwargs, retry=retry + 1)
            else:
                raise e

    def _ensure_valid_index(self, index_params: Union[list, dict]):
        embed_index_map = {
            DataType.FLOAT_VECTOR: ['FLAT', 'HNSW', 'IVF_FLAT', 'IVF_SQ8', 'IVF_PQ', 'AUTOINDEX', 'DISKANN'],
            DataType.SPARSE_FLOAT_VECTOR: ['SPARSE_INVERTED_INDEX', 'SPARSE_WAND'],
            DataType.VARCHAR: ['INVERTED_INDEX'],
            DataType.STRING: ['INVERTED_INDEX'],
            DataType.ARRAY: ['INVERTED_INDEX'],
            DataType.INT32: ['INVERTED_INDEX'],
            DataType.INT64: ['INVERTED_INDEX'],
            DataType.FLOAT: ['INVERTED_INDEX'],
            DataType.BOOLEAN: ['INVERTED_INDEX'],
        }

        def _replace_index_type(index_item: dict):
            '''
            Raise ValueError if the DataType is not supported by Milvus.
            Raise ValueError if the IndexType is not compatible with the DataType.
            Fallback to the default index type if the IndexType is compatible with the DataType
            but not supported by Milvus.
            '''
            embed_key = index_item.get('embed_key')
            dtype = self._embed_datatypes.get(embed_key)
            index_type = index_item.get('index_type').upper()
            if dtype not in embed_index_map:
                raise ValueError(f'[Milvus Store]: Unsupported data type: {DataType(dtype).name}.')
            if index_type not in embed_index_map.get(dtype):
                raise ValueError(f'[Milvus Store] {DataType(dtype).name}: Unsupported index type: {index_type}.')
            else:
                index_type = list(embed_index_map.get(dtype))[0]
                index_item['index_type'] = index_type
                self._ensure_params_defaults(index_item)

        if isinstance(index_params, list):
            for index_item in index_params:
                _replace_index_type(index_item)
        elif isinstance(index_params, dict):
            _replace_index_type(index_params)

    def _ensure_params_defaults(self, index_item: dict):
        '''
        Fill in the missing fields (index_type, metric_type, params) of a single index item.
        Do not override the fields explicitly provided by the user (only setdefault)
        params will be filled in with common defaults based on index_type
        (if params already exist, only fill in missing keys)
        '''
        if not isinstance(index_item, dict):
            return

        # Normalize index_type
        itype = index_item.get('index_type')
        if itype:
            itype_up = str(itype).upper()
            index_item['index_type'] = itype_up
        else:
            raise ValueError(f'cannot find `index_type` in `index_kwargs` of `{index_item}`')

        defaults = MILVUS_INDEX_TYPE_DEFAULTS.get(index_item['index_type'], None)
        if defaults is None:
            raise ValueError(f'[Milvus Store] Unsupported index type: {index_item["index_type"]}')

        # metric_type default fill (do not override user)
        if 'metric_type' not in index_item and 'metric_type' in defaults:
            index_item['metric_type'] = defaults['metric_type']

        default_params = defaults.get('params', {})
        if 'params' not in index_item or index_item.get('params') is None:
            index_item['params'] = dict(default_params)
        else:
            # fill in the missing keys of params
            if isinstance(index_item['params'], dict):
                for k, v in default_params.items():
                    index_item['params'].setdefault(k, v)
            else:
                # if user passed a non-dict (exception), replace it with the default dict
                index_item['params'] = dict(default_params)

    def _serialize_data(self, d: dict) -> dict:
        # only keep primary_key, embedding and global_meta
        res = {
            self._primary_key: d.get(self._primary_key, '')
        }
        for embed_key, value in d.get('embedding', {}).items():
            # set default value for SPARSE_FLOAT_VECTOR type
            if self._embed_datatypes.get(embed_key) == DataType.SPARSE_FLOAT_VECTOR:
                value = value or MILVUS_SPARSE_FLOAT_VECTOR_DEFAULT_VALUE
            res[self._gen_embed_key(embed_key)] = value
        global_meta = d.get('global_meta', {})
        for name, desc in self._global_metadata_desc.items():
            value = global_meta.get(name, desc.default_value)
            if value is not None:
                res[self._gen_global_meta_key(name)] = value
        return res

    def _deserialize_data(self, d: dict) -> dict:
        res = {
            self._primary_key: d.get(self._primary_key, ''),
            'embedding': {}
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
        res = {}
        criteria = dict(criteria)
        if self._primary_key in criteria:
            res['ids'] = criteria[self._primary_key]
        else:
            filter_str = ''
            for key, vaule in criteria.items():
                if key not in self._global_metadata_desc:
                    continue
                field_name = self._gen_global_meta_key(key)
                if len(filter_str) > 0:
                    filter_str += ' and '
                if isinstance(vaule, list):
                    filter_str += f'{field_name} in {vaule}'
                elif isinstance(vaule, str):
                    filter_str += f'{field_name} == "{vaule}"'
                else:
                    raise ValueError(f'invalid criteria type: {type(vaule)}')
            res['filter'] = filter_str
        return res

    @override
    def search(self, collection_name: str, query_embedding: Union[dict, List[float]], topk: int,
               filters: Optional[Dict[str, Union[List, set]]] = None, embed_key: Optional[str] = None,
               filter_str: Optional[str] = '', **kwargs) -> List[dict]:
        with self._client_context() as client:
            if not embed_key or embed_key not in self._embed_datatypes:
                raise ValueError(f'[Milvus Store - search] Not supported or None `embed_key`: {embed_key}')
            if not client.has_collection(collection_name):
                return []
            client.load_collection(collection_name)

            res = []
            filter_expr = self._construct_filter_expr(filters) if filters else ''
            if filter_str:
                filter_expr = f'{filter_expr} and {filter_str}' if filter_expr else filter_str

            results = client.search(collection_name=collection_name, data=[query_embedding], limit=topk,
                                    anns_field=self._gen_embed_key(embed_key),
                                    filter=filter_expr)
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
        ret_str = ''
        if not filters:
            return ret_str
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            key = self._gen_global_meta_key(name)
            if isinstance(candidates, str):
                candidates = [candidates]
            elif (not isinstance(candidates, list)) and (not isinstance(candidates, set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '
        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '
        return ret_str

    def validate_milvus_embed_keys(self, index_kwargs: Optional[Union[List, Dict]]):  # noqa: C901
        '''
        Validate and preprocess the index_kwargs of milvus store_conf:
        1. Auto fill the only one missing embed_key into the configuration without embed_key;
        2. The embed_key in self._embed must be a subset of the embed_key in store_conf;
        3. store_conf can contain additional embed_key;
        4. Duplicate embed_key is forbidden;
        5. If multiple embed_key are missing, raise an error.
        '''
        if not isinstance(index_kwargs, (list, dict)):
            raise TypeError(f'[Milvus Store] index_kwargs must be a list or dict, but got {type(index_kwargs)}')

        embed_keys = list(self._embed_datatypes.keys())
        if not embed_keys:
            raise ValueError('self._embed is empty, cannot build index configuration')

        normalized_index_kwargs = []
        no_embedkey_entries = []
        seen = set()
        if isinstance(index_kwargs, dict):
            index_kwargs = [index_kwargs]
        for i, idx_conf in enumerate(index_kwargs):
            if not isinstance(idx_conf, dict):
                raise TypeError(f'index_kwargs position {i} must be a dictionary, but got {type(idx_conf)}')

            embed_key = idx_conf.get('embed_key')
            if embed_key:
                if embed_key in seen:
                    raise ValueError(f'duplicate embed_key {embed_key} in index_kwargs position {i}')
                seen.add(embed_key)
            else:
                no_embedkey_entries.append((i, idx_conf))

            normalized_index_kwargs.append(idx_conf)

        store_embed_keys = seen
        missing_keys = set(embed_keys) - store_embed_keys

        if len(missing_keys) > 1:
            raise ValueError(
                f'[Milvus Store] store_conf is missing the following embed_key: {missing_keys} '
                f'(only supports auto filling one missing item)'
            )
        elif len(missing_keys) == 1:
            missing_key = next(iter(missing_keys))

            if len(no_embedkey_entries) == 1:
                idx = no_embedkey_entries[0][1]
                idx['embed_key'] = missing_key
            elif len(no_embedkey_entries) == 0:
                if self._embed_datatypes.get(missing_key) == DataType.FLOAT_VECTOR:
                    normalized_index_kwargs.append({
                        'embed_key': missing_key,
                        'index_type': 'FLAT',
                        'metric_type': 'COSINE'
                    })
                else:
                    normalized_index_kwargs.append({
                        'embed_key': missing_key,
                        'index_type': 'SPARSE_INVERTED_INDEX',
                        'metric_type': 'L2'
                    })
            else:
                raise ValueError(
                    f'[Milvus Store] Found multiple entries without embed_key, cannot determine '
                    f'which one to fill. Missing embed_keys: {missing_keys}'
                )

        return normalized_index_kwargs
