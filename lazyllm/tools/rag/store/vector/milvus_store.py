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
        p = p if p.suffix else (p / "milvus.db")
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

        max_pool_size = int(self._client_kwargs.pop('max_pool_size', 8))
        self._client_pool = _ClientPool(self._new_client, max_size=max_pool_size)
        LOG.info("[Milvus Vector Store] init success!")

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
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        try:
            with self._client_context() as client:
                if not client.has_collection(collection_name):
                    return []
                client.load_collection(collection_name)
                col_desc = client.describe_collection(collection_name=collection_name)
                field_names = [field.get('name') for field in col_desc.get('fields', [])
                               if field.get('name').startswith(EMBED_PREFIX)]
                if criteria and self._primary_key in criteria:
                    res = client.get(collection_name=collection_name, ids=criteria[self._primary_key])
                else:
                    filters = self._construct_criteria(criteria) if criteria else {}
                    if version.parse(pymilvus.__version__) >= version.parse('2.4.11'):
                        iterator = client.query_iterator(collection_name=collection_name,
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
                        res = client.query(collection_name=collection_name, output_fields=field_names, **filters)
            return [self._deserialize_data(r) for r in res]
        except Exception as e:
            LOG.error(f'[Milvus Store - get] error: {e}')
            LOG.error(traceback.format_exc())
            return []

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
        for k, kws in embed_kwargs.items():
            embed_field_name = self._gen_embed_key(k)
            field_list.append(pymilvus.FieldSchema(name=embed_field_name, **kws))
            if isinstance(original_index_kwargs, list):
                for item in original_index_kwargs:
                    embed_key = item.get('embed_key', None)
                    if not embed_key:
                        raise ValueError(f'cannot find `embed_key` in `index_kwargs` of `{item}`')
                    if embed_key == k:
                        index_kwarg = item.copy()
                        index_kwarg.pop('embed_key', None)
                        index_params.add_index(field_name=embed_field_name, **index_kwarg)
                        break
            elif isinstance(original_index_kwargs, dict):
                index_params.add_index(field_name=embed_field_name, **original_index_kwargs)
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
                self._replace_index_type_to_autoindex(wrong_index_type)
                LOG.warning(f'[Milvus Store] Unsupported index type: {wrong_index_type}. '
                            f'Fallback to AUTOINDEX and retry (try #{retry + 1}).')
                self._create_collection(client, collection_name, embed_kwargs, retry=retry + 1)
            else:
                raise e

    def _replace_index_type_to_autoindex(self, index_type: str):
        if index_type == 'AUTOINDEX':
            raise ValueError(f'[Milvus Store - replace_index_type_to_autoindex] Invalid index type: {index_type}')
        if isinstance(self._index_kwargs, list):
            for item in self._index_kwargs:
                if item.get('index_type') == index_type:
                    item['index_type'] = 'AUTOINDEX'
        elif isinstance(self._index_kwargs, dict):
            if self._index_kwargs.get('index_type') == index_type:
                self._index_kwargs['index_type'] = 'AUTOINDEX'
        return

    def _serialize_data(self, d: dict) -> dict:
        # only keep primary_key, embedding and global_meta
        res = {
            self._primary_key: d.get(self._primary_key, '')
        }
        for embed_key, value in d.get('embedding', {}).items():
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
