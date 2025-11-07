import copy
import threading
import traceback
from contextlib import contextmanager
from queue import Queue, Empty, Full
from typing import Dict, List, Union, Optional

from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.mysql import TEXT, LONGTEXT

from pyobvector import ObVecClient, ARRAY, VECTOR, SPARSE_VECTOR, RangeListPartInfo
from pyobvector.client.index_param import VecIndexType
from pyobvector.client.fts_index_param import FtsIndexParam, FtsParser
from pyobvector import inner_product, l2_distance, cosine_distance

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc
from lazyllm.tools.rag.store.store_base import LazyLLMStoreBase, StoreCapability, GLOBAL_META_KEY_PREFIX, EMBED_PREFIX

# Supported index types mapping
OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES = {
    'HNSW': VecIndexType.HNSW,
    'HNSW_SQ': VecIndexType.HNSW_SQ,
    'IVF': VecIndexType.IVFFLAT,  # Use IVFFLAT as default IVF implementation
    'IVF_FLAT': VecIndexType.IVFFLAT,
    'IVF_SQ': VecIndexType.IVFSQ,
    'IVF_PQ': VecIndexType.IVFPQ,
    'FLAT': VecIndexType.IVFFLAT,  # FLAT can be implemented as IVFFLAT with nlist=1
}

OCEANBASE_INDEX_TYPE_DEFAULTS = {
    'HNSW': {'metric_type': 'l2', 'params': {'M': 16, 'efConstruction': 200}},
    'HNSW_SQ': {'metric_type': 'l2', 'params': {'M': 16, 'efConstruction': 200}},
    'IVF': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_FLAT': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_SQ': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_PQ': {'metric_type': 'l2', 'params': {'nlist': 128, 'm': 8, 'nbits': 8}},  # params m is must needed for IVF_PQ
    'FLAT': {'metric_type': 'l2', 'params': {}},
}


OB_UPSERT_BATCH_SIZE = 500

DEFAULT_OCEANBASE_PAGINATION_OFFSET = 1000

class _ClientPool:
    def __init__(self, maker, max_size: int = 8):
        self._q = Queue(maxsize=max_size)
        self._maker = maker
        self._max_size = max_size
        self._current_size = 0
        self._lock = threading.Lock()

    def acquire(self):
        try:
            return self._q.get_nowait()
        except Empty:
            with self._lock:
                if self._current_size < self._max_size:
                    self._current_size += 1
                    return self._maker()
            # Increase timeout for SSH tunnel connections
            return self._q.get(timeout=60)

    def release(self, c):
        try:
            self._q.put_nowait(c)
        except Full:
            try:
                c.close()
                with self._lock:
                    self._current_size -= 1
            except Exception:
                pass


class OceanBaseStore(LazyLLMStoreBase):
    capability = StoreCapability.ALL
    need_embedding = True
    supports_index_registration = True

    def __init__(self, uri: str = '127.0.0.1:2881', user: str = 'root@test', password: str = '',
                 db_name: str = 'test', drop_old: bool = False, index_kwargs: Optional[Union[Dict, List]] = None,
                 client_kwargs: Optional[Dict] = None, max_pool_size: int = 8):
        self._uri = uri
        self._user = self._parse_user(user)
        self._password = password
        self._db_name = db_name
        self._index_kwargs = index_kwargs or {}
        self._client_kwargs = client_kwargs or {}
        self._ddl_lock = threading.Lock()
        self._db_ready = False
        self._drop_old = drop_old
        self._dropped_tables = set()
        self._embed_dims: Dict[str, int] = {}
        self._embed_datatypes: Union[Dict[str, DataType], Dict[str, Dict]] = {}
        self._global_metadata_desc: Dict[str, GlobalMetadataDesc] = {}
        self._primary_key = 'uid'

    @contextmanager
    def _client_context(self) -> ObVecClient:
        c = self._client_pool.acquire()
        try:
            yield c
        finally:
            self._client_pool.release(c)

    def _new_client(self):
        kwargs = dict(self._client_kwargs)
        try:
            # ObVecClient support vector search only or fulltext search only
            # TODO: enable Hybrid search with HybridSearch()
            c = ObVecClient(uri=self._uri, user=self._user, password=self._password, db_name=self._db_name, **kwargs)
            LOG.info(f'[OceanBaseStore] Successfully connected to {self._uri}')
            return c
        except Exception as e:
            LOG.error(f'[OceanBaseStore - _new_client] error: {e}')
            raise e

    @override
    def connect(self, embed_dims: Optional[Dict[str, int]] = None,
                embed_datatypes: Optional[Dict[str, DataType]] = None,
                global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs):
        self._embed_dims = embed_dims or {}
        self._embed_datatypes = embed_datatypes or {}
        self._global_metadata_desc = global_metadata_desc or {}
        self._set_constants()

        self._db_ready = False
        self._ensure_database()

        max_pool_size = int(self._client_kwargs.get('max_pool_size', 8))
        self._client_pool = _ClientPool(self._new_client, max_size=max_pool_size)
        LOG.info('[OceanBaseStore] init success!')

    def upsert(self, collection_name: str, data: List[dict], range_part: Optional[RangeListPartInfo] = None, **kwargs) -> bool:  # noqa: C901 E501
        try:
            if not data: return True

            all_embed_keys = set()
            for item in data:
                if 'embedding' in item and isinstance(item['embedding'], dict):
                    all_embed_keys.update(item['embedding'].keys())

            with self._client_context() as client:
                with self._ddl_lock:
                    if self._drop_old and collection_name not in self._dropped_tables:
                        self._dropped_tables.add(collection_name)
                        client.drop_table_if_exist(collection_name)
                    if not client.check_table_exists(collection_name):
                        embed_kwargs = {}
                        if all_embed_keys:
                            for embed_key in all_embed_keys:
                                assert self._embed_datatypes.get(embed_key), \
                                    f'cannot find embedding params for embed [{embed_key}]'
                                if embed_key not in embed_kwargs:
                                    embed_kwargs[embed_key] = {
                                        'dtype': self._type2oceanbase[self._embed_datatypes[embed_key]]
                                    }
                                if self._embed_dims.get(embed_key):
                                    embed_kwargs[embed_key]['dim'] = self._embed_dims[embed_key]

                        self._create_table_and_index(client, collection_name, embed_kwargs, range_part)

                for i in range(0, len(data), OB_UPSERT_BATCH_SIZE):
                    client.upsert(table_name=collection_name,
                                  data=[self._serialize_data(d) for d in data[i:i + OB_UPSERT_BATCH_SIZE]])
            return True
        except Exception as e:
            LOG.error(f'[OceanBaseStore - upsert] error: {e}')
            return False

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            with self._client_context() as client:
                if not client.check_table_exists(collection_name):
                    return True

                if not criteria:
                    with self._ddl_lock:
                        client.drop_table_if_exist(table_name=collection_name)
                else:
                    ids = None
                    where_clause = None

                    if self._primary_key in criteria:
                        ids = criteria[self._primary_key]
                        if isinstance(ids, str):
                            ids = [ids]
                    else:
                        where_clause = self._construct_where_clause(criteria)

                    client.delete(
                        table_name=collection_name,
                        ids=ids,
                        where_clause=where_clause,
                        **kwargs
                    )

            return True
        except Exception as e:
            LOG.error(f'[OceanBaseStore - delete] error: {e}')
            LOG.error(traceback.format_exc())
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        try:
            with self._client_context() as client:
                if not client.check_table_exists(collection_name):
                    return []

                ids, where_clause = None, None
                if criteria:
                    if self._primary_key in criteria:
                        ids = (
                            [criteria[self._primary_key]]
                            if isinstance(criteria[self._primary_key], str)
                            else criteria[self._primary_key]
                        )
                    else:
                        where_clause = self._construct_where_clause(criteria)

                # If no criteria, need to fetch all data with pagination
                if ids is None and where_clause is None:
                    return self._get_all_with_pagination(client, collection_name)

                res = client.get(
                    table_name=collection_name,
                    ids=ids,
                    where_clause=where_clause,
                    output_column_name=None,
                    **kwargs
                )

                if not res:
                    return []
                result = [r._mapping for r in res]

                return [self._deserialize_data(r) for r in result]

        except Exception as e:
            LOG.error(f'[OceanBaseStore - get] error: {e}')
            LOG.error(traceback.format_exc())
            return []

    def _get_all_with_pagination(self, client, collection_name: str) -> List[dict]:
        all_results = []
        offset = 0
        batch_size = DEFAULT_OCEANBASE_PAGINATION_OFFSET

        while True:
            try:
                # Use raw SQL to query with LIMIT and OFFSET
                sql = f'SELECT * FROM {collection_name} LIMIT {batch_size} OFFSET {offset}'
                batch_res = client.perform_raw_text_sql(sql)

                if not batch_res:
                    break

                if hasattr(batch_res, 'fetchall'):
                    rows = batch_res.fetchall()
                    if not rows:
                        break
                    columns = batch_res.keys() if hasattr(batch_res, 'keys') else []
                    batch_results = [dict(zip(columns, row)) for row in rows]
                else:
                    batch_results = [r._mapping if hasattr(r, '_mapping') else dict(r) for r in batch_res]

                if not batch_results:
                    break

                all_results.extend([self._deserialize_data(r) for r in batch_results])

                if len(batch_results) < batch_size:
                    break

                offset += batch_size

            except Exception as e:
                LOG.error(f'[OceanBaseStore - _get_all_with_pagination] error at offset {offset}: {e}')
                LOG.error(traceback.format_exc())
                break

        return all_results

    def search(self, collection_name: str, query_embedding: Union[dict, List[float]], topk: int, filters: Optional[Dict[str, Union[List, set]]] = None, embed_key: Optional[str] = None, filter_str: Optional[str] = '', **kwargs) -> List[dict]:  # noqa: C901 E501
        try:
            with self._client_context() as client:
                if not client.check_table_exists(collection_name):
                    return []
                if embed_key and embed_key not in self._embed_datatypes:
                    raise ValueError(f'[OceanBaseStore - search] `embed_key`: {embed_key} not exsits')

                if not embed_key:
                    if self._embed_datatypes:
                        embed_key = next(iter(self._embed_datatypes.keys()))
                        LOG.info(f'[OceanBaseStore - search] No embed_key provided, using default: {embed_key}')
                    else:
                        raise ValueError('[OceanBaseStore - search] No embedding datatypes available')

                where_clause = None
                if filters or filter_str:
                    filter_parts = []
                    if filters:
                        filter_expr = self._construct_filter_expr(filters)
                        if filter_expr:
                            filter_parts.append(filter_expr)
                    if filter_str:
                        filter_parts.append(filter_str)

                    if filter_parts:
                        combined_filter = ' and '.join(f'({part})' for part in filter_parts)
                        where_clause = [text(combined_filter)]

                if (
                    isinstance(query_embedding, dict)
                    and self._embed_datatypes.get(embed_key) != DataType.SPARSE_FLOAT_VECTOR
                ):
                    vec_data = query_embedding.get(embed_key)
                    if vec_data is None:
                        raise ValueError(
                            f'[OceanBaseStore - search] `embed_key`: {embed_key} not found in query_embedding'
                        )
                else:
                    vec_data = query_embedding

                if self._embed_datatypes[embed_key] == DataType.SPARSE_FLOAT_VECTOR:
                    if isinstance(vec_data, dict):
                        vec_data = {int(k) if isinstance(k, str) else k: v for k, v in vec_data.items()}
                    distance_func = inner_product
                else:
                    metric_type = self._get_metric_type_for_embed_key(embed_key)
                    distance_func = self._get_distance_function(metric_type)

                results = client.ann_search(
                    table_name=collection_name,
                    vec_data=vec_data,
                    vec_column_name=self._gen_embed_key(embed_key),
                    distance_func=distance_func,
                    topk=topk,
                    with_dist=True,
                    output_column_names=None,
                    where_clause=where_clause,
                    **kwargs
                )

                res = []
                if not results:
                    return []

                for row in results:
                    if hasattr(row, '_mapping'):
                        row_dict = row._mapping
                        uid = row_dict.get(self._primary_key, '')
                        score = row_dict.get('distance', 0)
                    elif isinstance(row, dict):
                        uid = row.get('id', row.get(self._primary_key, ''))
                        score = row.get('distance', 0)
                    elif isinstance(row, (tuple, list)):
                        if len(row) >= 2:
                            uid = row[0]
                            score = row[1]
                        elif len(row) == 1:
                            uid = row[0]
                            score = 0
                        else:
                            continue
                    else:
                        continue

                    if uid:
                        res.append({'uid': uid, 'score': score})

                return res
        except Exception as e:
            LOG.error(f'[OceanBaseStore - search] error: {e}')
            LOG.error(traceback.format_exc())
            return []

    def _create_table_and_index(self, client: ObVecClient, collection_name: str,
                                embed_kwargs: Dict, partitions: Optional[RangeListPartInfo] = None
                            ) -> bool:  # noqa: C901
        columns = copy.deepcopy(self._constant_columns)

        idx_params = client.prepare_index_params()
        original_index_kwargs = copy.deepcopy(self._index_kwargs)

        index_kwargs_lookup = {}
        fts_idxs = []
        has_explicit_fts = False

        if isinstance(original_index_kwargs, dict):
            original_index_kwargs = [original_index_kwargs]
        for item in original_index_kwargs:
            embed_key = item.get('embed_key', None)
            if not embed_key:
                has_explicit_fts = True
                field_names = item.get('field_names', ['content'])
                if isinstance(field_names, str):
                    field_names = [field_names]
                index_name = item.get('index_name', f'fts_{field_names[0]}')
                fts_idxs.append(
                    FtsIndexParam(
                        index_name=index_name,
                        field_names=field_names,
                        parser_type=item.get('parser_type', FtsParser.IK),
                    )
                )
                continue

            self._ensure_params_defaults(item)
            index_kwargs_lookup[embed_key] = item.copy()
            index_kwargs_lookup[embed_key].pop('embed_key', None)
        for k, kws in embed_kwargs.items():
            embed_field_name = self._gen_embed_key(k)
            dim = kws.get('dim', None)
            if not dim and kws.get('dtype') == VECTOR:
                raise ValueError(f'embedding `{k}` lack of dim parameter')
            columns.append(
                Column(embed_field_name, kws['dtype'](dim))
                if dim else Column(embed_field_name, kws['dtype'])
            )

            if k in index_kwargs_lookup:
                index_item = index_kwargs_lookup[k]
                if kws['dtype'] == VECTOR:
                    idx_params.add_index(
                        field_name=embed_field_name,
                        index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPES[index_item['index_type']],
                        metric_type=index_item['metric_type'],
                        index_name=f'vidx_{k}',
                        params=index_item.get('params', {})
                    )
                else:
                    idx_params.add_index(
                        field_name=embed_field_name,
                        index_type='daat',
                        index_name=f'vidx_{k}',
                        metric_type='inner_product',
                    )

        if not has_explicit_fts:
            fts_idxs.append(
                FtsIndexParam(
                    index_name='fts_content',
                    field_names=['content'],
                    parser_type=FtsParser.IK,
                )
            )

        try:
            client.create_table_with_index_params(
                table_name=collection_name,
                columns=columns,
                indexes=None,
                vidxs=idx_params,
                fts_idxs=fts_idxs,
                partitions=partitions,
            )
        except Exception as e:
            LOG.error(f'[OceanBaseStore - _create_table_and_index] error: {e}')
            LOG.error(traceback.format_exc())
            raise e
        return True

    def _get_metric_type_for_embed_key(self, embed_key: str) -> str:

        if isinstance(self._index_kwargs, dict):
            index_kwargs_list = [self._index_kwargs]
        else:
            index_kwargs_list = self._index_kwargs or []

        for index_kwarg in index_kwargs_list:
            if index_kwarg.get('embed_key') == embed_key:
                return index_kwarg.get('metric_type', 'l2')
        return 'l2'

    def _get_distance_function(self, metric_type: str):
        metric_type = metric_type.lower()
        if metric_type == 'inner_product':
            return inner_product
        elif metric_type == 'l2':
            return l2_distance
        elif metric_type == 'cosine':
            return cosine_distance
        else:
            raise ValueError(f'Unsupported metric type: {metric_type}')

    def _serialize_data(self, d: dict) -> dict:
        res = {
            self._primary_key: d.get(self._primary_key, '')
        }
        # content is reserved for fulltext index
        if 'content' in d:
            res['content'] = d['content']
        for embed_key, value in d.get('embedding', {}).items():
            # Convert sparse vector dict keys from str to int if needed
            if self._embed_datatypes.get(embed_key) == DataType.SPARSE_FLOAT_VECTOR:
                if isinstance(value, dict):
                    # Convert string keys to integers for sparse vectors
                    value = {int(k) if isinstance(k, str) else k: v for k, v in value.items()}
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
            'embedding': {},
            'global_meta': {}
        }
        if 'content' in d:
            res['content'] = d['content']
        for k, v in d.items():
            if k.startswith(EMBED_PREFIX):
                res['embedding'][k[len(EMBED_PREFIX):]] = v
            elif k.startswith(GLOBAL_META_KEY_PREFIX):
                meta_key = k[len(GLOBAL_META_KEY_PREFIX):]
                res['global_meta'][meta_key] = v
        return res

    def _gen_global_meta_key(self, k: str) -> str:
        return GLOBAL_META_KEY_PREFIX + k

    def _gen_embed_key(self, k: str) -> str:
        return EMBED_PREFIX + k

    def _ensure_database(self):
        DB_USER = self._user
        DB_PASSWORD = self._password
        uri_parts = self._uri.split(':')
        if len(uri_parts) < 2:
            raise ValueError(f'Invalid URI format: {self._uri}. Expected format: host:port')
        DB_HOST = uri_parts[0]
        DB_PORT = uri_parts[1]
        DB_NAME = self._db_name

        try:
            engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/', pool_pre_ping=True)

            with engine.connect() as connection:
                LOG.info('Successfully connected to OceanBase database server!')

                result = connection.execute(text('SHOW DATABASES'))
                databases = [row[0] for row in result]

                if DB_NAME in databases:
                    LOG.info(f'Database {DB_NAME} already exists.')
                else:
                    connection.execute(text(f'CREATE DATABASE {DB_NAME}'))
                    LOG.info(f'Database {DB_NAME} created successfully!')

        except Exception as e:
            LOG.error(f'[OceanBaseStore - _ensure_database] Unexpected error: {e}')
            raise
        finally:
            engine.dispose()

    def _parse_user(self, user: str) -> str:
        if ':' in user:
            _, tenant, username = user.split(':')
            return f'{username}@{tenant}'
        elif '#' in user:
            username, tenant_cluster = user.split('@')
            tenant, cluster = tenant_cluster.split('#')
            return f'{username}@{tenant}'
        else:
            return user

    def _set_constants(self):
        self._type2oceanbase = {
            DataType.ARRAY: ARRAY,
            DataType.FLOAT_VECTOR: VECTOR,
            DataType.SPARSE_FLOAT_VECTOR: SPARSE_VECTOR,
            DataType.STRING: TEXT,
            DataType.VARCHAR: String,
            DataType.INT32: Integer,
            DataType.INT64: Integer,
        }
        self._builtin_keys = {
            'uid': {'dtype': String(512), 'primary_key': True, 'autoincrement': False},
            'content': {'dtype': LONGTEXT}
        }
        self._constant_columns = self._get_constant_columns()

    def _get_constant_columns(self) -> list:
        column_list = []
        for k, kws in self._builtin_keys.items():
            kws_copy = dict(kws)
            dtype = kws_copy.pop('dtype')
            column_list.append(Column(k, dtype, **kws_copy))
        for k, desc in self._global_metadata_desc.items():
            field_name = self._gen_global_meta_key(k)
            if desc.data_type == DataType.ARRAY:
                if desc.element_type is None:
                    raise ValueError(f'OceanBase field [{field_name}]: '
                                     '`element_type` is required when `data_type` is ARRAY.')
                column_list.append(Column(field_name, ARRAY))
            elif desc.data_type == DataType.VARCHAR:
                column_list.append(Column(field_name, String(desc.max_size)))
            else:
                column_list.append(Column(field_name, self._type2oceanbase[desc.data_type]))
        return column_list

    def _ensure_params_defaults(self, index_item: dict):
        itype = index_item.get('index_type')
        if itype:
            itype_up = str(itype).upper()
            index_item['index_type'] = itype_up
        else:
            raise ValueError(f'cannot find `index_type` in `index_kwargs` of `{index_item}`')

        defaults = OCEANBASE_INDEX_TYPE_DEFAULTS.get(index_item['index_type'], None)
        if defaults is None:
            raise ValueError(f'[OceanBase Store] Unsupported index type: {index_item["index_type"]}')

        if 'metric_type' not in index_item and 'metric_type' in defaults:
            index_item['metric_type'] = defaults['metric_type']

        default_params = defaults.get('params', {})
        if 'params' not in index_item or index_item.get('params') is None:
            index_item['params'] = dict(default_params)
        else:
            if isinstance(index_item['params'], dict):
                for k, v in default_params.items():
                    index_item['params'].setdefault(k, v)
            else:
                index_item['params'] = dict(default_params)

    def _construct_where_clause(self, criteria: dict) -> Optional[list]:
        if not criteria:
            return None

        filter_parts = []
        for key, value in criteria.items():
            if key == self._primary_key:
                continue

            if key not in self._global_metadata_desc:
                continue

            field_name = self._gen_global_meta_key(key)

            if isinstance(value, list):
                if not value:
                    continue
                if isinstance(value[0], str):
                    values_str = ', '.join(f'"{v}"' for v in value)
                else:
                    values_str = ', '.join(str(v) for v in value)
                filter_parts.append(f'{field_name} in ({values_str})')
            elif isinstance(value, str):
                filter_parts.append(f'{field_name} = "{value}"')
            elif isinstance(value, (int, float)):
                filter_parts.append(f'{field_name} = {value}')
            else:
                raise ValueError(f'Unsupported criteria value type: {type(value)} for key: {key}')

        if not filter_parts:
            return None

        combined_filter = ' and '.join(filter_parts)
        return [text(combined_filter)]

    def _construct_filter_expr(self, filters: Dict[str, Union[List, set]]) -> str:
        if not filters:
            return ''

        filter_parts = []
        for key, value in filters.items():
            if key not in self._global_metadata_desc.keys():
                continue

            field_name = self._gen_global_meta_key(key)

            if isinstance(value, (list, set)):
                value_list = list(value)
                if not value_list:
                    continue
                if isinstance(value_list[0], str):
                    values_str = ', '.join(f'"{v}"' for v in value_list)
                else:
                    values_str = ', '.join(str(v) for v in value_list)
                filter_parts.append(f'{field_name} in ({values_str})')
            elif isinstance(value, str):
                filter_parts.append(f'{field_name} = "{value}"')
            elif isinstance(value, (int, float)):
                filter_parts.append(f'{field_name} = {value}')
            else:
                LOG.warning(f'[OceanBaseStore] Unsupported filter value type: {type(value)} for key: {key}')
                continue

        return ' and '.join(filter_parts)
