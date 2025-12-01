import copy
import json
import threading
import traceback
from contextlib import contextmanager
from queue import Queue, Empty, Full
from typing import Dict, List, Union, Optional
from lazyllm.thirdparty import numpy as np
import math
import sqlalchemy

from lazyllm.thirdparty import pyobvector

from lazyllm import LOG
from lazyllm.common import override
from ...data_type import DataType
from ...global_metadata import GlobalMetadataDesc
from ..store_base import (LazyLLMStoreBase, StoreCapability,
                          GLOBAL_META_KEY_PREFIX, EMBED_PREFIX, SegmentType)

OCEANBASE_INDEX_TYPE_DEFAULTS = {
    'HNSW': {'metric_type': 'l2', 'params': {'M': 16, 'efConstruction': 200}},
    'HNSW_SQ': {'metric_type': 'l2', 'params': {'M': 16, 'efConstruction': 200}},
    'IVF': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_FLAT': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_SQ': {'metric_type': 'l2', 'params': {'nlist': 128}},
    'IVF_PQ': {'metric_type': 'l2', 'params': {'nlist': 128, 'm': 8, 'nbits': 8}},
    'FLAT': {'metric_type': 'l2', 'params': {}},
}

OB_UPSERT_BATCH_SIZE = 200

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

    def __init__(self, uri: str = '127.0.0.1:2881', db_name: str = 'test',
                 index_kwargs: Optional[Union[Dict, List]] = None, client_kwargs: Optional[Dict] = None):
        self._uri = uri
        self._db_name = db_name
        self._index_kwargs = index_kwargs or {}
        self._client_kwargs = client_kwargs or {}
        self._ddl_lock = threading.Lock()
        self._embed_datatypes: Union[Dict[str, DataType], Dict[str, Dict]] = {}
        self._global_metadata_desc: Dict[str, GlobalMetadataDesc] = {}
        self._primary_key = 'uid'
        self._hnsw_ef_search = {}

    @contextmanager
    def _client_context(self) -> 'pyobvector.ObVecClient':
        c = self._client_pool.acquire()
        try:
            try:
                c.perform_raw_text_sql('SET SESSION ob_query_timeout = 300000000')
            except Exception as e:
                LOG.warning(f'[OceanBaseStore] Failed to set query timeout in context: {e}')
            yield c
        finally:
            self._client_pool.release(c)

    def _new_client(self):
        kwargs = dict(self._client_kwargs)
        try:
            c = pyobvector.ObVecClient(
                uri=self._uri, db_name=self._db_name,
                user=self._user, password=self._password, **kwargs
            )

            try:
                c.perform_raw_text_sql('SET SESSION ob_query_timeout = 300000000')
                result = c.perform_raw_text_sql("SHOW VARIABLES LIKE 'ob_query_timeout'")
                if result:
                    assert result.fetchone() is not None
            except Exception as e:
                LOG.warning(f'[OceanBaseStore] Failed to set/verify query timeout: {e}')

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

        # Extract connection parameters from client_kwargs
        self._user = self._parse_user(self._client_kwargs.pop('user', 'root@test'))
        self._password = self._client_kwargs.pop('password', '')
        self._normalize = self._client_kwargs.pop('normalize', False)
        self._enable_fulltext_index = self._client_kwargs.pop('enable_fulltext_index', False)
        max_pool_size = int(self._client_kwargs.pop('max_pool_size', 8))

        self._ensure_database()
        self._client_pool = _ClientPool(self._new_client, max_size=max_pool_size)
        LOG.info('[OceanBaseStore] init success!')

    def upsert(self, collection_name: str, data: List[dict], range_part: Optional['pyobvector.RangeListPartInfo'] = None, **kwargs) -> bool:  # noqa: C901 E501
        try:
            if not data:
                return True

            if not collection_name or not isinstance(collection_name, str):
                LOG.error('[OceanBaseStore - upsert] Invalid collection_name')
                return False

            all_embed_keys = set()
            for item in data:
                if 'embedding' in item and isinstance(item['embedding'], dict):
                    all_embed_keys.update(item['embedding'].keys())

            with self._client_context() as client:
                with self._ddl_lock:
                    if not client.check_table_exists(collection_name):
                        embed_kwargs = {}
                        if all_embed_keys:
                            for embed_key in all_embed_keys:
                                if not self._embed_datatypes.get(embed_key):
                                    LOG.error(f'[OceanBaseStore - upsert] Cannot find embedding for embed [{embed_key}]')
                                    return False

                                if embed_key not in embed_kwargs:
                                    embed_kwargs[embed_key] = {
                                        'dtype': self._type2oceanbase[self._embed_datatypes[embed_key]]
                                    }
                                if self._embed_dims.get(embed_key):
                                    embed_kwargs[embed_key]['dim'] = self._embed_dims[embed_key]
                                else:
                                    for item in data:
                                        if 'embedding' in item and embed_key in item['embedding']:
                                            emb = item['embedding'][embed_key]
                                            if isinstance(emb, list) and len(emb) > 0:
                                                embed_kwargs[embed_key]['dim'] = len(emb)
                                                break

                        self._create_table_and_index(client, collection_name, embed_kwargs, range_part)

                total_inserted = 0
                failed_batches = []

                serialized_data = [self._serialize_data(d) for d in data]

                for i in range(0, len(serialized_data), OB_UPSERT_BATCH_SIZE):
                    batch_num = i // OB_UPSERT_BATCH_SIZE + 1
                    try:
                        if i == 0 or batch_num % 10 == 0:
                            try:
                                client.perform_raw_text_sql('SET SESSION ob_query_timeout = 300000000')
                            except Exception as timeout_err:
                                LOG.warning(f'[OceanBaseStore - upsert] Failed to set '
                                            f'query timeout for batch {batch_num}: {timeout_err}')

                        batch_data = serialized_data[i:i + OB_UPSERT_BATCH_SIZE]
                        client.upsert(table_name=collection_name, data=batch_data)
                        total_inserted += len(batch_data)

                    except Exception as batch_err:
                        LOG.error(f'[OceanBaseStore - upsert] Failed to insert batch {batch_num}: {batch_err}')
                        LOG.error(f'[OceanBaseStore - upsert] Error details: {traceback.format_exc()}')
                        failed_batches.append(batch_num)
                        continue

                if failed_batches:
                    LOG.warning(f'[OceanBaseStore - upsert] Failed batches: {failed_batches}')
                    if total_inserted == 0:
                        return False
            return True
        except Exception as e:
            LOG.error(f'[OceanBaseStore - upsert] Unexpected error: {e}')
            LOG.error(traceback.format_exc())
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
                    ids, where_clause = self._get_ids_where_clause(criteria)

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

                ids, where_clause = self._get_ids_where_clause(criteria)

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

    def search(self, collection_name: str, query: str, query_embedding: Union[dict, List[float]], topk: int, filters: Optional[Dict[str, Union[List, set]]] = None, embed_key: Optional[str] = None, filter_str: Optional[str] = '', **kwargs) -> List[dict]:  # noqa: C901 E501
        if not query_embedding:
            raise NotImplementedError('Query fulltext search is not supported for now')
        try:
            if not collection_name or not isinstance(collection_name, str):
                LOG.error('[OceanBaseStore - search] Invalid collection_name')
                return []

            with self._client_context() as client:
                if not client.check_table_exists(collection_name):
                    LOG.warning(f'[OceanBaseStore - search] Table {collection_name} does not exist')
                    return []
                if embed_key and embed_key not in self._embed_datatypes:
                    LOG.error(f'[OceanBaseStore - search] embed_key: {embed_key} not exists')
                    return []

                if not embed_key:
                    if self._embed_datatypes:
                        embed_key = next(iter(self._embed_datatypes.keys()))
                        LOG.info(f'[OceanBaseStore - search] No embed_key provided, using default: {embed_key}')
                    else:
                        LOG.error('[OceanBaseStore - search] No embedding datatypes available')
                        return []

                where_clause = None
                if filters or filter_str:
                    filter_parts = []
                    if filters:
                        try:
                            filter_expr = self._construct_filter_expr(filters)
                            if filter_expr:
                                filter_parts.append(filter_expr)
                        except Exception as filter_err:
                            LOG.error(f'[OceanBaseStore - search] Failed to construct filter: {filter_err}')
                            LOG.error(traceback.format_exc())
                            raise RuntimeError(f'Failed to construct filter expression: {filter_err}') from filter_err

                    if filter_str:
                        filter_parts.append(filter_str)

                    if filter_parts:
                        combined_filter = ' and '.join(f'({part})' for part in filter_parts)
                        where_clause = [sqlalchemy.text(combined_filter)]

                if (
                    isinstance(query_embedding, dict)
                    and self._embed_datatypes.get(embed_key) != DataType.SPARSE_FLOAT_VECTOR
                ):
                    vec_data = query_embedding.get(embed_key)
                    if vec_data is None:
                        LOG.error(f'[OceanBaseStore - search] embed_key: {embed_key} not found in query_embedding')
                        return []
                else:
                    vec_data = query_embedding

                if vec_data is None:
                    LOG.error('[OceanBaseStore - search] Vector data is None')
                    return []

                if self._embed_datatypes[embed_key] == DataType.SPARSE_FLOAT_VECTOR:
                    if isinstance(vec_data, dict):
                        vec_data = {int(k) if isinstance(k, str) else k: v for k, v in vec_data.items()}
                    distance_func = pyobvector.inner_product
                else:
                    if self._normalize and isinstance(vec_data, list):
                        vec_data = self._normalize_vector(vec_data)

                    metric_type = self._get_metric_type_for_embed_key(embed_key)
                    distance_func = self._get_distance_function(metric_type)

                search_params = kwargs.get('search_params', {})
                index_type = self._get_index_type_for_embed_key(embed_key)

                if index_type in ['HNSW', 'HNSW_SQ']:
                    ef_search = search_params.get('efSearch', 64)  # Default efSearch
                    if self._hnsw_ef_search.get(embed_key) != ef_search:
                        try:
                            client.set_ob_hnsw_ef_search(ef_search)
                            self._hnsw_ef_search[embed_key] = ef_search
                        except Exception as e:
                            LOG.error(f'[OceanBaseStore - search] Failed to set efSearch: {e}')
                            LOG.error(traceback.format_exc())
                            raise RuntimeError(f'Failed to set efSearch parameter for HNSW index: {e}') from e

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
                    LOG.info('[OceanBaseStore - search] No results found')
                    return []

                for row in results:
                    try:
                        if hasattr(row, '_mapping'):
                            row_dict = dict(row._mapping)
                            score = row_dict.pop('distance', 0)
                        elif isinstance(row, dict):
                            row_dict = dict(row)
                            score = row_dict.pop('distance', 0)
                        else:
                            LOG.warning(f'[OceanBaseStore - search] Unsupported row type: {type(row)}')
                            continue

                        doc_data = self._deserialize_data(row_dict)

                        if not doc_data.get(self._primary_key):
                            LOG.warning('[OceanBaseStore - search] Row has no valid uid')
                            continue

                        doc_data['score'] = float(score)
                        res.append(doc_data)

                    except Exception as row_err:
                        LOG.warning(f'[OceanBaseStore - search] Failed to process row: {row_err}')
                        LOG.warning(traceback.format_exc())
                        continue

                LOG.info(f'[OceanBaseStore - search] Returning {len(res)} results')
                return res
        except Exception as e:
            LOG.error(f'[OceanBaseStore - search] Unexpected error: {e}')
            LOG.error(traceback.format_exc())
            return []

    def _create_table_and_index(self, client: 'pyobvector.ObVecClient', collection_name: str, embed_kwargs: Dict, partitions: Optional['pyobvector.RangeListPartInfo'] = None) -> bool:  # noqa: C901 E501
        columns = copy.deepcopy(self._constant_columns)

        idx_params = client.prepare_index_params()
        original_index_kwargs = copy.deepcopy(self._index_kwargs)

        index_kwargs_lookup = {}
        fts_idxs = []
        has_explicit_fts = False

        if isinstance(original_index_kwargs, dict):
            original_index_kwargs = [original_index_kwargs]

        for item in original_index_kwargs:
            if not isinstance(item, dict):
                LOG.warning(f'[OceanBaseStore - _create_table_and_index] Invalid index_kwargs item: {item}')
                continue

            embed_key = item.get('embed_key', None)
            if not embed_key:
                has_explicit_fts = True
                field_names = item.get('field_names', ['content'])
                if isinstance(field_names, str):
                    field_names = [field_names]
                index_name = item.get('index_name', f'fts_{field_names[0]}')
                fts_idxs.append(
                    pyobvector.client.fts_index_param.FtsIndexParam(
                        index_name=index_name,
                        field_names=field_names,
                        parser_type=item.get('parser_type', pyobvector.client.fts_index_param.FtsParser.IK),
                    )
                )
                continue

            self._ensure_params_defaults(item)
            index_kwargs_lookup[embed_key] = item.copy()
            index_kwargs_lookup[embed_key].pop('embed_key', None)

        for k, kws in embed_kwargs.items():
            embed_field_name = self._gen_embed_key(k)
            dim = kws.get('dim', None)
            dtype = kws.get('dtype')

            if not dtype:
                LOG.error(f'[OceanBaseStore - _create_table_and_index] No dtype specified for embed_key: {k}')
                raise ValueError(f'No dtype specified for embed_key: {k}')

            if dtype == pyobvector.VECTOR and not dim:
                raise ValueError(f'Embedding `{k}` lacks dim parameter (required for VECTOR type)')

            if dim:
                columns.append(sqlalchemy.Column(embed_field_name, dtype(dim)))
            else:
                columns.append(sqlalchemy.Column(embed_field_name, dtype))

            if k in index_kwargs_lookup:
                index_item = index_kwargs_lookup[k]
                index_type_str = index_item.get('index_type', 'HNSW')

                if index_type_str not in self._oceanbase_supported_vector_index_types:
                    LOG.warning(f'[OceanBaseStore - _create_table_and_index] Unsupported index type: {index_type_str}')
                    continue

                if dtype == pyobvector.VECTOR:
                    idx_params.add_index(
                        field_name=embed_field_name,
                        index_type=self._oceanbase_supported_vector_index_types[index_type_str],
                        metric_type=index_item.get('metric_type', 'l2'),
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
                    LOG.info(f'[OceanBaseStore - _create_table_and_index] Added DAAT index for sparse vector {k}')

            if self._enable_fulltext_index and not has_explicit_fts:
                fts_idxs.append(
                    pyobvector.client.fts_index_param.FtsIndexParam(
                        index_name='fts_content',
                        field_names=['content'],
                        parser_type=pyobvector.client.fts_index_param.FtsParser.IK,
                    )
                )

        try:
            client.create_table_with_index_params(
                table_name=collection_name,
                columns=columns,
                indexes=None,
                vidxs=idx_params,
                fts_idxs=fts_idxs if fts_idxs else None,
                partitions=partitions,
            )

            LOG.info(f'[OceanBaseStore - _create_table_and_index] Table {collection_name} created successfully')
            return True

        except Exception as e:
            LOG.error(f'[OceanBaseStore - _create_table_and_index] Failed to create table {collection_name}: {e}')
            LOG.error(traceback.format_exc())
            raise e

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
            return pyobvector.inner_product
        elif metric_type == 'l2':
            return pyobvector.l2_distance
        elif metric_type == 'cosine':
            return pyobvector.cosine_distance
        else:
            raise ValueError(f'Unsupported metric type: {metric_type}')

    def _get_index_type_for_embed_key(self, embed_key: str) -> str:
        if isinstance(self._index_kwargs, dict):
            index_kwargs_list = [self._index_kwargs]
        else:
            index_kwargs_list = self._index_kwargs or []

        for index_kwarg in index_kwargs_list:
            if index_kwarg.get('embed_key') == embed_key:
                return index_kwarg.get('index_type', 'HNSW').upper()
        return 'HNSW'

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        try:
            arr = np.array(vector)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        except ImportError:
            norm = math.sqrt(sum(x * x for x in vector))
            if norm > 0:
                return [x / norm for x in vector]
            return vector

    def _serialize_data(self, d: dict) -> dict:
        meta = d.get('meta', {})
        meta_str = meta if isinstance(meta, str) else json.dumps(meta, ensure_ascii=False) if meta else '{}'

        image_keys = d.get('image_keys', [])
        image_keys_str = (
            image_keys if isinstance(image_keys, str)
            else json.dumps(image_keys, ensure_ascii=False) if image_keys
            else '[]'
        )

        excluded_embed = d.get('excluded_embed_metadata_keys', [])
        excluded_embed_str = (
            excluded_embed if isinstance(excluded_embed, str)
            else json.dumps(excluded_embed, ensure_ascii=False) if excluded_embed
            else '[]'
        )

        excluded_llm = d.get('excluded_llm_metadata_keys', [])
        excluded_llm_str = (
            excluded_llm if isinstance(excluded_llm, str)
            else json.dumps(excluded_llm, ensure_ascii=False) if excluded_llm
            else '[]'
        )

        res = {
            self._primary_key: d.get(self._primary_key, ''),
            'doc_id': d.get('doc_id', ''),
            'group': d.get('group', ''),
            'content': d.get('content', ''),
            'meta': meta_str,
            'type': d.get('type', SegmentType.TEXT.value),
            'number': d.get('number', 0),
            'kb_id': d.get('kb_id', ''),
            'parent': d.get('parent', ''),
            'answer': d.get('answer', ''),
            'image_keys': image_keys_str,
            'excluded_embed_metadata_keys': excluded_embed_str,
            'excluded_llm_metadata_keys': excluded_llm_str,
        }

        for embed_key, value in d.get('embedding', {}).items():
            if self._embed_datatypes.get(embed_key) == DataType.SPARSE_FLOAT_VECTOR:
                if isinstance(value, dict):
                    value = {int(k) if isinstance(k, str) else k: v for k, v in value.items()}
            else:
                if self._normalize and isinstance(value, list):
                    value = self._normalize_vector(value)

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
            'doc_id': d.get('doc_id', ''),
            'group': d.get('group', ''),
            'content': d.get('content', ''),
            'meta': json.loads(d.get('meta', '{}')) if isinstance(d.get('meta'), str) else (d.get('meta') or {}),
            'type': d.get('type', SegmentType.TEXT.value),
            'number': d.get('number', 0),
            'kb_id': d.get('kb_id', ''),
            'parent': d.get('parent', ''),
            'answer': d.get('answer', ''),
            'image_keys': (
                json.loads(d.get('image_keys', '[]')) if isinstance(d.get('image_keys'), str)
                else (d.get('image_keys') or [])
            ),
            'excluded_embed_metadata_keys': (
                json.loads(d.get('excluded_embed_metadata_keys', '[]')) if isinstance(d.get('excluded_embed_metadata_keys'), str)  # noqa: E501
                else (d.get('excluded_embed_metadata_keys') or [])
            ),
            'excluded_llm_metadata_keys': (
                json.loads(d.get('excluded_llm_metadata_keys', '[]')) if isinstance(d.get('excluded_llm_metadata_keys'), str)  # noqa: E501
                else (d.get('excluded_llm_metadata_keys') or [])
            ),
            'embedding': {},
            'global_meta': {}
        }

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
            engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/', pool_pre_ping=True)  # noqa: E501

            with engine.connect() as connection:
                LOG.info('Successfully connected to OceanBase database server!')

                result = connection.execute(sqlalchemy.text('SHOW DATABASES'))
                databases = [row[0] for row in result]

                if DB_NAME in databases:
                    LOG.info(f'Database {DB_NAME} already exists.')
                else:
                    connection.execute(sqlalchemy.text(f'CREATE DATABASE {DB_NAME}'))
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
        self._oceanbase_supported_vector_index_types = {
            'HNSW': pyobvector.client.index_param.VecIndexType.HNSW,
            'HNSW_SQ': pyobvector.client.index_param.VecIndexType.HNSW_SQ,
            'IVF': pyobvector.client.index_param.VecIndexType.IVFFLAT,
            'IVF_FLAT': pyobvector.client.index_param.VecIndexType.IVFFLAT,
            'IVF_SQ': pyobvector.client.index_param.VecIndexType.IVFSQ,
            'IVF_PQ': pyobvector.client.index_param.VecIndexType.IVFPQ,
            'FLAT': pyobvector.client.index_param.VecIndexType.IVFFLAT,
        }
        self._type2oceanbase = {
            DataType.ARRAY: pyobvector.ARRAY,
            DataType.FLOAT_VECTOR: pyobvector.VECTOR,
            DataType.SPARSE_FLOAT_VECTOR: pyobvector.SPARSE_VECTOR,
            DataType.STRING: sqlalchemy.dialects.mysql.TEXT,
            DataType.VARCHAR: sqlalchemy.String,
            DataType.INT32: sqlalchemy.Integer,
            DataType.INT64: sqlalchemy.Integer,
        }
        self._builtin_keys = {
            'uid': {'dtype': sqlalchemy.String(512), 'primary_key': True, 'autoincrement': False},
            'doc_id': {'dtype': sqlalchemy.String(512)},
            'group': {'dtype': sqlalchemy.String(512)},
            'content': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
            'meta': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
            'type': {'dtype': sqlalchemy.Integer},
            'number': {'dtype': sqlalchemy.Integer},
            'kb_id': {'dtype': sqlalchemy.String(512)},
            'parent': {'dtype': sqlalchemy.String(512)},
            'answer': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
            'image_keys': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
            'excluded_embed_metadata_keys': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
            'excluded_llm_metadata_keys': {'dtype': sqlalchemy.dialects.mysql.LONGTEXT},
        }
        self._constant_columns = self._get_constant_columns()

    def _get_constant_columns(self) -> list:
        column_list = []
        for k, kws in self._builtin_keys.items():
            kws_copy = dict(kws)
            dtype = kws_copy.pop('dtype')
            column_list.append(sqlalchemy.Column(k, dtype, **kws_copy))
        for k, desc in self._global_metadata_desc.items():
            field_name = self._gen_global_meta_key(k)
            if desc.data_type == DataType.ARRAY:
                if desc.element_type is None:
                    raise ValueError(f'OceanBase field [{field_name}]: '
                                     '`element_type` is required when `data_type` is ARRAY.')
                column_list.append(sqlalchemy.Column(field_name, pyobvector.ARRAY))
            elif desc.data_type == DataType.VARCHAR:
                column_list.append(sqlalchemy.Column(field_name, sqlalchemy.String(desc.max_size)))
            else:
                column_list.append(sqlalchemy.Column(field_name, self._type2oceanbase[desc.data_type]))
        return column_list

    def _ensure_params_defaults(self, index_item: dict):
        itype = index_item.get('index_type')
        if itype:
            itype_up = str(itype).upper()
            index_item['index_type'] = itype_up
        else:
            LOG.error(f'[OceanBaseStore] Cannot find `index_type` in index_kwargs: {index_item}')
            raise ValueError(f'Cannot find `index_type` in `index_kwargs` of `{index_item}`')

        defaults = OCEANBASE_INDEX_TYPE_DEFAULTS.get(index_item['index_type'], None)
        if defaults is None:
            LOG.error(f'[OceanBaseStore] Unsupported index type: {index_item["index_type"]}')
            raise ValueError(f'[OceanBase Store] Unsupported index type: {index_item["index_type"]}')

        if 'metric_type' not in index_item and 'metric_type' in defaults:
            index_item['metric_type'] = defaults['metric_type']
            LOG.info(f'[OceanBaseStore] Using default metric_type: {defaults["metric_type"]}')

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
        return [sqlalchemy.text(combined_filter)]

    def _construct_filter_expr(self, filters: Dict[str, Union[List, set]]) -> str:  # noqa: C901
        if not filters:
            return ''

        filter_parts = []
        for key, value in filters.items():
            try:
                if key not in self._global_metadata_desc.keys():
                    LOG.debug(f'[OceanBaseStore - _construct_filter_expr] Skipping unknown key: {key}')
                    continue

                field_name = self._gen_global_meta_key(key)

                if isinstance(value, (list, set)):
                    value_list = list(value)
                    if not value_list:
                        continue

                    if isinstance(value_list[0], str):
                        escaped_values = [v.replace('"', '\\"') for v in value_list]
                        values_str = ', '.join(f'"{v}"' for v in escaped_values)
                    else:
                        values_str = ', '.join(str(v) for v in value_list)

                    filter_parts.append(f'{field_name} in ({values_str})')

                elif isinstance(value, str):
                    escaped_value = value.replace('"', '\\"')
                    filter_parts.append(f'{field_name} = "{escaped_value}"')

                elif isinstance(value, (int, float)):
                    filter_parts.append(f'{field_name} = {value}')

                elif isinstance(value, bool):
                    filter_parts.append(f'{field_name} = {1 if value else 0}')

                else:
                    continue

            except Exception as e:
                LOG.warning(f'[OceanBaseStore - _construct_filter_expr] Error processing filter {key}={value}: {e}')
                continue

        result = ' and '.join(filter_parts)
        if result:
            LOG.debug(f'[OceanBaseStore - _construct_filter_expr] Filter expression: {result}')
        return result

    def _get_ids_where_clause(self, criteria: dict):
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

        return ids, where_clause
