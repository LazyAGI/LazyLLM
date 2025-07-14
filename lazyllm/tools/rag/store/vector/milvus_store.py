""" Milvus Vector Store (For Vector Store Only)"""
import copy

from urllib import parse
from typing import Dict, List, Union, Optional, Set

from lazyllm import LOG
from lazyllm.thirdparty import pymilvus
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, BUILDIN_GLOBAL_META_DESC, StoreCapability, EMBED_PREFIX

from ..utils import parallel_do_embedding
from ..data_type import DataType
from ..doc_node import DocNode
from ..index_base import IndexBase
from ..global_metadata import GlobalMetadataDesc, RAG_DOC_ID

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
        LOG.info("[Milvus Vector Store] init success!")

    def _check_connection(self):
        if not pymilvus.connections.has_connection(alias=self._client._using):
            LOG.info("[Milvus Vector Store] try to reconnect...")
            if self._type == 'local':
                pymilvus.connections.connect(alias=self._client._using, uri=self._uri)
            else:
                pymilvus.connections.connect(alias=self._client._using, db_name=self._db_name, uri=self._uri)

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        """ upsert data to the store """
        if not data: return
        data_embeddings = data[0].get("embedding", {})
        if not data_embeddings: return
        if not self._client.has_collection(collection_name):
            embed_kwargs = {}
            for embed_key, embedding in data_embeddings.items():
                k = embed_key[len(EMBED_PREFIX):]
                assert self._embed_datatypes.get(k), \
                    f'cannot find embedding params for embed [{k}]'
                if k not in embed_kwargs: embed_kwargs[k] = {"dtype": TYPE2MILVUS[self._embed_datatypes[k]]}
                if self._embed_dims.get(k): embed_kwargs[k]["dim"] = self._embed_dims[k]
            self._create_collection(collection_name, embed_kwargs)

        self._check_connection()

        self._client.upsert(collection_name=collection_name, data=data)
        return True

    def _get_constant_field_schema(self) -> List[pymilvus.FieldSchema]:
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
        field_list = self._get_constant_field_schema()
        index_params = self._client.prepare_index_params()
        for k, kws in embed_kwargs.items():
            field_list.append(pymilvus.FieldSchema(name=k, **kws))
            index_params.add_index(field_name=k, **kws)
            if isinstance(self._index_kwargs, dict):
                index_params.add_index(field_name=k, **self._index_kwargs)
        schema = pymilvus.CollectionSchema(fields=field_list, auto_id=False, enable_dynamic_field=False)
        self._client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    def _gen_field_key(self, k: str) -> str:
        return self._global_metadata_key_prefix + k

    def _gen_col_name(self, db_name, group_name, embed_key):
        return self._col_name_format.format(db=db_name, group=group_name, embedding=embed_key)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """ update nodes to the store """
        # NOTE: 与之前不同的是，这里持久化存储的node，每次入库都要embedding
        # 因为从存储中拿出node后再入库的理由只可能是：1. meta更新 2. 切片重解析 这些都会改变node的内容
        # 导致原先embedding失效
        self._check_connection()
        parallel_do_embedding(embed=self._embed, embed_keys=[], nodes=nodes, group_embed_keys=self._group_embed_keys)
        group_embed_data_dict = {}
        group_cnt = {}
        for node in nodes:
            if not node.embedding:
                continue
            if node._group not in group_cnt:
                group_cnt[node._group] = 1
            node._metadata["number"] = group_cnt[node._group]
            group_cnt[node._group] += 1
            data = self._serialize_node(node)
            for embed_key in data['vectors'].keys():
                if node._group not in group_embed_data_dict:
                    group_embed_data_dict[node._group] = {}
                if embed_key not in group_embed_data_dict[node._group]:
                    group_embed_data_dict[node._group][embed_key] = []
                d = copy.deepcopy(data)
                d.pop('vectors', None)
                d['vector'] = data['vectors'][embed_key]
                group_embed_data_dict[node._group][embed_key].append(d)

        if not group_embed_data_dict:
            LOG.warning("[Milvus Vector Store] No nodes need to update.")

        for group_name, embed_data in group_embed_data_dict.items():
            for embed_key, data in embed_data.keys():
                col_name = self._gen_col_name(self._db_name, group_name, embed_key)
                if not self._client.has_collection(col_name):
                    self._create_col(col_name=col_name, embed_key=embed_key)
                for i in range(0, len(data), MILVUS_UPSERT_BATCH_SIZE):
                    self._client.upsert(collection_name=col_name, data=data[i:i + MILVUS_UPSERT_BATCH_SIZE])

    @override
    def remove_nodes(self, group_name: str, doc_ids: Optional[List[str]] = None,
                     uids: Optional[List[str]] = None) -> None:
        """ remove nodes from the store by doc_ids or uids """
        self._check_connection()
        for embed_key in self._group_embed_keys[group_name]:
            col_name = self._gen_col_name(self._db_name, group_name, embed_key)
            if self._client.has_collection(col_name):
                if uids:
                    self._client.delete(collection_name=col_name, ids=uids)
                elif doc_ids:
                    self._client.delete(collection_name=col_name,
                                        filter=f'{self._gen_field_key(RAG_DOC_ID)} in {doc_ids}')
                else:
                    self._client.drop_collection(collection_name=col_name)
        return

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        raise NotImplementedError("register_index is not supported for MilvusVecStore."
                                  "Please use register_index for store that support hook")

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        raise NotImplementedError('get_index is not supported for MilvusVecStore.')

    @override
    def clear_cache(self, group_names: Optional[List[str]]) -> bool:
        self._check_connection()
        if not group_names:
            group_names = [g_n for g_n, eks in self._group_embed_keys.items() if len(eks)]
        for g_n in group_names:
            for embed_key in self._group_embed_keys[g_n]:
                col_name = self._gen_col_name(self._db_name, g_n, embed_key)
                self._client.drop_collection(collection_name=col_name)

    @override
    def _serialize_node(self, node: DocNode):
        """ serialize node to a dict that can be stored in vector store """
        res = {'uid': node._uid, 'vectors': {k: v for k, v in node.embedding.items()}}
        for name, desc in self._global_metadata_desc.items():
            val = node.global_metadata.get(name, desc.default_value)
            if val is not None:
                res[self._gen_field_key(name)] = val
        return res

    def _construct_filter_expr(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret_str = ""
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')

            key = self._gen_field_key(name)
            if (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                candidates = list(candidates)
            if desc.data_type == DataType.ARRAY:
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '

        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '

        return ret_str

    @override
    def query(self, query: str, group_name: str, similarity_name: Optional[str] = None,
              similarity_cut_off: Optional[Union[float, Dict[str, float]]] = float('-inf'),
              topk: int = 10, embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[List, set]]] = None, **kwargs) -> List[DocNode]:
        if similarity_name is not None:
            raise ValueError('`similarity` MUST be None when Milvus backend is used.')

        if not embed_keys:
            raise ValueError('empty or None `embed_keys` is not supported.')

        filter_str = self._construct_filter_expr(filters) if filters else ""

        uid_score = {}
        self._check_connection()
        for key in embed_keys:
            col_name = self._gen_col_name(self._db_name, group_name, key)
            embed_func = self._embed.get(key)
            query_embedding = embed_func(query)
            results = self._client.search(collection_name=col_name, data=[query_embedding], limit=topk,
                                          anns_field=key, filter=filter_str)
            # we have only one `data` for search() so there is only one result in `results`
            if len(results) != 1:
                raise ValueError(f'number of results [{len(results)}] != expected [1]')
            sim_cut_off = similarity_cut_off if isinstance(similarity_cut_off, float) else similarity_cut_off[key]

            for result in results[0]:
                if result['distance'] < sim_cut_off:
                    continue
                uid_score[result['id']] = result['distance'] if result['id'] not in uid_score \
                    else max(uid_score[result['id']], result['distance'])
        return uid_score
