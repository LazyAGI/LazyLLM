import copy
from collections import defaultdict
from typing import Dict, List, Optional, Union, Callable, Set
from lazyllm.thirdparty import pymilvus
from .doc_node import DocNode
from .map_store import MapStore
from .utils import parallel_do_embedding
from .index_base import IndexBase
from .store_base import StoreBase
from .global_metadata import (GlobalMetadataDesc, RAG_DOC_ID, RAG_DOC_PATH, RAG_DOC_FILE_NAME,
                              RAG_DOC_FILE_TYPE, RAG_DOC_FILE_SIZE, RAG_DOC_CREATION_DATE,
                              RAG_DOC_LAST_MODIFIED_DATE, RAG_DOC_LAST_ACCESSED_DATE)
from .data_type import DataType
from lazyllm.common import override, obj2str, str2obj

MILVUS_UPSERT_BATCH_SIZE = 500
MILVUS_PAGINATION_OFFSET = 1000

class MilvusStore(StoreBase):
    # we define these variables as members so that pymilvus is not imported until MilvusStore is instantiated.
    def _def_constants(self) -> None:
        self._primary_key = 'uid'

        self._embedding_key_prefix = 'embedding_'
        self._global_metadata_key_prefix = 'global_metadata_'

        self._builtin_keys = {
            self._primary_key: {
                'dtype': pymilvus.DataType.VARCHAR,
                'max_length': 256,
                'is_primary': True,
            },
            'parent': {
                'dtype': pymilvus.DataType.VARCHAR,
                'max_length': 256,
            },
            'content': {
                'dtype': pymilvus.DataType.VARCHAR,
                'max_length': 65535,
            },
            'metadata': {
                'dtype': pymilvus.DataType.VARCHAR,
                'max_length': 65535,
            },
        }

        self._builtin_global_metadata_desc = {
            RAG_DOC_ID: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                           default_value=' ', max_size=512),
            RAG_DOC_PATH: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                             default_value=' ', max_size=65535),
            RAG_DOC_FILE_NAME: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                                  default_value=' ', max_size=128),
            RAG_DOC_FILE_TYPE: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                                  default_value=' ', max_size=64),
            RAG_DOC_FILE_SIZE: GlobalMetadataDesc(data_type=DataType.INT32,
                                                  default_value=0),
            RAG_DOC_CREATION_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                                      default_value=' ', max_size=10),
            RAG_DOC_LAST_MODIFIED_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                                           default_value=' ', max_size=10),
            RAG_DOC_LAST_ACCESSED_DATE: GlobalMetadataDesc(data_type=DataType.VARCHAR,
                                                           default_value=' ', max_size=10)
        }

        self._type2milvus = [
            pymilvus.DataType.VARCHAR,
            pymilvus.DataType.ARRAY,
            pymilvus.DataType.INT32,
            pymilvus.DataType.FLOAT_VECTOR,
            pymilvus.DataType.SPARSE_FLOAT_VECTOR,
        ]

    def __init__(self, group_embed_keys: Dict[str, Set[str]], embed: Dict[str, Callable], # noqa C901
                 embed_dims: Dict[str, int], embed_datatypes: Dict[str, DataType],
                 global_metadata_desc: Dict[str, GlobalMetadataDesc],
                 uri: str, index_kwargs: Optional[Union[Dict, List]] = None):
        self._def_constants()

        self._group_embed_keys = group_embed_keys
        self._embed = embed
        self._client = pymilvus.MilvusClient(uri=uri)

        if embed_dims is None:
            embed_dims = {}
        if embed_datatypes is None:
            embed_datatypes = {}

        # XXX milvus 2.4.x doesn't support `default_value`
        # https://milvus.io/docs/product_faq.md#Does-Milvus-support-specifying-default-values-for-scalar-or-vector-fields
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | self._builtin_global_metadata_desc
        else:
            self._global_metadata_desc = self._builtin_global_metadata_desc

        collections = self._client.list_collections()
        for group, embed_keys in group_embed_keys.items():
            if group in collections:
                continue

            field_list = []
            index_params = self._client.prepare_index_params()

            for key, info in self._builtin_keys.items():
                field_list.append(pymilvus.FieldSchema(name=key, **info))

            for key in embed_keys:
                datatype = embed_datatypes.get(key)
                if not datatype:
                    raise ValueError(f'cannot find embedding datatype if embed [{key}] in [{embed_datatypes}]')

                field_kwargs = {}
                dim = embed_dims.get(key)  # can be empty if embedding is sparse
                if dim:
                    field_kwargs['dim'] = dim

                field_name = self._gen_embedding_key(key)
                field_list.append(pymilvus.FieldSchema(name=field_name, dtype=self._type2milvus[datatype],
                                                       **field_kwargs))
                if isinstance(index_kwargs, list):
                    embed_key_field_name = "embed_key"
                    for item in index_kwargs:
                        item_key = item.get(embed_key_field_name, None)
                        if not item_key:
                            raise ValueError(f'cannot find `{embed_key_field_name}` in `index_kwargs` of `{field_name}`')
                        if item_key == key:
                            index_kwarg = item.copy()
                            index_kwarg.pop(embed_key_field_name, None)
                            index_params.add_index(field_name=field_name, **index_kwarg)
                            break
                elif isinstance(index_kwargs, dict):
                    index_params.add_index(field_name=field_name, **index_kwargs)

            if self._global_metadata_desc:
                for key, desc in self._global_metadata_desc.items():
                    if desc.data_type == DataType.ARRAY:
                        if not desc.element_type:
                            raise ValueError(f'Milvus field [{key}]: `element_type` is required when '
                                             '`data_type` is ARRAY.')
                        field_args = {
                            'element_type': self._type2milvus[desc.element_type],
                            'max_capacity': desc.max_size,
                        }
                    elif desc.data_type == DataType.VARCHAR:
                        field_args = {
                            'max_length': desc.max_size,
                        }
                    else:
                        field_args = {}
                    field_list.append(pymilvus.FieldSchema(name=self._gen_field_key(key),
                                                           dtype=self._type2milvus[desc.data_type],
                                                           default_value=desc.default_value,
                                                           **field_args))

            schema = pymilvus.CollectionSchema(fields=field_list, auto_id=False, enable_dynamic_field=False)
            self._client.create_collection(collection_name=group, schema=schema, index_params=index_params)
        valid_group_names = set(self._client.list_collections()) | set(group_embed_keys.keys())
        self._map_store = MapStore(list(valid_group_names), embed=embed)
        self._load_all_nodes_to(self._map_store)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        parallel_do_embedding(self._embed, [], nodes, self._group_embed_keys)
        group_embed_dict = defaultdict(list)
        for node in nodes:
            data = self._serialize_node_partial(node)
            group_embed_dict[node._group].append(data)
        for group_name, data in group_embed_dict.items():
            for i in range(0, len(data), MILVUS_UPSERT_BATCH_SIZE):
                self._client.upsert(collection_name=group_name, data=data[i:i + MILVUS_UPSERT_BATCH_SIZE])
        self._map_store.update_nodes(nodes)

    @override
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
        self._map_store.update_doc_meta(filepath, metadata)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            self._client.delete(collection_name=group_name,
                                filter=f'{self._primary_key} in {uids}')
        else:
            self._client.drop_collection(collection_name=group_name)

        self._map_store.remove_nodes(group_name, uids)

    @override
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self._map_store.get_nodes(group_name, uids)

    @override
    def is_group_active(self, name: str) -> bool:
        return self._map_store.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        return self._map_store.all_groups()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._map_store.register_index(type, index)

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        if type is None:
            type = 'default'
        return self._map_store.get_index(type)

    @override
    def query(self,
              query: str,
              group_name: str,
              similarity_name: Optional[str] = None,
              similarity_cut_off: Optional[Union[float, Dict[str, float]]] = None,
              topk: int = 10,
              embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[List, set]]] = None,
              **kwargs) -> List[DocNode]:
        if similarity_name is not None:
            raise ValueError('`similarity` MUST be None when Milvus backend is used.')

        if not embed_keys:
            raise ValueError('empty or None `embed_keys` is not supported.')

        filter_str = self._construct_filter_expr(filters) if filters else ""

        uidset = set()
        for key in embed_keys:
            embed_func = self._embed.get(key)
            query_embedding = embed_func(query)
            results = self._client.search(collection_name=group_name, data=[query_embedding],
                                          limit=topk, anns_field=self._gen_embedding_key(key),
                                          filter=filter_str)
            # we have only one `data` for search() so there is only one result in `results`
            if len(results) != 1:
                raise ValueError(f'number of results [{len(results)}] != expected [1]')

            for result in results[0]:
                uidset.add(result['id'])

        return self._map_store.get_nodes(group_name, list(uidset))

    # ----- internal helper functions ----- #

    def _gen_embedding_key(self, k: str) -> str:
        return self._embedding_key_prefix + k

    def _gen_field_key(self, k: str) -> str:
        return self._global_metadata_key_prefix + k

    def _load_all_nodes_to(self, store: StoreBase) -> None:
        uid2node = {}
        for group_name in self._client.list_collections():
            collection_desc = self._client.describe_collection(collection_name=group_name)
            field_names = [field.get("name") for field in collection_desc.get('fields', [])]

            iterator = self._client.query_iterator(
                collection_name=group_name,
                batch_size=MILVUS_PAGINATION_OFFSET,
                filter=f'{self._primary_key} != ""',
                output_fields=field_names
            )

            results = []
            while True:
                result = iterator.next()

                if not result:
                    iterator.close()
                    break
                results += result

            for result in results:
                node = self._deserialize_node_partial(result)
                node._group = group_name
                uid2node.setdefault(node._uid, node)

        # construct DocNode::parent and DocNode::children
        for node in uid2node.values():
            if node.parent:
                parent_uid = node.parent
                parent_node = uid2node.get(parent_uid)
                node.parent = parent_node
                parent_node.children[node._group].append(node)

        store.update_nodes(list(uid2node.values()))

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
                # https://github.com/milvus-io/milvus/discussions/35279
                # `array_contains_any` requires milvus >= 2.4.3 and is not supported in local(aka lite) mode.
                ret_str += f'array_contains_any({key}, {candidates}) and '
            else:
                ret_str += f'{key} in {candidates} and '

        if len(ret_str) > 0:
            return ret_str[:-5]  # truncate the last ' and '

        return ret_str

    def _serialize_node_partial(self, node: DocNode) -> Dict:
        res = {
            'uid': node._uid,
            'content': obj2str(node._content),
            'parent': node.parent._uid if node.parent else '',
            'metadata': obj2str(node._metadata),
        }

        for k, v in node.embedding.items():
            res[self._gen_embedding_key(k)] = v

        for name, desc in self._global_metadata_desc.items():
            val = node.global_metadata.get(name, desc.default_value)
            if val is not None:
                res[self._gen_field_key(name)] = val

        return res

    def _deserialize_node_partial(self, result: Dict) -> DocNode:
        record = copy.copy(result)

        doc = DocNode(
            uid=record.pop('uid'),
            content=str2obj(record.pop('content')),
            parent=record.pop('parent'),  # this is the parent's uid
            metadata=str2obj(record.pop('metadata')),
        )

        for k, v in record.items():
            if k.startswith(self._embedding_key_prefix):
                doc.embedding[k[len(self._embedding_key_prefix):]] = v
            elif k.startswith(self._global_metadata_key_prefix):
                if doc.is_root_node:
                    doc._global_metadata[k[len(self._global_metadata_key_prefix):]] = v

        return doc
