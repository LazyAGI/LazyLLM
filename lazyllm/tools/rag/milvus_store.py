import copy
from typing import Dict, List, Optional, Union, Callable, Set
import pymilvus
from pymilvus import MilvusClient, CollectionSchema, FieldSchema
from .doc_node import DocNode
from .map_store import MapStore
from .utils import parallel_do_embedding
from .index_base import IndexBase
from .store_base import StoreBase
from .doc_field_desc import DocFieldDesc
from .doc_builtin_field import DocBuiltinField
from lazyllm.common import override
import pickle
import base64

class MilvusStore(StoreBase):
    _primary_key = 'uid'

    _embedding_key_prefix = 'embedding_'
    _field_key_prefix = 'field_'

    _builtin_keys = {
        _primary_key: {
            'dtype': pymilvus.DataType.VARCHAR,
            'max_length': 256,
            'is_primary': True,
        },
        'parent': {
            'dtype': pymilvus.DataType.VARCHAR,
            'max_length': 256,
        },
        'text': {
            'dtype': pymilvus.DataType.VARCHAR,
            'max_length': 65535,
        },
        'metadata': {
            'dtype': pymilvus.DataType.VARCHAR,
            'max_length': 65535,
        },
    }

    _builtin_fields_desc = {
        DocBuiltinField.DOC_PATH: DocFieldDesc(data_type=DocFieldDesc.DTYPE_VARCHAR,
                                               default_value=' ', max_size=65535),
        DocBuiltinField.KB: DocFieldDesc(data_type=DocFieldDesc.DTYPE_ARRAY,
                                         element_type=DocFieldDesc.DTYPE_INT32,
                                         default_value=[2 ** 32 - 1], max_size=4096),
    }

    _type2milvus = [
        pymilvus.DataType.VARCHAR,
        pymilvus.DataType.ARRAY,
        pymilvus.DataType.INT32,
    ]

    def __init__(self, group_embed_keys: Dict[str, Set[str]], embed: Dict[str, Callable], # noqa C901
                 embed_dims: Dict[str, int], fields_desc: Dict[str, DocFieldDesc],
                 uri: str, embedding_index_type: Optional[str] = None,
                 embedding_metric_type: Optional[str] = None, **kwargs):
        self._group_embed_keys = group_embed_keys
        self._embed = embed
        self._client = MilvusClient(uri=uri)

        # XXX milvus 2.4.x doesn't support `default_value`
        # https://milvus.io/docs/product_faq.md#Does-Milvus-support-specifying-default-values-for-scalar-or-vector-fields
        if fields_desc:
            self._fields_desc = fields_desc | self._builtin_fields_desc
        else:
            self._fields_desc = self._builtin_fields_desc

        if not embedding_index_type:
            embedding_index_type = 'HNSW'

        if not embedding_metric_type:
            embedding_metric_type = 'COSINE'

        for group, embed_keys in group_embed_keys.items():
            field_list = []
            index_params = self._client.prepare_index_params()

            for key, info in self._builtin_keys.items():
                field_list.append(FieldSchema(name=key, **info))

            for key in embed_keys:
                dim = embed_dims.get(key)
                if not dim:
                    raise ValueError(f'cannot find embedding dim of embed [{key}] in [{embed_dims}]')

                field_name = self._gen_embedding_key(key)
                field_list.append(FieldSchema(name=field_name, dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dim))
                index_params.add_index(field_name=field_name, index_type=embedding_index_type,
                                       metric_type=embedding_metric_type)

            if self._fields_desc:
                for key, desc in self._fields_desc.items():
                    if desc.data_type == DocFieldDesc.DTYPE_ARRAY:
                        if not desc.element_type:
                            raise ValueError(f'Milvus field [{key}]: `element_type` is required when '
                                             '`data_type` is DTYPE_ARRAY.')
                        field_args = {
                            'element_type': self._type2milvus[desc.element_type],
                            'max_capacity': desc.max_size,
                        }
                    elif desc.data_type == DocFieldDesc.DTYPE_VARCHAR:
                        field_args = {
                            'max_length': desc.max_size,
                        }
                    else:
                        field_args = {}
                    field_list.append(FieldSchema(name=self._gen_field_key(key),
                                                  dtype=self._type2milvus[desc.data_type],
                                                  default_value=desc.default_value,
                                                  **field_args))

            schema = CollectionSchema(fields=field_list, auto_id=False, enable_dynamic_fields=False)
            self._client.create_collection(collection_name=group, schema=schema,
                                           index_params=index_params)

        self._map_store = MapStore(node_groups=list(group_embed_keys.keys()), embed=embed)
        self._load_all_nodes_to(self._map_store)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            embed_keys = self._group_embed_keys.get(node.group)
            if embed_keys:
                parallel_do_embedding(self._embed, embed_keys, [node])
            data = self._serialize_node_partial(node)
            self._client.upsert(collection_name=node.group, data=[data])

        self._map_store.update_nodes(nodes)

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
              similarity: Optional[str] = None,
              similarity_cut_off: Optional[Union[float, Dict[str, float]]] = None,
              topk: int = 10,
              embed_keys: Optional[List[str]] = None,
              filters: Optional[Dict[str, Union[List, set]]] = None,
              **kwargs) -> List[DocNode]:
        if similarity is not None:
            raise ValueError('`similarity` MUST be None when Milvus backend is used.')

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
                uidset.update(result['id'])

        return self._map_store.get_nodes(group_name, list(uidset))

    # ----- internal helper functions ----- #

    @classmethod
    def _gen_embedding_key(cls, k: str) -> str:
        return cls._embedding_key_prefix + k

    @classmethod
    def _gen_field_key(cls, k: str) -> str:
        return cls._field_key_prefix + k

    def _load_all_nodes_to(self, store: StoreBase):
        for group_name in self._client.list_collections():
            results = self._client.query(collection_name=group_name,
                                         filter=f'{self._primary_key} != ""')
            for result in results:
                doc = self._deserialize_node_partial(result)
                doc.group = group_name
                store.update_nodes([doc], group_name)

        # construct DocNode::parent and DocNode::children
        for group in self.all_groups():
            for node in self.get_nodes(group):
                if node.parent:
                    parent_uid = node.parent
                    parent_node = self._map_store.find_node_by_uid(parent_uid)
                    node.parent = parent_node
                    parent_node.children[node.group].append(node)

    def _construct_filter_expr(self, filters: Dict[str, Union[List, set]]) -> str:
        ret_str = ""
        for name, candidates in filters.items():
            desc = self._fields_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')

            key = self._gen_field_key(name)
            if not isinstance(candidates, List):
                candidates = list(candidates)
            if desc.data_type == DocFieldDesc.DTYPE_ARRAY:
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
            'uid': node.uid,
            'text': node.text,
            'parent': node.parent.uid if node.parent else '',
            'metadata': base64.b64encode(pickle.dumps(node._metadata)).decode('utf-8'),
        }

        for k, v in node.embedding.items():
            res[self._gen_embedding_key(k)] = v

        for name, desc in self._fields_desc.items():
            val = node.fields.get(name, desc.default_value)
            if val:
                res[self._gen_field_key(name)] = val

        return res

    def _deserialize_node_partial(self, result: Dict) -> DocNode:
        record = copy.copy(result)

        doc = DocNode(
            uid=record.pop('uid'),
            text=record.pop('text'),
            parent=record.pop('parent'),  # this is the parent's uid
            metadata=pickle.loads(base64.b64decode(record.pop('metadata').encode('utf-8'))),
        )

        for k, v in record.items():
            if k.startswith(self._embedding_key_prefix):
                doc.embedding[k[len(self._embedding_key_prefix):]] = v
            elif k.startswith(self._field_key_prefix):
                if doc.parent:
                    doc._fields[k[len(self._field_key_prefix):]] = v

        return doc
