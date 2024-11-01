import copy
from typing import Dict, List, Optional, Union, Callable
import pymilvus
from pymilvus import MilvusClient, FieldSchema, CollectionSchema
from .doc_node import DocNode
from .map_backend import MapBackend
from .embed_utils import parallel_do_embedding
from .index_base import IndexBase
from .store_base import StoreBase
from lazyllm.common import override

class MilvusField:
    DTYPE_VARCHAR = 0
    DTYPE_FLOAT_VECTOR = 1
    DTYPE_SPARSE_FLOAT_VECTOR = 2

    def __init__(self, name: str, data_type: int, index_type: Optional[str] = None,
                 metric_type: Optional[str] = "", index_params: Dict = {},
                 max_length: Optional[int] = None):
        self.name = name
        self.data_type = data_type
        self.index_type = index_type
        self.metric_type = metric_type
        self.index_params = index_params
        self.max_length = max_length


class MilvusStore(StoreBase):
    _type2milvus = [
        pymilvus.DataType.VARCHAR,  # DTYPE_VARCHAR
        pymilvus.DataType.FLOAT_VECTOR,  # DTYPE_FLOAT_VECTOR
        pymilvus.DataType.SPARSE_FLOAT_VECTOR,  # DTYPE_SPARSE_FLOAT_VECTOR
    ]

    def __init__(self, uri: str, embed: Dict[str, Callable],
                 # a field is either an embedding key or a metadata key
                 group_fields: Dict[str, List[MilvusField]]):
        self._primary_key = 'uid'
        self._embedding_keys = embed.keys()
        self._embed = embed
        self._client = MilvusClient(uri=uri)

        embed_dim = {k: len(e('a')) for k, e in embed.items()}
        builtin_fields = [
            FieldSchema(name=self._primary_key, dtype=pymilvus.DataType.VARCHAR,
                        max_length=128, is_primary=True),
            FieldSchema(name='text', dtype=pymilvus.DataType.VARCHAR,
                        max_length=65535),
            FieldSchema(name='parent', dtype=pymilvus.DataType.VARCHAR,
                        max_length=256),
        ]

        for group_name, field_list in group_fields.items():
            if group_name in self._client.list_collections():
                continue

            index_params = self._client.prepare_index_params()
            field_schema_list = copy.copy(builtin_fields)

            for field in field_list:
                field_schema = None
                if field.name in self._embedding_keys:
                    field_schema = FieldSchema(
                        name=self._gen_embedding_key(field.name),
                        dtype=self._type2milvus[field.data_type],
                        dim=embed_dim.get(field.name))
                else:
                    field_schema = FieldSchema(
                        name=self._gen_metadata_key(field.name),
                        dtype=self._type2milvus[field.data_type],
                        max_length=field.max_length)
                field_schema_list.append(field_schema)

                if field.index_type is not None:
                    index_params.add_index(field_name=field_schema.name,
                                           index_type=field.index_type,
                                           metric_type=field.metric_type,
                                           params=field.index_params)

            schema = CollectionSchema(fields=field_schema_list)
            self._client.create_collection(collection_name=group_name, schema=schema,
                                           index_params=index_params)

        self._map_backend = MapBackend(list(group_fields.keys()))
        self._load_all_nodes_to(self._map_backend)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        parallel_do_embedding(self._embed, nodes)
        for node in nodes:
            data = self._serialize_node_partial(node)
            self._client.upsert(collection_name=node.group, data=[data])

        self._map_backend.update_nodes(nodes)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            self._client.delete(collection_name=group_name,
                                filter=f'{self._primary_key} in {uids}')
        else:
            self._client.drop_collection(collection_name=group_name)

        self._map_backend.remove_nodes(group_name, uids)

    @override
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self._map_backend.get_nodes(group_name, uids)

    @override
    def is_group_active(self, name: str) -> bool:
        return self._map_backend.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        return self._map_backend.all_groups()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._map_backend.register_index(type, index)

    @override
    def get_index(self, type: str) -> Optional[IndexBase]:
        return self._map_backend.get_index(type)

    @override
    def query(self,
              query: str,
              group_name: str,
              similarity: Optional[str] = None,
              similarity_cut_off: Optional[Union[float, Dict[str, float]]] = None,
              topk: int = 10,
              embed_keys: Optional[List[str]] = None,
              **kwargs) -> List[DocNode]:
        uidset = set()
        for key in embed_keys:
            embed_func = self._embed.get(key)
            query_embedding = embed_func(query)
            results = self._client.search(collection_name=group_name, data=[query_embedding],
                                          limit=topk, anns_field=self._gen_embedding_key(key))
            # we have only one `data` for search() so there is only one result in `results`
            if len(results) != 1:
                raise ValueError(f'number of results [{len(results)}] != expected [1]')

            for result in results[0]:
                uidset.update(result['id'])

        return self._map_backend.get_nodes(group_name, list(uidset))

    # ----- internal helper functions ----- #

    @staticmethod
    def _gen_embedding_key(k: str) -> str:
        return 'embedding_' + k

    @staticmethod
    def _gen_metadata_key(k: str) -> str:
        return 'metadata_' + k

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
                    parent_node = self._map_backend.find_node_by_uid(parent_uid)
                    node.parent = parent_node
                    parent_node.children[node.group].append(node)

    def _serialize_node_partial(self, node: DocNode) -> Dict:
        res = {
            'uid': node.uid,
            'text': node.text,
        }

        if node.parent:
            res['parent'] = node.parent.uid
        else:
            res['parent'] = ''

        for k, v in node.embedding.items():
            res[self._gen_embedding_key(k)] = v
        for k, v in node.metadata.items():
            res[self._gen_metadata_key(k)] = v

        return res

    def _deserialize_node_partial(self, result: Dict) -> DocNode:
        '''
        without parent and children
        '''
        doc = DocNode(
            uid=result.get('uid'),
            text=result.get('text'),
            parent=result.get('parent'),  # this is the parent's uid
        )

        for k in self._embedding_keys:
            val = result.get(self._gen_embedding_key(k))
            if val:
                doc.embedding[k] = val
        for k in self._metadata_keys:
            val = result.get(self._gen_metadata_key(k))
            if val:
                doc._metadata[k] = val

        return doc
