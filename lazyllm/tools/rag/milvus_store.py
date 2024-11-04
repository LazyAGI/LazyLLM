import copy
from typing import Dict, List, Optional, Union, Callable
import pymilvus
from pymilvus import MilvusClient
from .doc_node import DocNode
from .map_store import MapStore
from .utils import parallel_do_embedding
from .index_base import IndexBase
from .store_base import StoreBase
from lazyllm.common import override

class MilvusField:
    DTYPE_VARCHAR = 0
    DTYPE_FLOAT_VECTOR = 1
    DTYPE_SPARSE_FLOAT_VECTOR = 2

    def __init__(self, data_type: int, index_type: Optional[str] = None,
                 metric_type: Optional[str] = "", index_params: Dict = {},
                 max_length: Optional[int] = None):
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

    def __init__(self, uri: str, group_fields: Dict[str, Dict[str, MilvusField]],
                 node_groups: List[str], embed: Dict[str, Callable], **kwargs):
        new_copy = copy.copy(group_fields)
        for g in node_groups:
            if g not in new_copy:
                new_copy[g] = {}
        group_fields = new_copy

        self._primary_key = 'uid'
        self._embedding_keys = embed.keys()
        self._embed = embed
        self._client = MilvusClient(uri=uri)

        embed_dim = {k: len(e('a')) for k, e in embed.items()}
        builtin_fields = {
            self._primary_key: {
                'datatype': pymilvus.DataType.VARCHAR,
                'max_length': 128,
                'is_primary': True,
            },
            'text': {
                'datatype': pymilvus.DataType.VARCHAR,
                'max_length': True,
            },
            'parent': {
                'datatype': pymilvus.DataType.VARCHAR,
                'max_length': 256,
            },
        }

        for group_name, fields in group_fields.items():
            if group_name in self._client.list_collections():
                continue

            index_params = self._client.prepare_index_params()
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)

            for name, field in builtin_fields.items():
                schema.add_field(field_name=name, **field)

            for name, field in fields.items():
                field_name = None
                if name in self._embedding_keys:
                    field_name = self._gen_embedding_key(name)
                    schema.add_field(
                        field_name=field_name,
                        datatype=self._type2milvus[field.data_type],
                        dim=embed_dim.get(name))
                else:
                    field_name = self._gen_metadata_key(name)
                    schema.add_field(
                        field_name=field_name,
                        datatype=self._type2milvus[field.data_type],
                        max_length=field.max_length)

                if field.index_type is not None:
                    index_params.add_index(field_name=field_name,
                                           index_type=field.index_type,
                                           metric_type=field.metric_type,
                                           params=field.index_params)

            self._client.create_collection(collection_name=group_name, schema=schema,
                                           index_params=index_params)

        self._map_store = MapStore(node_groups=list(group_fields.keys()), embed=embed)
        self._load_all_nodes_to(self._map_store)

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        parallel_do_embedding(self._embed, nodes)
        for node in nodes:
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
              **kwargs) -> List[DocNode]:
        if similarity is not None:
            raise ValueError('`similarity` MUST be None when Milvus backend is used.')

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

        return self._map_store.get_nodes(group_name, list(uidset))

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
                    parent_node = self._map_store.find_node_by_uid(parent_uid)
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
