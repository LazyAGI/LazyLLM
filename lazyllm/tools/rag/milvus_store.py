import copy
from typing import Dict, List, Optional, Union, Callable
import pymilvus
from pymilvus import MilvusClient
from .doc_node import DocNode
from .map_store import MapStore
from .utils import parallel_do_embedding
from .index_base import IndexBase
from .store_base import StoreBase
from .doc_field_info import DocFieldInfo
from lazyllm.common import override

class MilvusStore(StoreBase):
    _primary_key = 'uid'

    _embedding_key_prefix = 'embedding_'
    _field_key_prefix = 'field_'

    _builtin_fields = {
        _primary_key: {
            'datatype': pymilvus.DataType.VARCHAR,
            'max_length': 256,
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

    _type2milvus = [
        0,
        pymilvus.DataType.VARCHAR,
    ]

    def __init__(self, embed: Dict[str, Callable], fields_info: Dict[str, DocFieldInfo], uri: str,
                 embedding_index_type: Optional[str] = None, embedding_metric_type: Optional[str] = None,
                 **kwargs):
        self._embed = embed
        self._fields_info = fields_info
        self._embedding_index_type = embedding_index_type if embedding_index_type else 'HNSW'
        self._embedding_metric_type = embedding_metric_type if embedding_metric_type else 'COSINE'

        self._embedding_keys = embed.keys()
        self._embed_dim = {k: len(e('a')) for k, e in embed.items()}
        self._client = MilvusClient(uri=uri)

        self._map_store = MapStore(embed=embed)
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
    def add_group(self, name: str, embed_keys: Optional[List[str]] = None) -> None:
        if name in self._client.list_collections():
            return

        index_params = self._client.prepare_index_params()
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)

        for key in embed_keys:
            field_name = self._gen_embedding_key(key)
            schema.add_field(field_name=field_name, datatype=pymilvus.DataType.FLOAT_VECTOR)
            index_params.add_index(field_name=field_name, index_type=self._embedding_index_type,
                                   metric_type=self._embedding_metric_type)

        if self._fields_info:
            for key, info in self._fields_info.items():
                schema.add_field(field_name=self._gen_field_key(key),
                                 datatype=self._type2milvus[info.data_type])

        self._client.create_collection(collection_name=name, schema=schema,
                                       index_params=index_params)

        self._map_store.add_group(name, embed_keys)

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

    @classmethod
    def _gen_embedding_key(cls, k: str) -> str:
        return cls._embedding_key_prefix + k

    @classmethod
    def _gen_field_key(cls, k: str) -> str:
        return cls._field_key_prefix + k

    def _load_all_nodes_to(self, store: StoreBase):
        for group_name in self._client.list_collections():
            store.add_group(name=group_name, embed=self._embed)

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
            'parent': node.parent.uid if node.parent else '',
            'metadata': node._metadata,
        }

        for k, v in node.embedding.items():
            res[self._gen_embedding_key(k)] = v
        for k, v in node.fields.items():
            res[self._gen_field_key(k)] = v

        return res

    def _deserialize_node_partial(self, result: Dict) -> DocNode:
        record = copy.copy(result)

        doc = DocNode(
            uid=record.pop('uid'),
            text=record.pop('text'),
            parent=record.pop('parent'),  # this is the parent's uid
            metadata=record.pop('metadata'),
        )

        for k, v in record.items():
            if k.startswith(self._embedding_key_prefix):
                doc.embedding[k[len(self._embedding_key_prefix):]] = v
            elif k.startswith(self._field_key_prefix):
                if doc.parent:
                    doc._fields[k[len(self._field_key_prefix):]] = v

        return doc
