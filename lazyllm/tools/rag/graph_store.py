import os
import time
import json
import base64
import numpy as np
from typing import Dict, List, Optional, Callable
from lazyllm.thirdparty import nano_vectordb as nanodb
from lazyllm import LOG
from lazyllm.common import override
from .store_base import StoreBase, LAZY_GRAPH_NODE, LAZY_GRAPH_EDGE
from .doc_node import DocNode, GraphEntityNode, GraphRelationNode
from .index_base import IndexBase
from .utils import _FileNodeIndex
from .default_index import DefaultIndex
from .map_store import MapStore

# ---------------------------------------------------------------------------- #

def buffer_string_to_array(base64_str: str, dtype=float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    return data

class NanodbGraphStore(StoreBase):
    def __init__(self, group_embed_keys: Dict[str, str], embed: Dict[str, Callable],
                 embed_dims: Dict[str, int], dir: str, cosine_threshold: float = 0.3, **kwargs) -> None:
        self._entity_db = nanodb.NanoVectorDB(
            embedding_dim=embed_dims.get(group_embed_keys.get(LAZY_GRAPH_NODE)),
            storage_file=os.path.join(dir, 'vdb_entities.json'))
        self._relation_db = nanodb.NanoVectorDB(
            embedding_dim=embed_dims.get(group_embed_keys.get(LAZY_GRAPH_EDGE)),
            storage_file=os.path.join(dir, 'vdb_relationships.json'))
        LOG.success(f"Initialzed nanodb in path: {dir}")
        
        node_groups = list(group_embed_keys.keys())
        self._map_store = MapStore(node_groups=node_groups, embed=embed)
        self._load_store()
        self._embed = embed
        self._cosine_threshold = cosine_threshold
        
        self._name2index = {
            'default': DefaultIndex(embed, self._map_store),
            'file_node_map': _FileNodeIndex(),
        }

    @override
    def update_nodes(self, group_name: str, nodes: List[DocNode]) -> None:
        self._map_store.update_nodes(nodes)
        self._save_nodes(group_name, nodes)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            self._delete_group_nodes(group_name, uids)

        return self._map_store.remove_nodes(group_name, uids)

    @override
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
        self._map_store.update_doc_meta(filepath, metadata)

    @override
    def get_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        return self._map_store.get_nodes(group_name, uids)

    @override
    def is_group_active(self, name: str) -> bool:
        return self._map_store.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        return self._map_store.all_groups()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        if type is None:
            type = 'default'
        return self._name2index.get(type)
   
    @override
    def query(self, query: str, group_name: str, topk: int = 10,
              embed_keys: Optional[List[str]] = None) -> List[DocNode]:
        uidset = set()
        db_client = self._get_db_client(group_name)
        for key in embed_keys:
            embed_func = self._embed.get(key)
            query_embedding = json.loads(embed_func(query))
            results =  db_client.query(query = query_embedding,
                                        top_k = topk,
                                        better_than_threshold = self._cosine_threshold)

            results = [dp["__id__"] for dp in results]
            uidset.update(results)

        return self._map_store.get_nodes(group_name, list(uidset))

    def _load_store(self) -> None:
        # step 1 : load entity vdb
        self._load_storage(self._entity_db.storage_file, LAZY_GRAPH_NODE)
        
        # step 2 : load relationship vdb
        self._load_storage(self._relation_db.storage_file, LAZY_GRAPH_EDGE)

        LOG.success("Successfully Built nodes from nanodb.")

    def _save_nodes(self, group_name: str, nodes: List[DocNode]) -> None:
        if not nodes:
            return
        data, db_client = [], None
        current_time = time.time()
        if group_name == LAZY_GRAPH_NODE:
            db_client = self._entity_db
            for node in nodes:
                data.append({
                    "__id__": node._uid,
                    "__created_at__": current_time,
                    "entity_name": node.entity_name,
                    "__vector__": list(node._embedding.values())[0]})
        elif group_name == LAZY_GRAPH_EDGE:
            db_client = self._relation_db
            for node in nodes:
                data.append({
                    "__id__": node._uid,
                    "__created_at__": current_time,
                    "src_id": node.source,
                    "tgt_id": node.target,
                    "__vector__": list(node._embedding.values())[0]})
        else:
            raise NotImplementedError("Nanodb Graph Store only supports entity group and relation group")
        
        if data:
            results = db_client.upsert(data)
            db_client.save()
            LOG.info(f"Updated {len(results['update'])} node and inserted {len(results['insert'])} node into nano db ")

            
    def _delete_group_nodes(self, group_name: str, uids: List[str]) -> None:
        db_client = self._get_db_client(group_name)
        db_client.delete(uids)
        db_client.save()
        LOG.info(f"Deleted {uids} node from nano db.")

    def _load_storage(self, file_name: str, node_group: str) -> None:
        if not os.path.exists(file_name):
            LOG.info("No persistent data found, skip the entity loading phrase.")
            return

        with open(file_name, encoding="utf-8") as f:
            storage = json.load(f)

        matrix = np.frombuffer(base64.b64decode(storage["matrix"]), dtype=np.float32)
        storage["matrix"] = matrix.reshape(-1, storage["embedding_dim"])

        # Restore all nodes
        uid2node = {}
        nodes = self._build_nodes_from_json(storage, node_group)
        for node in nodes:
            uid2node[node._uid] = node

        self._map_store.update_nodes(list(uid2node.values()))

    def _build_nodes_from_json(self, results: Dict[str, List], node_group: str) -> List[DocNode]:
        nodes: List[DocNode] = []
        if node_group == LAZY_GRAPH_NODE:
            nodes = [GraphEntityNode(
                uid=data['__id__'],
                group=node_group,
                entity_name=data['entity_name'],
                embedding={node_group: results['matrix'][i]},
                metadata={'group': node_group}
            ) for i, data in enumerate(results['data'])]
        elif node_group == LAZY_GRAPH_EDGE:
            nodes = [GraphRelationNode(
                uid=data['__id__'],
                group=node_group,
                source=data['src_id'],
                target=data['tgt_id'],
                embedding={node_group: results['matrix'][i]},
                metadata={'group': node_group}
            ) for i, data in enumerate(results['data'])]

        return nodes

    def _get_db_client(self, group_name: str):
        if group_name == LAZY_GRAPH_NODE:
            return self._entity_db
        elif group_name == LAZY_GRAPH_EDGE:
            return self._relation_db
        else:
            raise NotImplementedError('NanoGraphStore only support query on entity or relation')