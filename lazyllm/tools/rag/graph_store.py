import os
import time
import json
import base64
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union
from lazyllm.thirdparty import nano_vectordb as nanodb
from lazyllm import LOG
from .store_base import LAZY_GRAPH_NODE, LAZY_GRAPH_EDGE
from .doc_node import GraphEntityNode, GraphRelationNode

# ---------------------------------------------------------------------------- #

class NanoDBGraphStore:
    json_template = dict(embedding_dim = 1024, data=[], matrix="")
    def __init__(self, embed: Callable, embed_dims: Union[int, Dict[str, int]],
                 dir: str, **kwargs) -> None:
        self.json_template["embedding_dim"] = embed_dims
        self._check_dir(dir)
        entity_db = nanodb.NanoVectorDB(embedding_dim=embed_dims,
                                        storage_file=os.path.join(dir, 'vdb_entities.json'))
        relation_db = nanodb.NanoVectorDB(embedding_dim=embed_dims,
                                          storage_file=os.path.join(dir, 'vdb_relationships.json'))
        self._db_client = {LAZY_GRAPH_NODE: entity_db, LAZY_GRAPH_EDGE: relation_db}
        LOG.success(f"Initialzed nanodb in path: {dir}")

        self._uid_to_nodes = {LAZY_GRAPH_NODE: {}, LAZY_GRAPH_EDGE: {}}
        self._embed_func = embed
        self._load_store()

    def update_entity_nodes(self, nodes: List[GraphEntityNode]) -> None:
        if nodes:
            up, ins = self._update_nodes(LAZY_GRAPH_NODE, nodes)
            LOG.info(f"Updated {len(up)} entity and inserted {len(ins)} entity into nano db ")

    def update_relation_nodes(self, nodes: List[GraphRelationNode]) -> None:
        if nodes:
            up, ins = self._update_nodes(LAZY_GRAPH_EDGE, nodes)
            LOG.info(f"Updated {len(up)} edge and inserted {len(ins)} edge into nano db ")

    def remove_entity_nodes(self, uids: Optional[List[str]] = None) -> None:
        self._remove_nodes(LAZY_GRAPH_NODE, uids)
        LOG.info(f"Deleted {len(uids)} entity from nano db.")

    def remove_relation_nodes(self, uids: Optional[List[str]] = None) -> None:
        self._remove_nodes(LAZY_GRAPH_EDGE, uids)
        LOG.info(f"Deleted {len(uids)} edge from nano db.")

    def get_entity_nodes(self, uids: List[str] = None) -> List[GraphEntityNode]:
        if uids:
            return [self._uid_to_nodes[LAZY_GRAPH_NODE].get(uid) for uid in uids]
        return list(self._uid_to_nodes[LAZY_GRAPH_NODE].values())

    def get_relation_nodes(self, uids: List[str] = None) -> List[GraphRelationNode]:
        if uids:
            return [self._uid_to_nodes[LAZY_GRAPH_EDGE].get(uid) for uid in uids]
        return list(self._uid_to_nodes[LAZY_GRAPH_EDGE].values())

    def query_on_entity(self, query: str, topk: int = 10,
                     similarity_cut_off: float = 0.3) -> List[GraphEntityNode]:
        uidset = self._cosine_query(query, topk, LAZY_GRAPH_NODE, similarity_cut_off)
        return self.get_entity_nodes(list(uidset))

    def query_on_relationship(self, query: str, topk: int = 10,
                     similarity_cut_off: float = 0.3) -> List[GraphRelationNode]:
        uidset = self._cosine_query(query, topk, LAZY_GRAPH_EDGE, similarity_cut_off)
        return self.get_relation_nodes(list(uidset))

    def _cosine_query(self, query: str, topk: int, store_key: str,
                      cosine_threshold: float = 0.3) -> List[str]:
        uidset = set()
        db_client = self._db_client.get(store_key)
        query_embedding = self._embed_func(query)
        results =  db_client.query(query = query_embedding,
                                    top_k = topk,
                                    better_than_threshold = cosine_threshold)

        results = [dp["__id__"] for dp in results]
        uidset.update(results)

        return uidset

    def _update_nodes(self, store_key: str, nodes: List[Union[GraphEntityNode, GraphRelationNode]])-> Dict[str, int]:
        db_client = self._db_client.get(store_key)
        if db_client is None:
            return 0, 0
        data = self._format_nodes_to_json(nodes)
        results = db_client.upsert(data)
        db_client.save()
        self._update_uid_mapping(store_key, nodes)
        return results['update'], results['insert']

    def _remove_nodes(self, store_key: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            db_client = self._db_client.get(store_key)
            db_client.delete(uids)
            db_client.save()
            self._remove_uid_mapping(store_key, uids)

    def _load_store(self) -> None:
        # step 1 : load entity vdb
        entity_storage = self._load_storage(LAZY_GRAPH_NODE)
        nodes = self._build_entity_nodes_from_json(entity_storage)
        self._update_uid_mapping(LAZY_GRAPH_NODE, nodes)

        # step 2 : load relationship vdb
        edge_storage = self._load_storage(LAZY_GRAPH_EDGE)
        nodes = self._build_relation_nodes_from_json(edge_storage)
        self._update_uid_mapping(LAZY_GRAPH_EDGE, nodes)

        LOG.success("Successfully Built nodes from nanodb.")

    def _load_storage(self, store_key: str) -> Dict[str, Any]:
        file_name = self._db_client.get(store_key).storage_file
        with open(file_name, encoding="utf-8") as f:
            storage = json.load(f)

        if len(storage)==0:
            LOG.info("No persistent data found, skip the entity loading phrase.")
            return

        matrix = np.frombuffer(base64.b64decode(storage["matrix"]), dtype=np.float32)
        storage["matrix"] = matrix.reshape(-1, storage["embedding_dim"])

        return storage

    def _format_nodes_to_json(self, nodes: List[Union[GraphEntityNode, GraphRelationNode]]) -> Dict[str, List]:
        data = []
        current_time = time.time()
        for node in nodes:
            node_json = node.to_dict()
            node_json['__created_time__'] = current_time
            node_json['__vector__'] = np.array(node_json['__vector__'])
            data.append(node_json)
        return data

    def _build_entity_nodes_from_json(self, storage: Dict[str, List]) -> List[GraphEntityNode]:
        if storage:
            nodes = []
            for i, data in enumerate(storage['data']):
                node = GraphEntityNode(
                    uid=data['__id__'],
                    entity_name=data['entity_name'],
                    embedding=storage['matrix'][i]
                )
                nodes.append(node)
            return nodes

    def _build_relation_nodes_from_json(self, storage: Dict[str, List]) -> List[GraphRelationNode]:
        if storage:
            nodes = []
            for i, data in enumerate(storage['data']):
                node = GraphRelationNode(
                    uid=data['__id__'],
                    source=data['src_id'],
                    target=data['tgt_id'],
                    embedding=storage['matrix'][i]
                )
                nodes.append(node)
            return nodes

    def _update_uid_mapping(self, store_key: str, nodes: List[Union[GraphEntityNode, GraphRelationNode]] = None) -> None:
        if nodes:
            self._uid_to_nodes[store_key].update({node._uid: node for node in nodes})

    def _remove_uid_mapping(self, store_key: str, uids: List[str] = None) -> None:
        if uids:
            for uid in uids:
                self._uid_to_nodes[store_key].pop(uid)

    def _check_dir(self, dir: str):
        if os.path.isdir(dir):
            if not os.path.exists(os.path.join(dir, 'vdb_entities.json')):
                with open(os.path.join(dir, 'vdb_entities.json'), 'w') as f:
                    json.dump(self.json_template, f)
            if not os.path.exists(os.path.join(dir, 'vdb_relationships.json')):
                with open(os.path.join(dir, 'vdb_relationships.json'), 'w') as f:
                    json.dump(self.json_template, f)
        else:
            raise OSError(f"The dir passed into NanoDBGraphStore must be directory.")
