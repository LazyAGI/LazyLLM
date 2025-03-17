import html
import os
from dataclasses import dataclass
from typing import Any, List, Union, cast, TypedDict
import networkx as nx
import numpy as np
from lazyllm import LOG
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from .graph_node import GraphEntityNode, GraphRelationNode, GraphChunkNode
from ..utils import validate_typed_dict
from collections import defaultdict


class NetworkNode(TypedDict):
    node_id: str
    entity_type: str
    description: str
    source_id: str


class NetworkEdge(TypedDict):
    src_id: str
    tgt_id: str
    weight: float
    description: str
    keywords: str
    source_id: str


class BaseGraphNetworkStore(ABC):

    _registry = {}

    @classmethod
    def register_subclass(cls, subclass):
        cls._registry[subclass.__name__] = subclass

    @classmethod
    def create_instance(cls, subclass_name, root_path: str, name_space: str, config: dict = {}):
        subclass = cls._registry.get(subclass_name)
        if subclass:
            return subclass(root_path, name_space, config)
        else:
            raise ValueError(f"Subclass '{subclass_name}' not found")

    def __init__(self, root_path: str, name_space: str, config: dict = {}):
        self._root_path = root_path
        self._name_space = name_space
        self._config = config

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        pass

    @abstractmethod
    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        pass

    @abstractmethod
    def node_degree(self, node_id: str) -> int:
        pass

    @abstractmethod
    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Union[GraphEntityNode, None]:
        pass

    def get_neighbor_nodeids(self, source_node_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[GraphRelationNode, None]:
        pass

    @abstractmethod
    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        pass

    @abstractmethod
    def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        pass

    @classmethod
    def convert_node_to_entity(cls, node_id: str, network_node: NetworkNode, sep: str = "<SEP>") -> GraphEntityNode:
        return (
            GraphEntityNode(
                entity_name=node_id,
                entity_type=network_node["entity_type"],
                description=network_node["description"],
                source_chunk_ids=network_node["source_id"].split(sep),
            )
            if network_node
            else None
        )

    @classmethod
    def convert_edge_to_relationship(cls, source_node_id: str, target_node_id: str, network_edge: NetworkEdge, sep: str = "<SEP>") -> GraphRelationNode:
        return (
            GraphRelationNode(
                src_id=source_node_id,
                tgt_id=target_node_id,
                weight=network_edge["weight"],
                description=network_edge["description"],
                keywords=network_edge["keywords"],
                source_chunk_ids=network_edge["source_id"].split(sep),
            )
            if network_edge
            else None
        )
    
    def sort_relations_by_degree(self, nodes: List[GraphRelationNode]) -> List[GraphRelationNode]:
        def sort_func(node: GraphRelationNode):
            return self.edge_degree(node.src_id, node.tgt_id), node.weight
        return sorted(nodes, key=sort_func, reverse=True)

    def sort_entities_by_degree(self, nodes: List[GraphEntityNode]) -> List[GraphEntityNode]:
        return sorted(nodes, key=lambda node: self.node_degree(node.entity_name), reverse=True)

    def get_sorted_entities_from_relations(self, relations: List[GraphRelationNode]) -> List[GraphEntityNode]:
        pass
    
    def get_sorted_chunkids_from_relations(self, relations: List[GraphRelationNode]) -> List[str]:
        pass

    def get_sorted_relations_from_entities(self, entities: List[GraphEntityNode]) -> List[GraphRelationNode]:
        relations: List[GraphRelationNode] = []
        visited_relations = set()
        for entity in entities:
            neighbor_entity_names = self.get_neighbor_nodeids(entity.entity_name)
            relations_for_entity = [self.get_edge(entity.entity_name, ng_entity_name) for ng_entity_name in neighbor_entity_names]
            relations_for_entity = [ele for ele in relations_for_entity if ele]
            for r in relations_for_entity:
                if (r.src_id, r.tgt_id) not in visited_relations:
                    relations.append(r)
                    visited_relations.add((r.src_id, r.tgt_id))
                    visited_relations.add((r.tgt_id, r.src_id))
        # sort first order: edge degree, second order: weight
        return sorted(relations, key=lambda r: (self.edge_degree(r.src_id, r.tgt_id), r.weight), reverse=True)

    def sort_entitity_chunkids(self, entity: GraphEntityNode) -> List[str]:
        # For chunks belonging to different entities, their sorting order remains consistent with the retrieval order 
        # based on the query in the vector database. So we just need to sort chunks in entity.
        neighbor_entity_names = self.get_neighbor_nodeids(entity.entity_name)
        cids_in_neighbor_count = defaultdict(int)
        for entity_name in neighbor_entity_names:
            for cid in self.get_node(entity_name).source_chunk_ids:
                cids_in_neighbor_count[cid] += 1
        def sort_func(cid: str):
            if cid in cids_in_neighbor_count:
                return cids_in_neighbor_count[cid]
            return 0
        return sorted(entity.source_chunk_ids, key=sort_func, reverse=True)
        

class NetworkXStore(BaseGraphNetworkStore):
    GRAPH_STORE_PATH = "graph_chunk_entity_relation.graphml"
    def __init__(self, root_path: str, name_space: str, config: dict = {}):
        super().__init__(root_path, name_space, config)
        self._graphml_xml_file = os.path.join(root_path, name_space, self.GRAPH_STORE_PATH)
        if Path(self._graphml_xml_file).exists():
            try:
                self._graph = nx.read_graphml(self._graphml_xml_file)
                LOG.info(
                    f"Loaded graph from {self._graphml_xml_file} with {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges"
                )
            except Exception as e:
                LOG.warning(f"Failed loading graphml file, exception: {str(e)}")
                self._graph = nx.Graph()
        else:
            self._graph = nx.Graph()

    def index_done_callback(self):
        LOG.info(f"Writing graph with {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        self._graph.write_graphml(self._graphml_xml_file)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    def get_node(self, node_id: str) -> Union[GraphEntityNode, None]:
        network_node = self._graph.nodes.get(node_id)
        return self.convert_node_to_entity(node_id, network_node)

    def get_neighbor_nodeids(self, source_node_id: str) -> List[str]:
        if self._graph.has_node(source_node_id):
            return list(self._graph.neighbors(source_node_id))
        return []

    def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[GraphRelationNode, None]:
        network_edge = self._graph.edges.get((source_node_id, target_node_id))
        if network_edge:
            return self.convert_edge_to_relationship(source_node_id, target_node_id, network_edge)
        return None

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)


BaseGraphNetworkStore.register_subclass(NetworkXStore)
