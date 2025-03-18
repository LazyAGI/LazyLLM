import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, TypedDict

# from lazyllm.thirdparty import nano_vectordb as
from nano_vectordb import NanoVectorDB
from .graph_node import GraphEntityNode, GraphRelationNode
from ..utils import validate_typed_dict

# ---------------------------------------------------------------------------- #


class BaseGraphERStore(ABC):
    _registry = {}

    @classmethod
    def register_subclass(cls, subclass):
        cls._registry[subclass.__name__] = subclass

    @classmethod
    def create_instance(cls, subclass_name, embed: Callable, root_path: str, name_space: str, config: Dict[str, Any]):
        subclass = cls._registry.get(subclass_name)
        if subclass:
            return subclass(embed, root_path, name_space, config)
        else:
            raise ValueError(f"Subclass '{subclass_name}' not found")

    def __init__(self, embed: Callable, root_path: str, name_space: str, config: Dict[str, Any]) -> None:
        self._embed_func = embed
        self._root_path = root_path
        self._name_space = name_space
        self._config = config

    @abstractmethod
    def query_on_entity(self, query: str, topk, similarity_cut_off, **kwargs) -> List[GraphEntityNode]:
        pass

    @abstractmethod
    def query_on_relationship(self, query: str, topk, similarity_cut_off, **kwargs) -> List[GraphRelationNode]:
        pass


class NanoGraphERStoreConfig(TypedDict):
    embedding_dim: int


class NanoDBGraphERStore(BaseGraphERStore):
    json_template = dict(embedding_dim=1024, data=[], matrix="")
    ENTITY_STORE_PATH = "vdb_entities.json"
    RELATION_STORE_PATH = "vdb_relationships.json"

    def __init__(self, embed: Callable, root_path: str, name_space: str, config: NanoGraphERStoreConfig) -> None:
        rt, msg = validate_typed_dict(config, NanoGraphERStoreConfig)
        if not rt:
            raise ValueError(msg)
        super().__init__(embed, root_path, name_space, config)
        self.json_template["embedding_dim"] = config["embedding_dim"]
        self._check_path()
        self._entity_db_client = NanoVectorDB(
            embedding_dim=config["embedding_dim"],
            storage_file=os.path.join(root_path, name_space, self.ENTITY_STORE_PATH),
        )
        self._relation_db_client = NanoVectorDB(
            embedding_dim=config["embedding_dim"],
            storage_file=os.path.join(root_path, name_space, self.RELATION_STORE_PATH),
        )

    def _query(self, db_client: NanoVectorDB, query: str, topk: int, similarity_cut_off: float) -> List[dict]:
        query_embedding = self._embed_func(query)
        query_embedding = np.array(query_embedding)
        results = db_client.query(query=query_embedding, top_k=topk, better_than_threshold=similarity_cut_off)
        # remove unuseful keys: __id__, __metrics__
        results = [
            {
                **{k: v for k, v in dp.items() if not k.startswith("__")},
            }
            for dp in results
        ]
        return results

    def query_on_entity(self, query: str, topk: int = 30, similarity_cut_off: float = 0.3) -> List[GraphEntityNode]:
        return self._query(self._entity_db_client, query, topk, similarity_cut_off)

    def query_on_relationship(
        self, query: str, topk: int = 30, similarity_cut_off: float = 0.3
    ) -> List[GraphRelationNode]:
        return self._query(self._relation_db_client, query, topk, similarity_cut_off)

    def _check_path(self):
        entity_store_path = os.path.join(self._root_path, self._name_space, self.ENTITY_STORE_PATH)
        if not os.path.exists(entity_store_path):
            with open(entity_store_path, 'w') as f:
                json.dump(self.json_template, f)
        relation_store_path = os.path.join(self._root_path, self._name_space, self.RELATION_STORE_PATH)
        if not os.path.exists(relation_store_path):
            with open(relation_store_path, 'w') as f:
                json.dump(self.json_template, f)


BaseGraphERStore.register_subclass(NanoDBGraphERStore)
