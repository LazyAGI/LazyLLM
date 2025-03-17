from pathlib import Path
import time
import json
import base64
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, TypedDict
from lazyllm.thirdparty import nano_vectordb as nanodb
from lazyllm import LOG
from .graph_node import GraphEntityNode, GraphRelationNode
from ..utils import validate_typed_dict
from .graph_node import GraphChunkNode

# ---------------------------------------------------------------------------- #

class BaseGraphChunkStore(ABC):
    _registry = {}

    @classmethod
    def register_subclass(cls, subclass):
        cls._registry[subclass.__name__] = subclass

    @classmethod
    def create_instance(cls, subclass_name, root_path: str, name_space: str, config: Dict[str, Any] = {}):
        subclass = cls._registry.get(subclass_name)
        if subclass:
            return subclass(root_path, name_space, config)
        else:
            raise ValueError(f"Subclass '{subclass_name}' not found")

    def __init__(self, root_path, name_space: str, config: Dict[str, Any] = {}) -> None:
        self._root_path = root_path
        self._name_space = name_space
        self._config = config

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Union[GraphChunkNode, None]:
        pass

    @abstractmethod
    def get_chunks(self, ids: List[str]) -> List[GraphChunkNode]:
        pass

class JsonChunkStore(BaseGraphChunkStore):
    JSON_STORE_PATH = "kv_store_text_chunks.json"
    def __init__(self, root_path: str, name_space: str, config: dict = {}):
        super().__init__(root_path, name_space, config)
        self._file_name = str(Path(root_path).joinpath(name_space, self.JSON_STORE_PATH))
        self._data: dict = self._load_json(self._file_name)
    
    def _load_json(self, json_path):
        if not Path(json_path).exists():
            return {}
        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOG.error(f"Failed to load {json_path}")
            return {}
    
    def get_chunk(self, chunk_id: str) -> Union[GraphChunkNode, None]:
        chunk = self._data.get(chunk_id)
        return GraphChunkNode(chunk_id=chunk_id, **chunk) if chunk else None

    def get_chunks(self, chunk_ids: List[str]) -> List[GraphChunkNode]:
        return [self.get_chunk(chunk_id) for chunk_id in chunk_ids]

BaseGraphChunkStore.register_subclass(JsonChunkStore)