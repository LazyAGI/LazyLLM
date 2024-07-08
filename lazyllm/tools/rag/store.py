from enum import Enum, auto
import uuid
from typing import Any, Dict, List, Optional


class MetadataMode(str, Enum):
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()


class DocNode:
    def __init__(
        self,
        uid: Optional[str] = None,
        text: Optional[str] = None,
        ntype: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        parent: Optional["DocNode"] = None,
    ) -> None:
        self.uid: str = uid if uid else str(uuid.uuid4())
        self.text: Optional[str] = text
        self.ntype: Optional[str] = ntype
        self.embedding: Optional[List[float]] = embedding
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        # Metadata keys that are excluded from text for the embed model.
        self.excluded_embed_metadata_keys: List[str] = (
            excluded_embed_metadata_keys
            if excluded_embed_metadata_keys is not None
            else []
        )
        # Metadata keys that are excluded from text for the LLM.
        self.excluded_llm_metadata_keys: List[str] = (
            excluded_llm_metadata_keys if excluded_llm_metadata_keys is not None else []
        )
        # Relationships to other node.
        self.parent = parent
        self.children: Dict[str, List["DocNode"]] = {}

    @property
    def root_node(self) -> Optional["DocNode"]:
        root = self.parent
        while root and root.parent:
            root = root.parent
        return root

    def __str__(self) -> str:
        children_str = {
            key: [node.uid for node in self.children[key]]
            for key in self.children.keys()
        }
        return (
            f"DocNode(id: {self.uid}, ntype: {self.ntype}, text: {self.get_content()}) parent: "
            f"{self.parent.uid if self.parent else None}, children: {children_str}"
        )

    def __repr__(self) -> str:
        return str(self)

    def get_embedding(self) -> List[float]:
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text if self.text else ""

        return f"{metadata_str}\n\n{self.text}".strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in metadata_keys:
                    metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in metadata_keys:
                    metadata_keys.remove(key)

        return "\n".join([f"{key}: {self.metadata[key]}" for key in metadata_keys])

    def get_text(self) -> str:
        return self.get_content(metadata_mode=MetadataMode.NONE)


# TODO: Have a common Base store class
class MapStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, DocNode]] = {}

    def add_nodes(self, category: str, nodes: List[DocNode]):
        if category not in self.store:
            self.store[category] = {}

        for node in nodes:
            self.store[category][node.uid] = node

    def has_nodes(self, category: str) -> bool:
        return category in self.store.keys()

    def get_node(self, category: str, node_id: str) -> Optional[DocNode]:
        return self.store.get(category, {}).get(node_id)

    def delete_node(self, category: str, node_id: str):
        if category in self.store and node_id in self.store[category]:
            del self.store[category][node_id]
        # TODO: delete node's relationship

    def traverse_nodes(self, category: str) -> List[DocNode]:
        return list(self.store.get(category, {}).values())


class ChromadbStore:
    pass
