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
        _id: Optional[str] = None,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        parent: Optional["DocNode"] = None,
        children: Optional[List["DocNode"]] = None,
    ) -> None:
        self._id: str = _id if _id else str(uuid.uuid4())
        self.text: Optional[str] = text
        self.embedding: Optional[List[float]] = embedding
        self._metadata: Dict[str, Any] = metadata if metadata is not None else {}
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
        # A mapping of relationships to other node information.
        self.parent = parent
        self.children = children if children is not None else []

    @property
    def node_id(self) -> str:
        return self._id

    @property
    def metadata(self) -> Dict[str, Any]:
        # fetch the original metadata if needed
        if not self._metadata and self.root_node:
            self._metadata = self.root_node.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        self._metadata = metadata

    @property
    def root_node(self) -> Optional["DocNode"]:
        root = self.parent
        while root and root.parent:
            root = root.parent
        return root

    def __str__(self) -> str:
        return f"DocNode(id: {self.node_id}, text: {self.get_content()}) parent: {self.parent.node_id}, children: {[c.node_id for c in self.children]}"

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

    @property
    def node_info(self) -> Dict[str, Any]:
        """Deprecated: Get node info."""
        return self.get_node_info()

    def get_node_info(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "excluded_embed_metadata_keys": self.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": self.excluded_llm_metadata_keys,
            "parent": self.parent,
            "children": self.children,
        }

