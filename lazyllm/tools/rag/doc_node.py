from typing import Optional, Dict, Any, Union, Callable, List
from enum import Enum, auto
from collections import defaultdict
from PIL import Image
from lazyllm import config, reset_on_pickle
from lazyllm.components.utils.file_operate import image_to_base64
from .global_metadata import RAG_DOC_PATH
import uuid
import threading
import time
import copy

class MetadataMode(str, Enum):
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()


@reset_on_pickle(('_lock', threading.Lock))
class DocNode:
    def __init__(self, uid: Optional[str] = None, content: Optional[Union[str, List[Any]]] = None,
                 group: Optional[str] = None, embedding: Optional[Dict[str, List[float]]] = None,
                 parent: Optional["DocNode"] = None, metadata: Optional[Dict[str, Any]] = None,
                 global_metadata: Optional[Dict[str, Any]] = None, *, text: Optional[str] = None):
        if text and content:
            raise ValueError('`text` and `content` cannot be set at the same time.')

        self._uid: str = uid if uid else str(uuid.uuid4())
        self._content: Optional[Union[str, List[Any]]] = content if content else text
        self._group: Optional[str] = group
        self._embedding: Optional[Dict[str, List[float]]] = embedding or {}
        self._metadata: Dict[str, Any] = metadata or {}
        # Metadata keys that are excluded from text for the embed model.
        self._excluded_embed_metadata_keys: List[str] = []
        # Metadata keys that are excluded from text for the LLM.
        self._excluded_llm_metadata_keys: List[str] = []
        self._parent: Optional["DocNode"] = parent
        self._children: Dict[str, List["DocNode"]] = defaultdict(list)
        self._lock = threading.Lock()
        self._embedding_state = set()
        self.relevance_score = None
        self.similarity_score = None

        if global_metadata and parent:
            raise ValueError('only ROOT node can set global metadata.')
        self._global_metadata = global_metadata or {}

    @property
    def text(self) -> str:
        if isinstance(self._content, str):
            return self._content
        elif isinstance(self._content, list):
            if unexcepted := set([type(ele) for ele in self._content if not isinstance(ele, str)]):
                raise TypeError(f"Found non-string element in content: {unexcepted}")
            return '\n'.join(self._content)
        else:
            raise TypeError(f"content type '{type(self._content)}' is neither a str nor a list")

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, v: Optional[Dict[str, List[float]]]):
        self._embedding = v

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, v: Optional["DocNode"]):
        self._parent = v

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, v: Dict[str, List["DocNode"]]):
        self._children = v

    @property
    def root_node(self) -> Optional["DocNode"]:
        root = self.parent
        while root and root.parent:
            root = root.parent
        return root or self

    @property
    def is_root_node(self) -> bool:
        return (not self.parent)

    @property
    def global_metadata(self) -> Dict[str, Any]:
        return self.root_node._global_metadata

    @global_metadata.setter
    def global_metadata(self, global_metadata: Dict) -> None:
        if self.parent:
            raise ValueError("only root node can set global metadata.")
        self._global_metadata = global_metadata

    @property
    def metadata(self) -> Dict:
        return {**self.root_node._metadata, **self._metadata}

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        self._metadata = metadata

    @property
    def excluded_embed_metadata_keys(self) -> List:
        return list(set(self.root_node._excluded_embed_metadata_keys + self._excluded_embed_metadata_keys))

    @excluded_embed_metadata_keys.setter
    def excluded_embed_metadata_keys(self, excluded_embed_metadata_keys: List) -> None:
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys

    @property
    def excluded_llm_metadata_keys(self) -> List:
        return list(set(self.root_node._excluded_llm_metadata_keys + self._excluded_llm_metadata_keys))

    @excluded_llm_metadata_keys.setter
    def excluded_llm_metadata_keys(self, excluded_llm_metadata_keys: List) -> None:
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

    @property
    def docpath(self) -> str:
        return self.root_node._global_metadata.get(RAG_DOC_PATH, '')

    @docpath.setter
    def docpath(self, path):
        assert not self.parent, 'Only root node can set docpath'
        self._global_metadata[RAG_DOC_PATH] = str(path)

    def get_children_str(self) -> str:
        return str(
            {key: [node._uid for node in nodes] for key, nodes in self.children.items()}
        )

    def get_parent_id(self) -> str:
        return self.parent._uid if self.parent else ""

    def __str__(self) -> str:
        return (
            f"DocNode(id: {self._uid}, group: {self._group}, content: {self._content}) parent: {self.get_parent_id()}, "
            f"children: {self.get_children_str()}"
        )

    def __repr__(self) -> str:
        return str(self) if config["debug"] else f'<Node id={self._uid}>'

    def __eq__(self, other):
        if isinstance(other, DocNode):
            return self._uid == other._uid
        return False

    def __hash__(self):
        return hash(self._uid)

    def has_missing_embedding(self, embed_keys: Union[str, List[str]]) -> List[str]:
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        assert len(embed_keys) > 0, "The ebmed_keys to be checked must be passed in."
        if self.embedding is None: return embed_keys
        return [k for k in embed_keys if k not in self.embedding]

    def do_embedding(self, embed: Dict[str, Callable]) -> None:
        generate_embed = {k: e(self.get_text(MetadataMode.EMBED)) for k, e in embed.items()}
        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding = {**self.embedding, **generate_embed}

    def check_embedding_state(self, embed_key: str) -> None:
        while True:
            with self._lock:
                if not self.has_missing_embedding(embed_key):
                    self._embedding_state.discard(embed_key)
                    break
            time.sleep(1)

    def get_content(self) -> str:
        return self.get_text(MetadataMode.LLM)

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

    def get_text(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        metadata_str = self.get_metadata_str(metadata_mode).strip()
        if not metadata_str:
            return self.text if self.text else ""
        return f"{metadata_str}\n\n{self.text}".strip()

    def to_dict(self) -> Dict:
        return dict(content=self._content, embedding=self.embedding, metadata=self.metadata)

    def with_score(self, score):
        node = copy.copy(self)
        node.relevance_score = score
        return node

    def with_sim_score(self, score):
        node = copy.copy(self)
        node.similarity_score = score
        return node


class QADocNode(DocNode):
    def __init__(self, query: str, answer: str, uid: Optional[str] = None, group: Optional[str] = None,
                 embedding: Optional[Dict[str, List[float]]] = None, parent: Optional["DocNode"] = None,
                 metadata: Optional[Dict[str, Any]] = None, *, text: Optional[str] = None):
        super().__init__(uid, query, group, embedding, parent, metadata, None, text=text)
        self._answer = answer.strip()

    def get_text(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        if metadata_mode == MetadataMode.LLM:
            return f'query:\n{self.text}\nanswer\n{self._answer}'
        return super().get_text(metadata_mode)


class ImageDocNode(DocNode):
    def __init__(self, image_path: str, uid: Optional[str] = None, group: Optional[str] = None,
                 embedding: Optional[Dict[str, List[float]]] = None, parent: Optional["DocNode"] = None,
                 metadata: Optional[Dict[str, Any]] = None, global_metadata: Optional[Dict[str, Any]] = None,
                 *, text: Optional[str] = None):
        super().__init__(uid, None, group, embedding, parent, metadata, global_metadata=global_metadata, text=text)
        self._image_path = image_path.strip()
        self._modality = "image"

    def do_embedding(self, embed: Dict[str, Callable]) -> None:
        for k, e in embed.items():
            emb = e(self.get_content(MetadataMode.EMBED), modality=self._modality)
            generate_embed = {k: emb[0]}

        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding = {**self.embedding, **generate_embed}

    def get_content(self, metadata_mode=MetadataMode.LLM) -> str:
        if metadata_mode == MetadataMode.LLM:
            return Image.open(self._image_path)
        elif metadata_mode == MetadataMode.EMBED:
            image_base64, mime = image_to_base64(self._image_path)
            return [f"data:{mime};base64,{image_base64}"]
        else:
            return self.get_text()

    @property
    def image_path(self):
        return self._image_path

    def get_text(self) -> str:  # Disable access to self._content
        return self._image_path

    @property
    def text(self) -> str:  # Disable access to self._content
        return self._image_path
