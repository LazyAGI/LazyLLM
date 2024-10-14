from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
import chromadb
from lazyllm import LOG, config
from chromadb.api.models.Collection import Collection
import pymilvus
import threading
import json
import time


LAZY_ROOT_NAME = "lazyllm_root"
EMBED_DEFAULT_KEY = '__default__'
config.add("rag_store_type", str, "map", "RAG_STORE_TYPE")  # "map", "chroma"
config.add("rag_persistent_path", str, "./lazyllm_chroma", "RAG_PERSISTENT_PATH")


class MetadataMode(str, Enum):
    ALL = auto()
    EMBED = auto()
    LLM = auto()
    NONE = auto()


class DocNode:
    def __init__(self, uid: Optional[str] = None, text: Optional[str] = None, group: Optional[str] = None,
                 embedding: Optional[Dict[str, List[float]]] = None, parent: Optional["DocNode"] = None,
                 metadata: Optional[Dict[str, Any]] = None, classfication: Optional[str] = None):
        self.uid: str = uid if uid else str(uuid.uuid4())
        self.text: Optional[str] = text
        self.group: Optional[str] = group
        self.embedding: Optional[Dict[str, List[float]]] = embedding or None
        self._metadata: Dict[str, Any] = metadata or {}
        # Metadata keys that are excluded from text for the embed model.
        self._excluded_embed_metadata_keys: List[str] = []
        # Metadata keys that are excluded from text for the LLM.
        self._excluded_llm_metadata_keys: List[str] = []
        self.parent: Optional["DocNode"] = parent
        self.children: Dict[str, List["DocNode"]] = defaultdict(list)
        self.is_saved: bool = False
        self._docpath = None
        self._lock = threading.Lock()
        self._embedding_state = set()
        # store will create index cache for classfication to speed up retrieve
        self._classfication = classfication

    @property
    def root_node(self) -> Optional["DocNode"]:
        root = self.parent
        while root and root.parent:
            root = root.parent
        return root or self

    @property
    def metadata(self) -> Dict:
        return self.root_node._metadata

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        self._metadata = metadata

    @property
    def excluded_embed_metadata_keys(self) -> List:
        return self.root_node._excluded_embed_metadata_keys

    @excluded_embed_metadata_keys.setter
    def excluded_embed_metadata_keys(self, excluded_embed_metadata_keys: List) -> None:
        self._excluded_embed_metadata_keys = excluded_embed_metadata_keys

    @property
    def excluded_llm_metadata_keys(self) -> List:
        return self.root_node._excluded_llm_metadata_keys

    @excluded_llm_metadata_keys.setter
    def excluded_llm_metadata_keys(self, excluded_llm_metadata_keys: List) -> None:
        self._excluded_llm_metadata_keys = excluded_llm_metadata_keys

    @property
    def docpath(self) -> str:
        return self.root_node._docpath or ''

    @docpath.setter
    def docpath(self, path):
        assert not self.parent, 'Only root node can set docpath'
        self._docpath = str(path)

    def get_children_str(self) -> str:
        return str(
            {key: [node.uid for node in nodes] for key, nodes in self.children.items()}
        )

    def get_parent_id(self) -> str:
        return self.parent.uid if self.parent else ""

    def __str__(self) -> str:
        return (
            f"DocNode(id: {self.uid}, group: {self.group}, text: {self.get_text()}) parent: {self.get_parent_id()}, "
            f"children: {self.get_children_str()}"
        )

    def __repr__(self) -> str:
        return str(self) if config["debug"] else f'<Node id={self.uid}>'

    def __eq__(self, other):
        if isinstance(other, DocNode):
            return self.uid == other.uid
        return False

    def __hash__(self):
        return hash(self.uid)

    def has_missing_embedding(self, embed_keys: Union[str, List[str]]) -> List[str]:
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        assert len(embed_keys) > 0, "The ebmed_keys to be checked must be passed in."
        if self.embedding is None: return embed_keys
        return [k for k in embed_keys if k not in self.embedding.keys() or self.embedding.get(k, [-1])[0] == -1]

    def do_embedding(self, embed: Dict[str, Callable]) -> None:
        generate_embed = {k: e(self.get_text(MetadataMode.EMBED)) for k, e in embed.items()}
        with self._lock:
            self.embedding = self.embedding or {}
            self.embedding = {**self.embedding, **generate_embed}
        self.is_saved = False

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
        return dict(text=self.text, embedding=self.embedding, metadata=self.metadata)

# ---------------------------------------------------------------------------- #

class BaseStore(ABC):
    @abstractmethod
    def update_nodes(self, nodes: List[DocNode]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def get_node(self, group_name: str, node_id: str) -> Optional[DocNode]:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def remove_nodes(self, nodes: List[DocNode]) -> None:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def has_group(self, group_name: str) -> bool:
        raise NotImplementedError("not implemented yet.")

    @abstractmethod
    def traverse_group(self, group_name: str) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")

    # XXX NOTE the following APIs should be private.

    @abstractmethod
    def get_nodes_by_files(self, files: List[str]) -> List[DocNode]:
        raise NotImplementedError("not implemented yet.")

# ---------------------------------------------------------------------------- #

class MapStore(BaseStore):
    def __init__(self, node_groups: List[str]):
        # Dict[group_name, Dict[uuid, DocNode]]
        self._group2docs: Dict[str, Dict[str, DocNode]] = {
            group: {} for group in node_groups
        }
        self._file_node_map = {}

    # override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            if node.group == LAZY_ROOT_NAME and "file_name" in node.metadata:
                self._file_node_map[node.metadata["file_name"]] = node
            self._group2docs[node.group][node.uid] = node

    # override
    def get_node(self, group_name: str, node_id: str) -> Optional[DocNode]:
        return self._group2docs.get(group_name, {}).get(node_id)

    # override
    def remove_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            assert node.group in self._group2docs, f"Unexpected node group {node.group}"
            self._group2docs[node.group].pop(node.uid, None)

    # override
    def has_group(self, group_name: str) -> bool:
        return group_name in self._group2docs

    # override
    def traverse_group(self, group_name: str) -> List[DocNode]:
        return list(self._group2docs.get(group_name, {}).values())

    def get_group_docs(self) -> Dict[str, Dict[str, DocNode]]:
        return self._group2docs

    def find_node_by_uid(self, uid: str) -> Optional[DocNode]:
        for docs in self._group2docs.values():
            doc = docs.get(uid)
            if doc:
                return doc
        return None

    # XXX NOTE the following APIs should be private.

    # override
    def get_nodes_by_files(self, files: List[str]) -> List[DocNode]:
        nodes = []
        for file in files:
            if file in self._file_node_map:
                nodes.append(self._file_node_map[file])
        return nodes

# ---------------------------------------------------------------------------- #

class ChromadbStore(BaseStore):
    def __init__(
        self, node_groups: List[str], embed: Dict[str, Callable]
    ) -> None:
        self._map_store = MapStore(node_groups)
        self._db_client = chromadb.PersistentClient(path=config["rag_persistent_path"])
        LOG.success(f"Initialzed chromadb in path: {config['rag_persistent_path']}")
        self._collections: Dict[str, Collection] = {
            group: self._db_client.get_or_create_collection(group)
            for group in node_groups
        }
        self._placeholder = {k: [-1] * len(e("a")) for k, e in embed.items()} if embed else {EMBED_DEFAULT_KEY: []}
        self._load_store()

    # override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    # override
    def get_node(self, group_name: str, node_id: str) -> Optional[DocNode]:
        return self._map_store.get_node(group_name, node_id)

    # override
    def remove_nodes(self, nodes: List[DocNode]) -> None:
        return self._map_store.remove_nodes(nodes)

    # override
    def has_group(self, group_name: str) -> bool:
        return self._map_store.has_group(group_name)

    # override
    def traverse_group(self, group_name: str) -> List[DocNode]:
        return self._map_store.traverse_group(group_name)

    def _load_store(self) -> None:
        if not self._collections[LAZY_ROOT_NAME].peek(1)["ids"]:
            LOG.info("No persistent data found, skip the rebuilding phrase.")
            return

        # Restore all nodes
        for group in self._collections.keys():
            results = self._peek_all_documents(group)
            nodes = self._build_nodes_from_chroma(results)
            self._map_store.update_nodes(nodes)

        # Rebuild relationships
        group2docs = self._map_store.get_group_docs()
        for group, nodes_dict in group2docs.items():
            for node in nodes_dict.values():
                if node.parent:
                    parent_uid = node.parent
                    parent_node = self._map_store.find_node_by_uid(parent_uid)
                    node.parent = parent_node
                    parent_node.children[node.group].append(node)
            LOG.debug(f"build {group} nodes from chromadb: {nodes_dict.values()}")
        LOG.success("Successfully Built nodes from chromadb.")

    def _save_nodes(self, nodes: List[DocNode]) -> None:
        if not nodes:
            return
        # Note: It's caller's duty to make sure this batch of nodes has the same group.
        group = nodes[0].group
        ids, embeddings, metadatas, documents = [], [], [], []
        collection = self._collections.get(group)
        assert (
            collection
        ), f"Group {group} is not found in collections {self._collections}"
        for node in nodes:
            if node.is_saved:
                continue
            if miss_keys := node.has_missing_embedding(self._placeholder.keys()):
                node.embedding = node.embedding or {}
                node.embedding = {**node.embedding, **{k: self._placeholder[k] for k in miss_keys}}
            metadata = self._make_chroma_metadata(node)
            metadata["embedding"] = json.dumps(node.embedding)
            ids.append(node.uid)
            embeddings.append([item for subembed in node.embedding.values() for item in subembed])
            metadatas.append(metadata)
            documents.append(node.get_text())
            node.is_saved = True
        if ids:
            collection.upsert(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            LOG.debug(f"Saved {group} nodes {ids} to chromadb.")

    # XXX NOTE should be private.
    # override
    def get_nodes_by_files(self, files: List[str]) -> List[DocNode]:
        return self._map_store.get_nodes_by_files(files)

    def _build_nodes_from_chroma(self, results: Dict[str, List]) -> List[DocNode]:
        nodes: List[DocNode] = []
        for i, uid in enumerate(results['ids']):
            chroma_metadata = results['metadatas'][i]
            node = DocNode(
                uid=uid,
                text=results["documents"][i],
                group=chroma_metadata["group"],
                embedding=json.loads(chroma_metadata['embedding']),
                parent=chroma_metadata["parent"],
            )
            node.is_saved = True
            nodes.append(node)
        return nodes

    def _make_chroma_metadata(self, node: DocNode) -> Dict[str, Any]:
        metadata = {
            "group": node.group,
            "parent": node.parent.uid if node.parent else "",
        }
        return metadata

    def _peek_all_documents(self, group: str) -> Dict[str, List]:
        assert group in self._collections, f"group {group} not found."
        collection = self._collections[group]
        return collection.peek(collection.count())

# ---------------------------------------------------------------------------- #

class MilvusStore(BaseStore):
    def __init__(self, node_groups: List[str], uri: str):
        self._client = pymilvus.MilvusClient(uri=uri)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name='uid',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=65535,
            is_primary=True,
        )
        schema.add_field(
            field_name='text',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name='group',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=1024,
        )
        schema.add_field(
            field_name='embedding_json',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name='parent_uid',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=1024,
        )
        schema.add_field(
            field_name='metadata_json',
            datatype=pymilvus.DataType.VARCHAR,
            max_length=65535,
        )

        for group in node_groups:
            if group not in self._client.list_collections():
                self._client.create_collection(collection_name=group, schema=schema)

        self._map_store = MapStore(node_groups)
        self._load_store()

    # override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    # override
    def get_node(self, group_name: str, node_id: str) -> Optional[DocNode]:
        return self._map_store.get_node(group_name, node_id)

    # override
    def remove_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._client.delete(collection_name=node.group, filter=f'uid in ["{node.uid}"]')
        self._map_store.remove_nodes(nodes)

    # override
    def has_group(self, group_name: str) -> bool:
        return self._map_store.has_group(group_name)

    # override
    def traverse_group(self, group_name: str) -> List[DocNode]:
        return self._map_store.traverse_group(group_name)

    def _load_store(self) -> None:
        groups = self._client.list_collections()
        for group in groups:
            results = self._client.query(collection_name=group,
                                         output_fields=["count(*)"],
                                         limit=1)
            if len(results) != 1:
                raise ValueError(f"query count(*) of collection [{group}] failed.")

            count = int(results[0]['count(*)'])
            if count == 0:
                continue

            results = self._client.query(collection_name=group,
                                         query_expression="uid in []",
                                         output_fields=["*"],
                                         limit=count)
            for record in results:
                doc_node = DocNode(
                    uid=record['uid'],
                    text=record['text'],
                    group=record['group'],
                    embedding=json.loads(record['embedding_json']),
                    parent=record['parent_uid'],  # NOTE: will be updated later
                    metadata=json.loads(record['metadata_json']),
                )
                self._map_store.update_nodes([doc_node])

        # update doc's parent
        group_docs = self._map_store.get_group_docs()
        for group, docs in group_docs.items():
            for uid, doc in docs.items():
                # before find_node_by_uid() `doc.parent` is the parent's uid
                doc.parent = self._map_store.find_node_by_uid(doc.parent)

    def _save_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            data = {
                'uid': node.uid,
                'text': node.text,
                'group': node.group,
                'embedding_json': json.dumps(node.embedding),
                'parent_uid': node.parent.uid if node.parent else '',
                'metadata_json': json.dumps(node._metadata)
            }
            self._client.upsert(collection_name=node.group, data=data)

    # XXX NOTE the following APIs should be private.

    # override
    def get_nodes_by_files(self, files: List[str]) -> List[DocNode]:
        return self._map_store.get_nodes_by_files(files)
