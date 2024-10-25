from typing import Any, Dict, List, Optional
import chromadb
from lazyllm import LOG, config
from chromadb.api.models.Collection import Collection
from .base_store import BaseStore
from .base_index import BaseIndex
from .doc_node import DocNode
import json

# ---------------------------------------------------------------------------- #

LAZY_ROOT_NAME = "lazyllm_root"
EMBED_DEFAULT_KEY = '__default__'
config.add("rag_store_type", str, "map", "RAG_STORE_TYPE")  # "map", "chroma"
config.add("rag_persistent_path", str, "./lazyllm_chroma", "RAG_PERSISTENT_PATH")

# ---------------------------------------------------------------------------- #

class StoreWrapper(BaseStore):
    def __init__(self, store: BaseStore):
        self._store = store
        self._name2index = {}

    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._store.update_nodes(nodes)
        self._update_indices(self._name2index, nodes)

    def get_group_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        return self._store.get_group_nodes(group_name, uids)

    def remove_group_nodes(self, group_name: str, uids: List[str] = None) -> None:
        self._store.remove_group_nodes(group_name, uids)
        self._remove_from_indices(self._name2index, uids, group_name)

    def group_is_active(self, group_name: str) -> bool:
        return self._store.group_is_active(group_name)

    def group_names(self) -> List[str]:
        return self._store.group_names()

    def register_index(self, type: str, index: BaseIndex) -> None:
        self._name2index[type] = index

    def remove_index(self, type: str) -> None:
        self._name2index.pop(type, None)

    def get_index(self, type: str) -> Optional[BaseIndex]:
        index = self._store.get_index(type)
        if not index:
            index = self._name2index.get(type)
        return index

# ---------------------------------------------------------------------------- #

class MapStore(BaseStore):
    def __init__(self, node_groups: List[str]):
        # Dict[group_name, Dict[uuid, DocNode]]
        self._group2docs: Dict[str, Dict[str, DocNode]] = {
            group: {} for group in node_groups
        }
        self._name2index = {}

    # override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._group2docs[node.group][node.uid] = node

        self._update_indices(self._name2index, nodes)

    # override
    def get_group_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        docs = self._group2docs.get(group_name)
        if not docs:
            return []

        if not uids:
            return list(docs.values())

        ret = []
        for uid in uids:
            doc = docs.get(uid)
            if doc:
                ret.append(doc)
        return ret

    # override
    def remove_group_nodes(self, group_name: str, uids: List[str] = None) -> None:
        if uids:
            docs = self._group2docs.get(group_name)
            if docs:
                self._remove_from_indices(self._name2index, uids)
                for uid in uids:
                    docs.pop(uid, None)
        else:
            docs = self._group2docs.pop(group_name, None)
            if docs:
                self._remove_from_indices(self._name2index, [doc.uid for doc in docs])

    # override
    def group_is_active(self, group_name: str) -> bool:
        docs = self._group2docs.get(group_name)
        return True if docs else False

    # override
    def group_names(self) -> List[str]:
        return self._group2docs.keys()

    # override
    def register_index(self, type: str, index: BaseIndex) -> None:
        self._name2index[type] = index

    # override
    def remove_index(self, type: str) -> None:
        self._name2index.pop(type, None)

    # override
    def get_index(self, type: str) -> Optional[BaseIndex]:
        return self._name2index.get(type)

    def find_node_by_uid(self, uid: str) -> Optional[DocNode]:
        for docs in self._group2docs.values():
            doc = docs.get(uid)
            if doc:
                return doc
        return None

# ---------------------------------------------------------------------------- #

class ChromadbStore(BaseStore):
    def __init__(
        self, node_groups: List[str], embed_dim: Dict[str, int]
    ) -> None:
        self._map_store = MapStore(node_groups)
        self._db_client = chromadb.PersistentClient(path=config["rag_persistent_path"])
        LOG.success(f"Initialzed chromadb in path: {config['rag_persistent_path']}")
        self._collections: Dict[str, Collection] = {
            group: self._db_client.get_or_create_collection(group)
            for group in node_groups
        }
        self._embed_dim = embed_dim

    # override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    # override
    def get_group_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
        return self._map_store.get_group_nodes(group_name, uids)

    # override
    def remove_group_nodes(self, group_name: str, uids: List[str]) -> None:
        if uids:
            self._delete_group_nodes(group_name, uids)
        else:
            self._db_client.delete_collection(name=group_name)
        return self._map_store.remove_group_nodes(group_name, uids)

    # override
    def group_is_active(self, group_name: str) -> bool:
        return self._map_store.group_is_active(group_name)

    # override
    def group_names(self) -> List[str]:
        return self._map_store.group_names()

    # override
    def register_index(self, type: str, index: BaseIndex) -> None:
        self._map_store.register_index(type, index)

    # override
    def remove_index(self, type: str) -> Optional[BaseIndex]:
        return self._map_store.remove_index(type)

    # override
    def get_index(self, type: str) -> Optional[BaseIndex]:
        return self._map_store.get_index(type)

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
        for group_name in self._map_store.group_names():
            nodes = self._map_store.get_group_nodes(group_name)
            for node in nodes:
                if node.parent:
                    parent_uid = node.parent
                    parent_node = self._map_store.find_node_by_uid(parent_uid)
                    node.parent = parent_node
                    parent_node.children[node.group].append(node)
            LOG.debug(f"build {group} nodes from chromadb: {nodes}")
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
            metadata = self._make_chroma_metadata(node)
            metadata["embedding"] = json.dumps(node.embedding)
            ids.append(node.uid)
            embeddings.append([0])  # we don't use chroma for retrieving
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

    def _delete_group_nodes(self, group_name: str, uids: List[str]) -> None:
        collection = self._collections.get(group_name)
        if collection:
            collection.delete(ids=uids)

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

            if node.embedding:
                # convert sparse embedding to List[float]
                new_embedding_dict = {}
                for key, embedding in node.embedding.items():
                    if isinstance(embedding, dict):
                        dim = self._embed_dim.get(key)
                        if not dim:
                            raise ValueError(f'dim of embed [{key}] is not determined.')
                        new_embedding = [0] * dim
                        for idx, val in embedding.items():
                            new_embedding[int(idx)] = val
                        new_embedding_dict[key] = new_embedding
                    else:
                        new_embedding_dict[key] = embedding
                node.embedding = new_embedding_dict

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
