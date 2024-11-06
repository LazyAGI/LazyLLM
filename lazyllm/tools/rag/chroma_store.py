from typing import Any, Dict, List, Optional, Callable, Set
import chromadb
from lazyllm import LOG
from lazyllm.common import override
from chromadb.api.models.Collection import Collection
from .store_base import StoreBase, LAZY_ROOT_NAME
from .doc_node import DocNode
from .index_base import IndexBase
from .utils import _FileNodeIndex
from .default_index import DefaultIndex
from .map_store import MapStore
import pickle
import base64

# ---------------------------------------------------------------------------- #

class ChromadbStore(StoreBase):
    def __init__(self, group_embed_keys: Dict[str, Set[str]], embed: Dict[str, Callable],
                 embed_dims: Dict[str, int], dir: str, **kwargs) -> None:
        self._db_client = chromadb.PersistentClient(path=dir)
        LOG.success(f"Initialzed chromadb in path: {dir}")
        node_groups = list(group_embed_keys.keys())
        self._collections: Dict[str, Collection] = {
            group: self._db_client.get_or_create_collection(group)
            for group in node_groups
        }

        self._map_store = MapStore(node_groups=node_groups, embed=embed)
        self._load_store(embed_dims)

        self._name2index = {
            'default': DefaultIndex(embed, self._map_store),
            'file_node_map': _FileNodeIndex(),
        }

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            self._delete_group_nodes(group_name, uids)
        else:
            self._db_client.delete_collection(name=group_name)
        return self._map_store.remove_nodes(group_name, uids)

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
    def query(self, *args, **kwargs) -> List[DocNode]:
        return self.get_index('default').query(*args, **kwargs)

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        if type is None:
            type = 'default'
        return self._name2index.get(type)

    def _load_store(self, embed_dims: Dict[str, int]) -> None:
        if not self._collections[LAZY_ROOT_NAME].peek(1)["ids"]:
            LOG.info("No persistent data found, skip the rebuilding phrase.")
            return

        # Restore all nodes
        for group in self._collections.keys():
            results = self._peek_all_documents(group)
            nodes = self._build_nodes_from_chroma(results, embed_dims)
            self._map_store.update_nodes(nodes)

        # Rebuild relationships
        for group_name in self._map_store.all_groups():
            nodes = self._map_store.get_nodes(group_name)
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

    def _build_nodes_from_chroma(self, results: Dict[str, List], embed_dims: Dict[str, int]) -> List[DocNode]:
        nodes: List[DocNode] = []
        for i, uid in enumerate(results['ids']):
            chroma_metadata = results['metadatas'][i]

            parent = chroma_metadata['parent']
            fields = pickle.loads(base64.b64decode(chroma_metadata['fields'].encode('utf-8')))\
                if parent else None

            node = DocNode(
                uid=uid,
                text=results["documents"][i],
                group=chroma_metadata["group"],
                embedding=pickle.loads(base64.b64decode(chroma_metadata['embedding'].encode('utf-8'))),
                parent=parent,
                fields=fields,
            )

            if node.embedding:
                # convert sparse embedding to List[float]
                new_embedding_dict = {}
                for key, embedding in node.embedding.items():
                    if isinstance(embedding, dict):
                        dim = embed_dims.get(key)
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
            "embedding": base64.b64encode(pickle.dumps(node.embedding)).decode('utf-8'),
        }

        if node.parent:
            metadata["fields"] = base64.b64encode(pickle.dumps(node.fields)).decode('utf-8')

        return metadata

    def _peek_all_documents(self, group: str) -> Dict[str, List]:
        assert group in self._collections, f"group {group} not found."
        collection = self._collections[group]
        return collection.peek(collection.count())
