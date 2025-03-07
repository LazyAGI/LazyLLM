from typing import Any, Dict, List, Optional, Callable, Set
from lazyllm.thirdparty import chromadb
from lazyllm import LOG
from lazyllm.common import override, obj2str, str2obj
from .store_base import StoreBase, LAZY_ROOT_NAME
from .doc_node import DocNode
from .index_base import IndexBase
from .utils import _FileNodeIndex, sparse2normal
from .default_index import DefaultIndex
from .map_store import MapStore

# ---------------------------------------------------------------------------- #

class ChromadbStore(StoreBase):
    def __init__(self, group_embed_keys: Dict[str, Set[str]], embed: Dict[str, Callable],
                 embed_dims: Dict[str, int], dir: str, **kwargs) -> None:
        self._db_client = chromadb.PersistentClient(path=dir)
        LOG.success(f"Initialzed chromadb in path: {dir}")
        node_groups = list(group_embed_keys.keys())
        self._collections: Dict[str, chromadb.api.models.Collection.Collection] = {
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
    def update_doc_meta(self, filepath: str, metadata: dict) -> None:
        self._map_store.update_doc_meta(filepath, metadata)

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
        uid2node = {}
        for group in self._collections.keys():
            results = self._peek_all_documents(group)
            nodes = self._build_nodes_from_chroma(results, embed_dims)
            for node in nodes:
                uid2node[node._uid] = node

        # Rebuild relationships
        for node in uid2node.values():
            if node.parent:
                parent_uid = node.parent
                parent_node = uid2node.get(parent_uid)
                node.parent = parent_node
                parent_node.children[node._group].append(node)
        LOG.debug(f"build {group} nodes from chromadb: {nodes}")

        self._map_store.update_nodes(list(uid2node.values()))
        LOG.success("Successfully Built nodes from chromadb.")

    def _save_nodes(self, nodes: List[DocNode]) -> None:
        if not nodes:
            return
        # Note: It's caller's duty to make sure this batch of nodes has the same group.
        group = nodes[0]._group
        ids, embeddings, metadatas, documents = [], [], [], []
        collection = self._collections.get(group)
        assert (
            collection
        ), f"Group {group} is not found in collections {self._collections}"
        for node in nodes:
            metadata = self._make_chroma_metadata(node)
            ids.append(node._uid)
            embeddings.append([0])  # we don't use chroma for retrieving
            metadatas.append(metadata)
            documents.append(obj2str(node._content))
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
            local_metadata = str2obj(chroma_metadata['metadata'])
            global_metadata = str2obj(chroma_metadata['global_metadata']) if not parent else None

            node = DocNode(
                uid=uid,
                content=str2obj(results["documents"][i]),
                group=chroma_metadata["group"],
                embedding=str2obj(chroma_metadata['embedding']),
                parent=parent,
                metadata=local_metadata,
                global_metadata=global_metadata,
            )

            if node.embedding:
                # convert sparse embedding to List[float]
                new_embedding_dict = {}
                for key, embedding in node.embedding.items():
                    if isinstance(embedding, dict):
                        dim = embed_dims.get(key)
                        if not dim:
                            raise ValueError(f'dim of embed [{key}] is not determined.')
                        new_embedding_dict[key] = sparse2normal(embedding, dim)
                    else:
                        new_embedding_dict[key] = embedding
                node.embedding = new_embedding_dict

            nodes.append(node)
        return nodes

    def _make_chroma_metadata(self, node: DocNode) -> Dict[str, Any]:
        metadata = {
            "group": node._group,
            "parent": node.parent._uid if node.parent else "",
            "embedding": obj2str(node.embedding),
            "metadata": obj2str(node._metadata),
        }

        if node.is_root_node:
            metadata["global_metadata"] = obj2str(node.global_metadata)

        return metadata

    def _peek_all_documents(self, group: str) -> Dict[str, List]:
        assert group in self._collections, f"group {group} not found."
        collection = self._collections[group]
        return collection.peek(collection.count())
