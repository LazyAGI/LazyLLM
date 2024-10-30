from typing import Any, Dict, List, Optional
import chromadb
from lazyllm import LOG, config
from lazyllm.common import override
from chromadb.api.models.Collection import Collection
from .store_base import StoreBase
from .index_base import IndexBase
from .doc_node import DocNode
import json

# ---------------------------------------------------------------------------- #

LAZY_ROOT_NAME = "lazyllm_root"
EMBED_DEFAULT_KEY = '__default__'
config.add("rag_store_type", str, "map", "RAG_STORE_TYPE")  # "map", "chroma"
config.add("rag_persistent_path", str, "./lazyllm_chroma", "RAG_PERSISTENT_PATH")

# ---------------------------------------------------------------------------- #

def _update_indices(name2index: Dict[str, IndexBase], nodes: List[DocNode]) -> None:
    for _, index in name2index.items():
        index.update(nodes)

def _remove_from_indices(name2index: Dict[str, IndexBase], uids: List[str],
                         group_name: Optional[str] = None) -> None:
    for _, index in name2index.items():
        index.remove(uids, group_name)

class MapStore(StoreBase, IndexBase):
    def __init__(self, node_groups: List[str]):
        super().__init__()
        # Dict[group_name, Dict[uuid, DocNode]]
        self._group2docs: Dict[str, Dict[str, DocNode]] = {
            group: {} for group in node_groups
        }
        self._name2index = {}

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            self._group2docs[node.group][node.uid] = node
        _update_indices(self._name2index, nodes)

    @override
    def remove_nodes(self, group_name: str, uids: List[str] = None) -> None:
        if uids:
            docs = self._group2docs.get(group_name)
            if docs:
                _remove_from_indices(self._name2index, uids)
                for uid in uids:
                    docs.pop(uid, None)
        else:
            docs = self._group2docs.pop(group_name, None)
            if docs:
                _remove_from_indices(self._name2index, [doc.uid for doc in docs])

    @override
    def get_nodes(self, group_name: str, uids: List[str] = None) -> List[DocNode]:
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

    @override
    def is_group_active(self, name: str) -> bool:
        docs = self._group2docs.get(name)
        return True if docs else False

    @override
    def all_groups(self) -> List[str]:
        return self._group2docs.keys()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._name2index[type] = index

    @override
    def get_index(self, Optional[type]: str = None) -> Optional[IndexBase]:
        if type:
            return self._name2index.get(type)
        return self

    @override
    def update(nodes: List[DocNode]) -> None:
        self.update_nodes(nodes)

    @override
    def remove(uids: List[str], group_name: Optional[str] = None) -> None:
        if group_name:
            self.remove_nodes(group_name, uids)
        else:
            for _, docs in self._group2docs.items():
                for uid in uids:
                    docs.pop(uid, None)
        _remove_from_indices(self._name2index, uids)

    @override
    def query(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self.get_nodes(group_name, uids)

    def find_node_by_uid(self, uid: str) -> Optional[DocNode]:
        for docs in self._group2docs.values():
            doc = docs.get(uid)
            if doc:
                return doc
        return None

# ---------------------------------------------------------------------------- #

class ChromadbStore(StoreBase):
    def __init__(
        self, node_groups: List[str], embed_dim: Dict[str, int]
    ) -> None:
        super().__init__()
        self._map_store = MapStore(node_groups)
        self._db_client = chromadb.PersistentClient(path=config["rag_persistent_path"])
        LOG.success(f"Initialzed chromadb in path: {config['rag_persistent_path']}")
        self._collections: Dict[str, Collection] = {
            group: self._db_client.get_or_create_collection(group)
            for group in node_groups
        }
        self._embed_dim = embed_dim

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

# ---------------------------------------------------------------------------- #

class MilvusStore(StoreBase, IndexBase):
    def __init__(self, uri: str, embed: Dict[str, Callable],
                 group_fields: Dict[str, List[pymilvus.FieldSchema]],
                 group_indices: Dict[str, pymilvus.IndexParams]):
        self._primary_key = 'uid'
        self._embedding_keys = list(embed.keys())
        self._metadata_keys = filter(lambda x: x not in embed.keys(), group_fields.keys())

        self._embed = embed
        self._client = pymilvus.MilvusClient(uri=uri)

        id_field = pymilvus.FieldSchema(
            name=self._primary_key, dtype=pymilvus.DataType.VARCHAR,
            max_length=128, is_primary=True)

        for group_name, field_list in group_fields.items():
            if group_name in self._client.list_collections():
                continue

            schema = CollectionSchema(fields=id_field+field_list)
            index_params = group_indices.get(group_name)

            self._client.create_collection(collection_name=group_name, schema=schema,
                                           index_params=index_params)

        self._map_store = MapStore(list(group_fields.keys()))
        self._load_all_nodes_to(self._map_store)

    # ----- Store APIs ----- #

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        parallel_do_embedding(self._embed, nodes)
        for node in nodes:
            data = self._serialize_node_partial(node)
            self._client.upsert(collection_name=node.group, data=data)

        self._map_store.update_nodes(nodes)

    @override
    def remove_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> None:
        if uids:
            self._client.delete(collection_name=group_name,
                                filter=f'{self._primary_key} in {uids}')
        else:
            self._client.drop_collection(collection_name=group_name)

        self._map_store.remove_nodes(group_name, uids)

    @override
    def get_nodes(self, group_name: str, uids: Optional[List[str]] = None) -> List[DocNode]:
        return self._map_store.get_nodes(group_name, uids)

    @override
    def is_group_active(self, name: str) -> bool:
        return self._map_store.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        return _map_store.all_groups()

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        self._map_store.register_index(type, index)

    @override
    def get_index(self, Optional[type]: str = None) -> Optional[IndexBase]:
        if type:
            return self._map_store.get_index(type)
        return self

    # ----- Index APIs ----- #

    @override
    def update(self, nodes: List[DocNode]) -> None:
        self.update_nodes(nodes)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        self.remove_nodes(group_name, uids)

    @override
    def query(self,
              query: str,
              group_name: str,
              similarity_name: str,
              similarity_cut_off: Union[float, Dict[str, float]], # ignored
              topk: int,
              embed_keys: Optional[List[str]] = None,
              **kwargs) -> List[DocNode]:
        reqs = []
        for key in embed_keys:
            embed_func = self._embed.get(key)
            query_embedding = embed_func(query)
            # TODO set search params according to similarity_name
            req = AnnSearchRequest(
                data=query_embedding,
                anns_field=key,
                limit=topk,
            )
            reqs.append(req)

        results = self._client.hybrid_search(collection_name=group_name, reqs=reqs,
                                             ranker=ranker, limit=topk)
        if len(results) != 1:
            raise ValueError(f'return results size [{len(results)}] != 1')

        uidset = set()
        for record in results[0]:
            uidset.insert(record['id'])
        return self._map_store.get_nodes(group_name, list(uidset))

    def _load_all_nodes_to(self, store: StoreBase):
        results = self._client.query(collection_name=group_name,
                                     filter=f'{self._primary_key} != ""')
        for result in results:
            doc = self._deserialize_node_partial(result)
            store.update_nodes([doc], group)

        # construct DocNode::parent and DocNode::children
        for group in all_groups():
            for node in self.get_nodes(group):
                if node.parent:
                    parent_uid = node.parent
                    parent_node = self._map_store.find_node_by_uid(parent_uid)
                    node.parent = parent_node
                    parent_node.children[node.group].append(node)

    @staticmethod
    def _serialize_node_partial(node: DocNode) -> Dict:
        res = {
            'uid': node.uid,
            'text': node.text,
            'group': node.group,
        }

        if self.parent:
            res['parent'] = node.parent.uid

        for k, v in node.embedding.items():
            res['embedding_' + k] = v
        for k, v in self.metadata.items():
            res['metadata_' + k] = v

        return res

    @staticmethod
    def _deserialize_node_partial(result: Dict) -> DocNode:
        '''
        without parent and children
        '''
        doc = DocNode(
            uid=result.get('uid'),
            text=result.get('text'),
            group=result.get('group'),
            parent=result.get('parent'),  # this is the parent's uid
        )

        for k in self._embedding_keys:
            doc.embedding[k] = result.get('embedding_' + k)
        for k in self._metadata_keys:
            doc._metadata[k] = result.get('metadata_' + k)

        return doc
