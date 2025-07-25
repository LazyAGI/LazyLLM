from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable, Set, Union

from .store_base import StoreBase, LAZY_ROOT_NAME
from .map_store import MapStore

from ..doc_node import DocNode
from ..index_base import IndexBase
from ..default_index import DefaultIndex
from ..utils import sparse2normal

from lazyllm import LOG
from lazyllm.common import override, obj2str, str2obj
from lazyllm.thirdparty import chromadb


class ChromadbStore(StoreBase):
    """
继承自 StoreBase 抽象基类。它主要用于存储和管理文档节点(DocNode)，支持节点增删改查、索引管理和持久化存储。

Args:
     group_embed_keys (Dict[str, Set[str]]): 指定每个文档分组所对应的嵌入字段。
    embed (Dict[str, Callable]): 嵌入生成函数或其映射，支持多嵌入源。
    embed_dims (Dict[str, int]): 每种嵌入类型对应的维度。
    dir (str): chromadb 数据库存储路径。
    kwargs (Dict): 其他可选参数，传递给父类或内部组件。


Examples:

    >>> from lazyllm.tools.rag.chroma_store import ChromadbStore
    >>> from typing import Dict, List
    >>> import numpy as np
    >>> store = ChromadbStore(
    ...     group_embed_keys={"articles": {"title_embed", "content_embed"}},
    ...     embed={
    ...         "title_embed": lambda x: np.random.rand(128).tolist(),
    ...         "content_embed": lambda x: np.random.rand(256).tolist()
    ...     },
    ...     embed_dims={"title_embed": 128, "content_embed": 256},
    ...     dir="./chroma_data"
    ... )
    >>> store.update_nodes([node1, node2])
    >>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
    >>> for node in results:
    ...     print(f"找到文档: {node._content[:20]}...")
    >>> store.remove_nodes(doc_ids=["doc1"])

    """
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
        }

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """
更新一组 DocNode 节点。
Args:
    nodes(DocNode): 需要更新的 DocNode 列表。
"""
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    @override
    def remove_nodes(self, doc_ids: List[str], group_name: Optional[str] = None,
                     uids: Optional[List[str]] = None) -> None:
        """
删除指定条件的节点。
Args:
    doc_ids(str): 按文档 ID 删除。
    group_name(str): 限定删除的组名。
    uids(str): 按节点唯一 ID 删除。
"""
        nodes = self._map_store.get_nodes(group_name=group_name, doc_ids=doc_ids, uids=uids)
        group2uids = defaultdict(list)
        for node in nodes:
            group2uids[node._group].append(node._uid)
        for group, uids in group2uids.items():
            self._delete_group_nodes(group, uids)
            self._map_store.remove_nodes(doc_ids=doc_ids, uids=uids)

    @override
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        """
更新文档的元数据。。
Args:
    doc_id(str):需要更新的文档 ID。
    metadata(dict):新的元数据（键值对）。
"""
        self._map_store.update_doc_meta(doc_id=doc_id, metadata=metadata)
        for group in self.activated_groups():
            nodes = self.get_nodes(group_name=group, doc_ids=[doc_id])
            self._save_nodes(nodes)

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,
                  doc_ids: Optional[Set] = None, **kwargs) -> List[DocNode]:
        """
根据条件查询节点。
Args:
    group_name(str]):节点所属的组名。
    uids(List[str]):节点唯一 ID 列表。
    doc_ids	(Set[str])：文档 ID 集合。
    **kwargs:其他扩展参数。
"""
        return self._map_store.get_nodes(group_name, uids, doc_ids, **kwargs)

    @override
    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        """
激活指定的组。
Args:
    group_names([str, List[str]])：按组名激活。

"""
        return self._map_store.activate_group(group_names)

    @override
    def activated_groups(self):
        """
激活组，返回当前激活的组名列表。

"""
        return self._map_store.activated_groups()

    @override
    def is_group_active(self, name: str) -> bool:
        """
检查指定组是否激活。
Args:
    name(str)：组名。
"""
        return self._map_store.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        """
返回所有组名列表。

"""
        return self._map_store.all_groups()

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        """
通过默认索引执行查询。
Args:
    args：查询参数。
    kwargs：其他扩展参数。
"""
        return self.get_index('default').query(*args, **kwargs)

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        """
注册自定义索引。
Args:
    type(str):索引类型名称。
    index(IndexBase):实现 IndexBase 的对象。
"""
        self._name2index[type] = index

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """
获取指定类型的索引。
Args:
    type(str):索引类型
"""
        if type is None:
            type = 'default'
        return self._name2index.get(type)

    @override
    def clear_cache(self, group_names: Optional[List[str]] = None):
        """
清除指定组或所有组的 ChromaDB 集合和内存缓存。
Args:
    group_names(List[str])：组名列表，为 None 时清除所有组。

"""
        if group_names is None:
            for group_name in self.activated_groups():
                self._db_client.delete_collection(name=group_name)
            self._collections.clear()
            self._map_store.clear_cache()
        elif isinstance(group_names, str):
            group_names = [group_names]
        elif isinstance(group_names, (tuple, list, set)):
            group_names = list(group_names)
        else:
            raise TypeError(f"Invalid type {type(group_names)} for group_names, expected list of str")
        for group_name in group_names:
            self._db_client.delete_collection(name=group_name)
        self._map_store.clear_cache(group_names)

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
