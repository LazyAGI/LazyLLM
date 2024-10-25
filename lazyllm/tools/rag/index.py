import concurrent
import os
from typing import List, Callable, Optional, Dict, Union, Tuple
from .doc_node import DocNode
from .base_store import BaseStore
from .base_index import BaseIndex
import numpy as np
from .component.bm25 import BM25
from lazyllm import LOG, config, ThreadPoolExecutor
import pymilvus

# ---------------------------------------------------------------------------- #

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    "max_embedding_workers",
    int,
    min(32, (os.cpu_count() or 1) + 4),
    "MAX_EMBEDDING_WORKERS",
)

# ---------------------------------------------------------------------------- #

def parallel_do_embedding(embed: Dict[str, Callable], nodes: List[DocNode]) -> List[DocNode]:
    '''
    returns a list of modified nodes
    '''
    modified_nodes = []
    with ThreadPoolExecutor(config["max_embedding_workers"]) as executor:
        futures = []
        for node in nodes:
            miss_keys = node.has_missing_embedding(embed.keys())
            if not miss_keys:
                continue
            modified_nodes.append(node)
            for k in miss_keys:
                with node._lock:
                    if node.has_missing_embedding(k):
                        future = executor.submit(node.do_embedding, {k: embed[k]}) \
                            if k not in node._embedding_state else executor.submit(node.check_embedding_state, k)
                        node._embedding_state.add(k)
                        futures.append(future)
        if len(futures) > 0:
            for future in concurrent.futures.as_completed(futures):
                future.result()
    return modified_nodes

class DefaultIndex(BaseIndex):
    """Default Index, registered for similarity functions"""

    registered_similarity = dict()

    def __init__(self, embed: Dict[str, Callable], store: BaseStore, **kwargs):
        self.embed = embed
        self.store = store

    @classmethod
    def register_similarity(
        cls: "DefaultIndex",
        func: Optional[Callable] = None,
        mode: str = "",
        descend: bool = True,
        batch: bool = False,
    ) -> Callable:
        def decorator(f):
            def wrapper(query, nodes, **kwargs):
                if mode != "embedding":
                    if batch:
                        return f(query, nodes, **kwargs)
                    else:
                        return [(node, f(query, node, **kwargs)) for node in nodes]
                else:
                    assert isinstance(query, dict), "query must be of dict type, used for similarity calculation."
                    similarity = {}
                    if batch:
                        for key, val in query.items():
                            nodes_embed = [node.embedding[key] for node in nodes]
                            similarity[key] = f(val, nodes_embed, **kwargs)
                    else:
                        for key, val in query.items():
                            similarity[key] = [(node, f(val, node.embedding[key], **kwargs)) for node in nodes]
                    return similarity
            cls.registered_similarity[f.__name__] = (wrapper, mode, descend)
            return wrapper

        return decorator(func) if func else decorator

    # override
    def update(self, nodes: List[DocNode]) -> None:
        pass

    # override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        pass

    # override
    def query(
        self,
        query: str,
        group_name: str,
        similarity_name: str,
        similarity_cut_off: Union[float, Dict[str, float]],
        topk: int,
        embed_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> List[DocNode]:
        if similarity_name not in self.registered_similarity:
            raise ValueError(
                f"{similarity_name} not registered, please check your input."
                f"Available options now: {self.registered_similarity.keys()}"
            )
        similarity_func, mode, descend = self.registered_similarity[similarity_name]

        nodes = self.store.get_group_nodes(group_name)
        if mode == "embedding":
            assert self.embed, "Chosen similarity needs embed model."
            assert len(query) > 0, "Query should not be empty."
            query_embedding = {k: self.embed[k](query) for k in (embed_keys or self.embed.keys())}
            modified_nodes = parallel_do_embedding(self.embed, nodes)
            self.store.update_nodes(modified_nodes)
            similarities = similarity_func(query_embedding, nodes, topk=topk, **kwargs)
        elif mode == "text":
            similarities = similarity_func(query, nodes, topk=topk, **kwargs)
        else:
            raise NotImplementedError(f"Mode {mode} is not supported.")

        if not isinstance(similarities, dict):
            results = self._filter_nodes_by_score(similarities, topk, similarity_cut_off, descend)
        else:
            results = []
            for key in (embed_keys or similarities.keys()):
                sims = similarities[key]
                sim_cut_off = similarity_cut_off if isinstance(similarity_cut_off, float) else similarity_cut_off[key]
                results.extend(self._filter_nodes_by_score(sims, topk, sim_cut_off, descend))
        results = list(set(results))
        LOG.debug(f"Retrieving query `{query}` and get results: {results}")
        return results

    def _filter_nodes_by_score(self, similarities: List[Tuple[DocNode, float]], topk: int,
                               similarity_cut_off: float, descend) -> List[DocNode]:
        similarities.sort(key=lambda x: x[1], reverse=descend)
        if topk is not None:
            similarities = similarities[:topk]

        return [node for node, score in similarities if score > similarity_cut_off]

@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="en", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25_chinese(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="zh", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="embedding")
def cosine(query: List[float], node: List[float], **kwargs) -> float:
    product = np.dot(query, node)
    norm = np.linalg.norm(query) * np.linalg.norm(node)
    return product / norm


# User-defined similarity decorator
def register_similarity(
    func: Optional[Callable] = None,
    mode: str = "",
    descend: bool = True,
    batch: bool = False,
) -> Callable:
    return DefaultIndex.register_similarity(func, mode, descend, batch)

# ---------------------------------------------------------------------------- #

class MilvusEmbeddingField:
    def __init__(self, name: str, dim: int, data_type: int, index_type: str,
                 metric_type: str, index_params={}):
        self.name = name
        self.dim = dim
        self.data_type = data_type
        self.index_type = index_type
        self.metric_type = metric_type
        self.index_params = index_params

class MilvusIndex(BaseIndex):
    def __init__(self,
                 embed: Dict[str, Callable],
                 group_embedding_fields: Dict[str, List[MilvusEmbeddingField]],
                 uri: str, full_data_store: BaseStore):
        self._embed = embed
        self._full_data_store = full_data_store

        self._primary_key = 'uid'
        self._client = pymilvus.MilvusClient(uri=uri)

        for group_name, embedding_fields in group_embedding_fields.items():
            if group_name in self._client.list_collections():
                continue

            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                field_name=self._primary_key,
                datatype=pymilvus.DataType.VARCHAR,
                max_length=128,
                is_primary=True,
            )
            for field in embedding_fields:
                schema.add_field(
                    field_name=field.name,
                    datatype=field.data_type,
                    dim=field.dim)

            index_params = self._client.prepare_index_params()
            for field in embedding_fields:
                index_params.add_index(field_name=field.name, index_type=field.index_type,
                                       metric_type=field.metric_type, params=field.index_params)

            self._client.create_collection(collection_name=group_name, schema=schema,
                                           index_params=index_params)

    # override
    def update(self, nodes: List[DocNode]) -> None:
        parallel_do_embedding(self._embed, nodes)
        for node in nodes:
            data = node.embedding.copy()
            data[self._primary_key] = node.uid
            self._client.upsert(collection_name=node.group, data=data)

    # override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        if group_name:
            self._client.delete(collection_name=group_name,
                                filter=f'{self._primary_key} in {uids}')
        else:
            for group_name in self._client.list_collections():
                self._client.delete(collection_name=group_name,
                                    filter=f'{self._primary_key} in {uids}')

    # override
    def query(self,
              query: str,
              group_name: str,
              embed_keys: Optional[List[str]] = None,
              topk: int = 10,
              **kwargs) -> List[DocNode]:
        uids = set()
        for embed_name in embed_keys:
            embed_func = self._embed.get(embed_name)
            query_embedding = embed_func(query)
            results = self._client.search(collection_name=group_name, data=[query_embedding],
                                          limit=topk, anns_field=embed_name)
            if len(results) > 0:
                # we have only one `data` for search() so there is only one result in `results`
                for result in results[0]:
                    uids.update(result['id'])

        return self._full_data_store.get_group_nodes(group_name, list(uids))
