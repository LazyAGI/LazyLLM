from typing import List, Callable, Optional, Dict, Union, Tuple
from .doc_node import DocNode
from .store_base import StoreBase
from .index_base import IndexBase
import numpy as np
from .component.bm25 import BM25
from lazyllm import LOG
from lazyllm.common import override
from .embed_utils import parallel_do_embedding

# ---------------------------------------------------------------------------- #

class DefaultIndex(IndexBase):
    """Default Index, registered for similarity functions"""

    registered_similarity = dict()

    def __init__(self, embed: Dict[str, Callable], store: StoreBase, **kwargs):
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

    @override
    def update(self, nodes: List[DocNode]) -> None:
        pass

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        pass

    @override
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

        nodes = self.store.get_nodes(group_name)
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