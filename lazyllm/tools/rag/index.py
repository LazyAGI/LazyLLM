import concurrent
import os
from typing import List, Callable, Optional, Dict
from .store import DocNode, BaseStore
import numpy as np
from .component.bm25 import BM25
from lazyllm import LOG, config, ThreadPoolExecutor

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    "max_embedding_workers",
    int,
    min(32, (os.cpu_count() or 1) + 4),
    "MAX_EMBEDDING_WORKERS",
)


class DefaultIndex:
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
                if batch:
                    return f(query, nodes, **kwargs)
                else:
                    return [(node, f(query, node, **kwargs)) for node in nodes]

            cls.registered_similarity[f.__name__] = (wrapper, mode, descend)
            return wrapper

        return decorator(func) if func else decorator

    def _parallel_do_embedding(self, nodes: List[DocNode]) -> List[DocNode]:
        with ThreadPoolExecutor(config["max_embedding_workers"]) as executor:
            futures = {
                executor.submit(node.do_embedding, self.embed): node
                for node in nodes
                if not node.has_embedding()
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return nodes

    def query(
        self,
        query: str,
        nodes: List[DocNode],
        similarity_name: str,
        similarity_cut_off: float,
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

        if mode == "embedding":
            assert self.embed, "Chosen similarity needs embed model."
            assert len(query) > 0, "Query should not be empty."
            query_embedding = {k: e(query) for k, e in self.embed.items()}
            nodes = self._parallel_do_embedding(nodes)
            self.store.try_save_nodes(nodes)
            similarities = similarity_func(query_embedding, nodes, topk=topk, **kwargs)
        elif mode == "text":
            similarities = similarity_func(query, nodes, topk=topk, **kwargs)
        else:
            raise NotImplementedError(f"Mode {mode} is not supported.")

        if not isinstance(similarities[0][1], dict):
            similarities.sort(key=lambda x: x[1], reverse=descend)
            if topk is not None:
                similarities = similarities[:topk]

            results = [node for node, score in similarities if score > similarity_cut_off]
            LOG.debug(f"Retrieving query `{query}` and get results: {results}")
        else:
            sort_keys = embed_keys or similarities[0][1].keys()
            results = []
            # nodeIds = []
            for key in sort_keys:
                sorted_sim = sorted(similarities, key=lambda item: item[1][key], reverse=descend)
                if topk is not None:
                    sorted_sim = sorted_sim[:topk]

                results.extend([node for node, score in sorted_sim if score[key] > similarity_cut_off])
        return list(set(results))


@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="en", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25_chinese(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="zh", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="embedding")
def cosine(query: Dict[str, List[float]], node: DocNode, **kwargs) -> Dict[str, float]:
    similarity = {}
    for k, q in query.items():
        product = np.dot(q, node.embedding.get(k))
        norm = np.linalg.norm(q) * np.linalg.norm(node.embedding.get(k))
        similarity[k] = product / norm
    return similarity


# User-defined similarity decorator
def register_similarity(
    func: Optional[Callable] = None,
    mode: str = "",
    descend: bool = True,
    batch: bool = False,
) -> Callable:
    return DefaultIndex.register_similarity(func, mode, descend, batch)
