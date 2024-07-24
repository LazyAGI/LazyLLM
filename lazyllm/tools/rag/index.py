from typing import List, Callable
from .store import DocNode, BaseStore
import numpy as np
from .component.bm25 import BM25
from lazyllm import LOG


class DefaultIndex:
    """Default Index, registered for similarity functions"""

    registered_similarity = dict()

    def __init__(self, embed: Callable, store: BaseStore, **kwargs):
        self.embed = embed
        self.store = store

    @classmethod
    def register_similarity(cls, func=None, mode=None, descend=True, batch=False):
        def decorator(f):
            def wrapper(query, nodes, **kwargs):
                if batch:
                    return f(query, nodes, **kwargs)
                else:
                    return [(node, f(query, node, **kwargs)) for node in nodes]

            cls.registered_similarity[f.__name__] = (wrapper, mode, descend)
            return wrapper

        return decorator(func) if func else decorator

    def query(
        self,
        query: str,
        nodes: List[DocNode],
        similarity_name: str,
        topk: int,
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
            query_embedding = self.embed(query)
            for node in nodes:
                if not node.has_embedding():
                    node.do_embedding(self.embed)
            self.store.try_save_nodes(nodes[0].group, nodes)
            nodes = similarity_func(query_embedding, nodes, topk=topk, **kwargs)
        elif mode == "text":
            similarities = similarity_func(query, nodes, topk=topk, **kwargs)
        else:
            raise NotImplementedError(f"Mode {mode} is not supported.")

        similarities.sort(key=lambda x: x[1], reverse=descend)
        if topk is not None:
            similarities = similarities[:topk]
        return [node for node, _ in similarities]


@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="en", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="text", batch=True)
def bm25_chinese(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="zh", **kwargs)
    return bm25_retriever.retrieve(query)


@DefaultIndex.register_similarity(mode="embedding")
def cosine(query: List[float], node: DocNode, **kwargs):
    product = np.dot(query, node.embedding)
    norm = np.linalg.norm(query) * np.linalg.norm(node.embedding)
    return product / norm


# User-defined similarity decorator
def register_similarity(func=None, mode=None, descend=True, batch=False):
    return DefaultIndex.register_similarity(func, mode, descend, batch)
