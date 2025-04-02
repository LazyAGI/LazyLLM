from typing import Optional, Callable, Literal, List
from .component.bm25 import BM25
from lazyllm.thirdparty import numpy as np
from .doc_node import DocNode
import functools

registered_similarities = dict()

def register_similarity(
    func: Optional[Callable] = None,
    mode: Optional[Literal['text', 'embedding']] = None,
    descend: bool = True,
    batch: bool = False,
) -> Callable:
    def decorator(f):
        @functools.wraps(f)
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
                        similarity[key] = [(node, sim) for node, sim in zip(nodes, f(val, nodes_embed, **kwargs))]
                else:
                    for key, val in query.items():
                        similarity[key] = [(node, f(val, node.embedding[key], **kwargs)) for node in nodes]
                return similarity
        registered_similarities[f.__name__] = (wrapper, mode, descend)
        return wrapper

    return decorator(func) if func else decorator

@register_similarity(mode="text", batch=True)
def bm25(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="en", **kwargs)
    return bm25_retriever.retrieve(query)


@register_similarity(mode="text", batch=True)
def bm25_chinese(query: str, nodes: List[DocNode], **kwargs) -> List:
    bm25_retriever = BM25(nodes, language="zh", **kwargs)
    return bm25_retriever.retrieve(query)


@register_similarity(mode="embedding")
def cosine(query: List[float], node: List[float], **kwargs) -> float:
    product = np.dot(query, node)
    norm = np.linalg.norm(query) * np.linalg.norm(node)
    return product / norm
