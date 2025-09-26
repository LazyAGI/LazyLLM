from typing import Optional, Callable, Literal, List, Dict
from .component.bm25 import BM25
from lazyllm.thirdparty import numpy as np
from .doc_node import DocNode
import functools
import hashlib
import threading

registered_similarities = dict()
_bm25_cache: Dict[str, BM25] = {}
_bm25_cache_lock = threading.RLock()

def register_similarity(
    func: Optional[Callable] = None,
    mode: Optional[Literal['text', 'embedding']] = None,
    descend: bool = True,
    batch: bool = False,
) -> Callable:
    def decorator(f):
        @functools.wraps(f)
        def wrapper(query, nodes, **kwargs):
            if mode != 'embedding':
                if batch:
                    return f(query, nodes, **kwargs)
                else:
                    return [(node, f(query, node, **kwargs)) for node in nodes]
            else:
                assert isinstance(query, dict), 'query must be of dict type, used for similarity calculation.'
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

def _hash_nodes(nodes: List[DocNode], language: str) -> str:
    m = hashlib.sha256()
    m.update(language.encode('utf-8'))
    # keep the order of nodes
    for node in nodes:
        uid = node.uid
        m.update(uid.encode('utf-8'))
        # to detect content changes, add the short hash of the content
        text_hash = hashlib.sha256(node.text.encode('utf-8')).digest()
        m.update(text_hash)
    return m.hexdigest()

def _get_bm25_from_cache(nodes: List[DocNode], language: str = 'en', **kwargs) -> BM25:
    key = _hash_nodes(nodes, language)
    with _bm25_cache_lock:
        bm = _bm25_cache.get(key)
        if bm is None:
            bm = BM25(nodes, language=language, **kwargs)
            _bm25_cache[key] = bm
    return bm

def clear_bm25_cache() -> None:
    with _bm25_cache_lock:
        _bm25_cache.clear()

@register_similarity(mode='text', batch=True)
def bm25(query: str, nodes: List[DocNode], **kwargs) -> List:
    return _get_bm25_from_cache(nodes, language='en', **kwargs).retrieve(query)


@register_similarity(mode='text', batch=True)
def bm25_chinese(query: str, nodes: List[DocNode], **kwargs) -> List:
    return _get_bm25_from_cache(nodes, language='zh', **kwargs).retrieve(query)


@register_similarity(mode='embedding')
def cosine(query: List[float], node: List[float], **kwargs) -> float:
    product = np.dot(query, node)
    norm = np.linalg.norm(query) * np.linalg.norm(node)
    return product / norm
