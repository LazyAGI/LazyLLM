import numpy as np


class DefaultIndex:
    """ Default Index, registered for similarity functions, does nothing else"""
    registered_similarity = dict()

    def __init__(self, **kwargs):
        pass

    @classmethod
    def register_similarity(cls, func=None, use_embedding=False):
        def decorator(f):
            cls.registered_similarity[f.__name__] = (f, use_embedding)
            return f

        return decorator(func) if func else decorator


@DefaultIndex.register_similarity
def dummy_similarity(query, nodes, **kwargs):
    """dummy similarity that return the topk nodes in alphabet order"""
    return sorted(nodes, key=lambda node: len(node.text), reverse=True)


@DefaultIndex.register_similarity(use_embedding=True)
def cosine(query_embedding, nodes, topk, **kwargs):
    def cosine_similarity(embedding1, embedding2):
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm

    nodes_with_similarity = [
        (node, cosine_similarity(query_embedding, node.embedding)) for node in nodes
    ]
    nodes_with_similarity.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in nodes_with_similarity[:topk]]
