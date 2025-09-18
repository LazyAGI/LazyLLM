from typing import List, Callable, Optional, Dict, Union, Tuple, Any
from .doc_node import DocNode
from .index_base import IndexBase
from lazyllm import LOG
from lazyllm.common import override
from .utils import parallel_do_embedding, generic_process_filters, is_sparse
from .similarity import registered_similarities

# ---------------------------------------------------------------------------- #

class DefaultIndex(IndexBase):
    """Default index implementation responsible for querying, updating, and removing document nodes in the underlying store based on embedding or text similarity.  
Supports multiple similarity metrics and performs embedding computation and node updates when required.  

Args:
    embed (Dict[str, Callable]): Mapping of embedding names to functions that generate vector representations from strings.
    store (StoreBase): Underlying storage to persist and retrieve `DocNode` objects.
    **kwargs: Reserved for future extension.

**Returns:**

- DefaultIndex: The default index instance.
"""
    def __init__(self, embed: Dict[str, Callable], store, **kwargs):
        self.embed = embed
        self.store = store

    @override
    def update(self, nodes: List[DocNode]) -> None:
        """Update the index with the given list of document nodes. This is a placeholder implementation and should be provided/extended in concrete usage.

Args:
    nodes (List[DocNode]): Document nodes to add or update in the index.
"""
        pass

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        """Remove nodes with specified UIDs from the index. Optionally scoped to a group. This is a no-op placeholder and should be implemented in concrete usage.

Args:
    uids (List[str]): List of unique IDs of nodes to remove.
    group_name (Optional[str]): Optional group name to scope the removal.
"""
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
        filters: Optional[Dict[str, List]] = None,
        **kwargs,
    ) -> List[DocNode]:
        """Perform a query against the index, supporting both embedding-based and text-based similarity modes. Filters and ranks nodes according to similarity functions and cutoffs.

Args:
    query (str): The raw query string.
    group_name (str): The group name from which to retrieve nodes.
    similarity_name (str): Name of the similarity metric to use; must be registered in registered_similarities.
    similarity_cut_off (Union[float, Dict[str, float]]): Similarity threshold(s) used to filter results; can be a single float or a mapping per embedding.
    topk (int): Maximum number of candidates to keep per similarity channel before final filtering.
    embed_keys (Optional[List[str]]): Specific embedding keys to use; defaults to all available if not provided.
    filters (Optional[Dict[str, List]]): Additional pre-filters applied to nodes before similarity computation.
    **kwargs: Extra keyword arguments forwarded to the similarity function.

**Returns**

- list: List[DocNode]: Deduplicated list of document nodes passing similarity and cutoff criteria.
"""
        if similarity_name not in registered_similarities:
            raise ValueError(
                f'{similarity_name} not registered, please check your input. '
                f'Available options now: {registered_similarities.keys()}'
            )
        similarity_func, mode, descend = registered_similarities[similarity_name]

        nodes = self.store.get_nodes(group=group_name)
        if filters:
            nodes = generic_process_filters(nodes, filters)

        if mode == 'embedding':
            assert self.embed, 'Chosen similarity needs embed model.'
            assert len(query) > 0, 'Query should not be empty.'
            if not embed_keys:
                embed_keys = list(self.embed.keys())
            query_embedding = {k: self.embed[k](query) for k in embed_keys}
            self._check_supported(similarity_name, query_embedding)
            modified_nodes = parallel_do_embedding(self.embed, embed_keys, nodes)
            self.store.update_nodes(modified_nodes)
            similarities = similarity_func(query_embedding, nodes, topk=topk, **kwargs)
        elif mode == 'text':
            similarities = similarity_func(query, nodes, topk=topk, **kwargs)
        else:
            raise NotImplementedError(f'Mode {mode} is not supported.')

        if not isinstance(similarities, dict):
            results = self._filter_nodes_by_score(similarities, topk, similarity_cut_off, descend)
        else:
            results = []
            for key in (embed_keys or similarities.keys()):
                sims = similarities[key]
                sim_cut_off = similarity_cut_off if isinstance(similarity_cut_off, float) else similarity_cut_off[key]
                results.extend(self._filter_nodes_by_score(sims, topk, sim_cut_off, descend))
        results = list(set(results))
        LOG.debug(f'Retrieving query `{query}` and get results: {results}')
        return results

    def _filter_nodes_by_score(self, similarities: List[Tuple[DocNode, float]], topk: int,
                               similarity_cut_off: float, descend) -> List[DocNode]:
        similarities.sort(key=lambda x: x[1], reverse=descend)
        if topk is not None:
            similarities = similarities[:topk]

        return [node.with_sim_score(score) for node, score in similarities if score > similarity_cut_off]

    def _check_supported(self, similarity_name: str, query_embedding: Dict[str, Any]) -> None:
        if similarity_name.lower() == 'cosine':
            for k, e in query_embedding.items():
                if is_sparse(e):
                    raise NotImplementedError(f'embed `{k}`, which is sparse, is not supported.')
