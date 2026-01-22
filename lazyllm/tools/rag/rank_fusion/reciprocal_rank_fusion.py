from typing import Sequence, List, Union
from lazyllm import ModuleBase
from ..doc_node import DocNode


class RRFFusion(ModuleBase):
    def __init__(self, top_k: int = 1) -> None:
        super().__init__()
        self.top_k = top_k

    def _is_nested_docnode_sequence(self, obj):
        try:
            return isinstance(obj, Sequence) and isinstance(obj[0], Sequence) and isinstance(obj[0][0], DocNode)
        except (IndexError, TypeError):
            return False

    def _normalize_mixed_input(self, *args: Union[List[DocNode], List[List[DocNode]]]) -> List[List[DocNode]]:
        normalized_lists = []
        for arg in args:
            if not arg:
                continue
            try:
                if self._is_nested_docnode_sequence(arg):
                    normalized_lists.extend(arg)
                elif isinstance(arg, Sequence) and len(arg) > 0 and isinstance(arg[0], DocNode):
                    normalized_lists.append(list(arg))
                else:
                    continue
            except (IndexError, TypeError):
                continue
        return normalized_lists

    def __call__(self, *args: Union[List[DocNode], List[List[DocNode]]]) -> list[DocNode]:
        # Reciprocal Rank Fusion on multiple rank lists. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively

        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            single_arg = args[0]
            if self._is_nested_docnode_sequence(single_arg):
                doc_nodes_lists = single_arg
            else:
                doc_nodes_lists = self._normalize_mixed_input(single_arg)
        else:
            doc_nodes_lists = self._normalize_mixed_input(*args)

        if not doc_nodes_lists: return []
        elif not self._is_nested_docnode_sequence(doc_nodes_lists):
            return doc_nodes_lists

        K = 60
        fused_scores = {}
        text_to_node = {}
        for nodes in doc_nodes_lists:
            for rank, node in enumerate(sorted(nodes, key=lambda x: x.similarity_score or 0.0, reverse=True), start=1):
                text = node.text
                # same nodes may have same text, but it's ok
                text_to_node[text] = node

                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + K)

        reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))
        reranked_nodes = []
        for text, score in reranked_results.items():
            node = text_to_node[text]
            node.score = score
            reranked_nodes.append(node)
        return reranked_nodes[:self.top_k] if self.top_k > 0 else reranked_nodes
