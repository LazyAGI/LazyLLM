from lazyllm.thirdparty import jieba
from lazyllm.thirdparty import pandas as pd

from typing import Callable, List, Optional, cast, Set

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.utils import globals_helper


def simple_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """Extract keywords with simple algorithm."""
    tokens = jieba.cut(text_chunk)
    if filter_stopwords:
        tokens = [t for t in tokens if t not in globals_helper.stopwords]
    value_counts = pd.Series(tokens).value_counts()
    keywords = value_counts.index.tolist()[:max_keywords]
    return set(keywords)

def tokenize_remove_stopwords(text: str) -> List[str]:
    text = text.lower()
    words = list(simple_extract_keywords(text))
    return words


class ChineseBM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Optional[Callable[[str], List[str]]],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")

        self._nodes = nodes
        self._tokenizer = tokenizer or tokenize_remove_stopwords
        self._similarity_top_k = similarity_top_k
        self._corpus = [self._tokenizer(node.get_content(MetadataMode.NONE)) for node in self._nodes]

        self.bm25 = BM25Okapi(self._corpus)
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
            cls,
            index: Optional[VectorStoreIndex] = None,
            nodes: Optional[List[BaseNode]] = None,
            docstore: Optional[BaseDocumentStore] = None,
            tokenizer: Optional[Callable[[str], List[str]]] = None,
            similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
            verbose: bool = False):
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        tokenizer = tokenizer or tokenize_remove_stopwords
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:
        tokenized_query = self._tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        nodes: List[NodeWithScore] = []
        for i, node in enumerate(self._nodes):
            nodes.append(NodeWithScore(node=node, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
