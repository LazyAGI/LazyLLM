from typing import List, Tuple
from ..doc_node import DocNode
import Stemmer
from lazyllm.thirdparty import jieba, bm25s
from .stopwords import STOPWORDS_CHINESE


class BM25:
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes."""

    def __init__(
        self,
        nodes: List[DocNode],
        language: str = "en",
        topk: int = 2,
        **kwargs,
    ) -> None:
        if language == "en":
            self._stemmer = Stemmer.Stemmer("english")
            self._stopwords = language
            self._tokenizer = lambda t: t
        elif language == "zh":
            self._stemmer = None
            # TODO(ywt): after bm25s supports cn stopwards, update this
            self._stopwords = STOPWORDS_CHINESE
            self._tokenizer = lambda t: " ".join(jieba.lcut(t))
        self.topk = min(topk, len(nodes))
        self.nodes = nodes

        corpus_tokens = bm25s.tokenize(
            [self._tokenizer(node.get_text()) for node in nodes],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

    def retrieve(self, query: str) -> List[Tuple[DocNode, float]]:
        tokenized_query = bm25s.tokenize(
            self._tokenizer(query), stopwords=self._stopwords, stemmer=self._stemmer
        )
        indexs, scores = self.bm25.retrieve(tokenized_query, k=self.topk)
        results = []
        for idx, score in zip(indexs[0], scores[0]):
            results.append((self.nodes[idx], score))
        return results
