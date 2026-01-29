"""
Embedding Hard Negative Miner Operator

This operator mines hard negative samples for embedding model training.
该算子挖掘困难负样本，用于提升 Embedding 模型的训练效果。
"""
import json
import random
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from lazyllm import LOG
from ...base_data import data_register

funcs = data_register.new_group('function')
classes = data_register.new_group('class')
class EmbeddingHardNegativeMiner(classes):
    """
    Mine hard negative samples for embedding training.
    为 Embedding 训练挖掘困难负样本。

    Hard negatives are passages that are semantically similar to the query
    but are not the correct answer. They help the model learn finer distinctions.

    Mining strategies:
    - random: Random sampling from corpus (baseline)
    - bm25: BM25-based lexical similarity
    - semantic: Embedding-based semantic similarity

    Args:
        mining_strategy: Strategy for mining negatives ("random", "bm25", "semantic")
        num_negatives: Number of negative samples per query (default: 7)
        embedding_serving: Embedding service for semantic mining (optional)
        seed: Random seed for reproducibility
    """

    def __init__(
            self,
            mining_strategy: str = "random",
            num_negatives: int = 7,
            embedding_serving=None,
            seed: int = 42,
    ):
        self.mining_strategy = mining_strategy
        self.num_negatives = num_negatives
        self.embedding_serving = embedding_serving
        self.seed = seed
        self._bm25 = None
        self._corpus_embeddings = None
        LOG.info(f"Initializing {self.__class__.__name__} with strategy: {mining_strategy}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingHardNegativeMiner 算子用于挖掘 Embedding 训练的困难负样本。\n\n"
                "核心功能：\n"
                "- 支持多种负样本挖掘策略（随机、BM25、语义相似度）\n"
                "- 自动过滤正样本，避免将正确答案作为负样本\n"
                "- 可配置负样本数量\n\n"
                "输入参数：\n"
                "- input_query_key: 查询字段名（默认：'query'）\n"
                "- input_pos_key: 正样本字段名（默认：'pos'）\n"
                "- corpus_key: 语料库字段名（默认：'passage'）\n"
                "- output_neg_key: 输出负样本字段名（默认：'neg'）\n\n"
                "输出：包含困难负样本的数据列表"
            )
        else:
            return (
                "EmbeddingHardNegativeMiner mines hard negative samples for embedding training.\n\n"
                "Features:\n"
                "- Multiple mining strategies (random, BM25, semantic similarity)\n"
                "- Automatic filtering of positive samples\n"
                "- Configurable number of negatives\n\n"
                "Input:\n"
                "- input_query_key: Query field name (default: 'query')\n"
                "- input_pos_key: Positive sample field name (default: 'pos')\n"
                "- corpus_key: Corpus field name (default: 'passage')\n"
                "- output_neg_key: Output negative field name (default: 'neg')"
            )

    def _init_bm25(self, corpus: List[str]):
        """Initialize BM25 index for lexical similarity mining."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)
            LOG.info("BM25 index initialized successfully.")
        except ImportError:
            LOG.error("rank_bm25 not installed. Please run: pip install rank-bm25")
            raise

    def _compute_corpus_embeddings(self, corpus: List[str]):
        """Compute embeddings for the entire corpus."""
        if self.embedding_serving is None:
            raise ValueError("Embedding serving is required for semantic mining strategy")

        LOG.info(f"Computing embeddings for {len(corpus)} documents...")
        self._corpus_embeddings = self.embedding_serving.generate_embedding_from_input(corpus)
        self._corpus_embeddings = np.array(self._corpus_embeddings)
        LOG.info("Corpus embeddings computed successfully.")

    def _mine_random(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using random sampling."""
        random.seed(self.seed)
        candidates = [doc for doc in corpus if doc not in pos_set]
        if len(candidates) <= self.num_negatives:
            return candidates
        return random.sample(candidates, self.num_negatives)

    def _mine_bm25(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using BM25 similarity."""
        if self._bm25 is None:
            self._init_bm25(corpus)

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, filter out positives
        scored_docs = [(score, doc) for score, doc in zip(scores, corpus) if doc not in pos_set]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top-k as hard negatives
        return [doc for _, doc in scored_docs[:self.num_negatives]]

    def _mine_semantic(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using semantic similarity."""
        if self._corpus_embeddings is None:
            self._compute_corpus_embeddings(corpus)

        # Get query embedding
        query_embedding = self.embedding_serving.generate_embedding_from_input([query])[0]
        query_embedding = np.array(query_embedding)

        # Compute cosine similarity
        similarities = np.dot(self._corpus_embeddings, query_embedding) / (
            np.linalg.norm(self._corpus_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        # Sort by similarity descending, filter out positives
        scored_docs = [(sim, doc) for sim, doc in zip(similarities, corpus) if doc not in pos_set]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top-k as hard negatives (high similarity but not positive)
        return [doc for _, doc in scored_docs[:self.num_negatives]]

    def __call__(
            self,
            data,
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            corpus_key: str = "passage",
            output_neg_key: str = "neg",
            corpus: Optional[List[str]] = None,
    ):
        """
        Mine hard negative samples for each query.

        Args:
            data: List of dict or pandas DataFrame with query-pos pairs
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field (list of passages)
            corpus_key: Key for corpus passages (used if corpus not provided)
            output_neg_key: Key for output negative samples field
            corpus: Optional external corpus for mining negatives

        Returns:
            List of dict with mined hard negatives added
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        # Build corpus from data if not provided
        if corpus is None:
            if corpus_key in dataframe.columns:
                corpus = dataframe[corpus_key].tolist()
            else:
                # Extract unique passages from all positive samples
                all_passages = []
                for pos_list in dataframe[input_pos_key].tolist():
                    if isinstance(pos_list, list):
                        all_passages.extend(pos_list)
                    else:
                        all_passages.append(pos_list)
                corpus = list(set(all_passages))

        LOG.info(f"Mining hard negatives for {len(dataframe)} queries from corpus of {len(corpus)} documents...")

        # Select mining function
        if self.mining_strategy == "random":
            mine_func = self._mine_random
        elif self.mining_strategy == "bm25":
            mine_func = self._mine_bm25
        elif self.mining_strategy == "semantic":
            mine_func = self._mine_semantic
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

        # Mine negatives for each query
        results = []
        for _, row in dataframe.iterrows():
            query = row[input_query_key]
            pos_samples = row[input_pos_key]

            # Convert to set for fast lookup
            if isinstance(pos_samples, list):
                pos_set = set(pos_samples)
            else:
                pos_set = {pos_samples}

            # Mine hard negatives
            negatives = mine_func(query, pos_set, corpus)

            new_row = row.to_dict()
            new_row[output_neg_key] = negatives
            results.append(new_row)

        LOG.info(f"Hard negative mining completed for {len(results)} samples.")
        return results

