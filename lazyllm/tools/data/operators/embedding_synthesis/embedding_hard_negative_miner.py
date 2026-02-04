"""
Embedding Hard Negative Miner Operator

This operator mines hard negative samples for embedding model training.
该算子挖掘困难负样本，用于提升 Embedding 模型的训练效果。

原始算法参考：
1. lazyllm/tools/rag/component/bm25.py (BM25 实现)
2. lazyllm/tools/rag/similarity.py (相似度计算，包含 cosine 函数)
3. lazyllm/thirdparty (bm25s, jieba, numpy 等)

核心思想：通过不同策略（随机、BM25词汇相似度、语义向量相似度）挖掘困难负样本，
         提升 Embedding 模型对相似但不相关文本的区分能力。
"""
import random
from typing import List, Optional
from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.thirdparty import numpy as np, bm25s, jieba, Stemmer
from lazyllm.tools.rag.component.stopwords import STOPWORDS_CHINESE
from ...base_data import data_register

# 复用已存在的 embedding 组
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


class EmbeddingHardNegativeMiner(embedding):
    """
    Mine hard negative samples for embedding training.
    为 Embedding 训练挖掘困难负样本。

    Hard negatives are passages that are semantically similar to the query
    but are not the correct answer. They help the model learn finer distinctions.

    原始算法：使用 LazyLLM 内置模块
    - random: 简单随机采样（基线方法）
    - bm25: 基于 bm25s 的词汇相似度挖掘（参考 lazyllm/tools/rag/component/bm25.py）
    - semantic: 基于向量的余弦相似度挖掘（参考 lazyllm/tools/rag/similarity.py）

    Mining strategies:
    - random: Random sampling from corpus (baseline)
    - bm25: BM25-based lexical similarity (using lazyllm.thirdparty.bm25s)
    - semantic: Embedding-based semantic similarity (using lazyllm.thirdparty.numpy)

    Args:
        mining_strategy: Strategy for mining negatives ("random", "bm25", "semantic")
        num_negatives: Number of negative samples per query (default: 7)
        embedding_serving: Embedding service for semantic mining (optional)
        language: Language for BM25 tokenization ("en" or "zh", default: "zh")
        seed: Random seed for reproducibility
        _concurrency_mode: Concurrency mode ('process', 'thread', 'single')
        _save_data: Whether to save intermediate data
    """

    def __init__(
            self,
            mining_strategy: str = "random",
            num_negatives: int = 7,
            embedding_serving=None,
            language: str = "zh",
            seed: int = 42,
            _concurrency_mode: str = 'single',
            _save_data: bool = True,
            **kwargs
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, _save_data=_save_data, **kwargs)
        self.mining_strategy = mining_strategy
        self.num_negatives = num_negatives
        self.embedding_serving = embedding_serving
        self.language = language
        self.seed = seed
        self._bm25 = None
        self._corpus = None
        self._corpus_embeddings = None
        # 初始化分词器和停用词（参考 lazyllm/tools/rag/component/bm25.py）
        if language == 'en':
            self._stemmer = Stemmer.Stemmer('english')
            self._stopwords = language
            self._tokenizer = lambda t: t
        elif language == 'zh':
            self._stemmer = None
            self._stopwords = STOPWORDS_CHINESE
            self._tokenizer = lambda t: ' '.join(jieba.lcut(t))
        else:
            self._stemmer = None
            self._stopwords = None
            self._tokenizer = lambda t: t
        LOG.info(f"Initializing {self.__class__.__name__} with strategy: {mining_strategy}, language: {language}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingHardNegativeMiner 算子用于挖掘 Embedding 训练的困难负样本。\n\n"
                "原始算法（使用 LazyLLM 内置模块）：\n"
                "- random: 随机采样（基线）\n"
                "- bm25: BM25 词汇相似度（lazyllm.thirdparty.bm25s）\n"
                "- semantic: 语义向量余弦相似度（lazyllm.thirdparty.numpy）\n\n"
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
                "Original Algorithms (using LazyLLM built-in modules):\n"
                "- random: Random sampling (baseline)\n"
                "- bm25: BM25 lexical similarity (lazyllm.thirdparty.bm25s)\n"
                "- semantic: Cosine similarity (lazyllm.thirdparty.numpy)\n\n"
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
        """Initialize BM25 index for lexical similarity mining.
        参考 lazyllm/tools/rag/component/bm25.py 的实现。
        """
        LOG.info(f"Initializing BM25 index for {len(corpus)} documents...")
        self._corpus = corpus
        corpus_tokens = bm25s.tokenize(
            [self._tokenizer(doc) for doc in corpus],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)
        LOG.info("BM25 index initialized successfully.")

    def _compute_corpus_embeddings(self, corpus: List[str]):
        """Compute embeddings for the entire corpus.
        使用 lazyllm.thirdparty.numpy 进行向量计算。
        """
        if self.embedding_serving is None:
            raise ValueError("Embedding serving is required for semantic mining strategy")

        LOG.info(f"Computing embeddings for {len(corpus)} documents...")
        self._corpus = corpus
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
        """Mine negatives using BM25 similarity.
        参考 lazyllm/tools/rag/component/bm25.py 的 retrieve 方法。
        """
        if self._bm25 is None or self._corpus != corpus:
            self._init_bm25(corpus)

        # Tokenize query using the same method as corpus
        tokenized_query = bm25s.tokenize(
            self._tokenizer(query),
            stopwords=self._stopwords,
            stemmer=self._stemmer
        )

        # Retrieve all documents with scores
        k = min(len(corpus), self.num_negatives + len(pos_set) + 10)  # Get more than needed
        indices, scores = self._bm25.retrieve(tokenized_query, k=k)

        # Filter out positives and take top-k
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = corpus[idx]
            if doc not in pos_set:
                results.append(doc)
                if len(results) >= self.num_negatives:
                    break

        return results

    def _cosine_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and corpus.
        参考 lazyllm/tools/rag/similarity.py 的 cosine 函数。
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms > 0, corpus_norms, 1)  # Avoid division by zero
        corpus_normalized = corpus_embeddings / corpus_norms

        # Compute cosine similarity
        similarities = np.dot(corpus_normalized, query_embedding)
        return similarities

    def _mine_semantic(self, query: str, pos_set: set, corpus: List[str]) -> List[str]:
        """Mine negatives using semantic similarity.
        使用 lazyllm.thirdparty.numpy 计算余弦相似度。
        """
        if self._corpus_embeddings is None or self._corpus != corpus:
            self._compute_corpus_embeddings(corpus)

        # Get query embedding
        query_embedding = self.embedding_serving.generate_embedding_from_input([query])[0]
        query_embedding = np.array(query_embedding)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self._corpus_embeddings)

        # Sort by similarity descending, filter out positives
        scored_docs = [(sim, doc) for sim, doc in zip(similarities, corpus) if doc not in pos_set]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top-k as hard negatives (high similarity but not positive)
        return [doc for _, doc in scored_docs[:self.num_negatives]]

    def forward_batch_input(
            self,
            inputs: List[dict],
            input_query_key: str = "query",
            input_pos_key: str = "pos",
            corpus_key: str = "passage",
            output_neg_key: str = "neg",
            corpus: Optional[List[str]] = None,
            **kwargs
    ) -> List[dict]:
        """
        Mine hard negative samples for each query.

        Args:
            inputs: List of dict with query-pos pairs
            input_query_key: Key for query field
            input_pos_key: Key for positive samples field (list of passages)
            corpus_key: Key for corpus passages (used if corpus not provided)
            output_neg_key: Key for output negative samples field
            corpus: Optional external corpus for mining negatives

        Returns:
            List of dict with mined hard negatives added
        """
        assert isinstance(inputs, list), "inputs must be a list of dict"

        # Build corpus from data if not provided
        if corpus is None:
            # Check if corpus_key exists in any item
            if inputs and corpus_key in inputs[0]:
                corpus = [item.get(corpus_key, "") for item in inputs]
            else:
                # Extract unique passages from all positive samples
                all_passages = []
                for item in inputs:
                    pos_list = item.get(input_pos_key, [])
                    if isinstance(pos_list, list):
                        all_passages.extend(pos_list)
                    else:
                        all_passages.append(pos_list)
                corpus = list(set(all_passages))

        LOG.info(f"Mining hard negatives for {len(inputs)} queries from corpus of {len(corpus)} documents...")

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
        for item in inputs:
            query = item.get(input_query_key, "")
            pos_samples = item.get(input_pos_key, [])

            # Convert to set for fast lookup
            if isinstance(pos_samples, list):
                pos_set = set(pos_samples)
            else:
                pos_set = {pos_samples}

            # Mine hard negatives
            negatives = mine_func(query, pos_set, corpus)

            new_row = item.copy()
            new_row[output_neg_key] = negatives
            results.append(new_row)

        LOG.info(f"Hard negative mining completed for {len(results)} samples.")
        return results
