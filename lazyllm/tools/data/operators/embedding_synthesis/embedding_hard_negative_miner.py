import random
from typing import List, Optional, Callable

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.thirdparty import numpy as np, bm25s, jieba, Stemmer
from lazyllm.tools.rag.component.stopwords import STOPWORDS_CHINESE

from ...base_data import data_register


# Get or create embedding group
if 'data' in LazyLLMRegisterMetaClass.all_clses and 'embedding' in LazyLLMRegisterMetaClass.all_clses['data']:
    embedding = LazyLLMRegisterMetaClass.all_clses['data']['embedding'].base
else:
    embedding = data_register.new_group('embedding')


def _normalize_pos_samples(pos_samples) -> set:
    if isinstance(pos_samples, list):
        return set(pos_samples)
    return {pos_samples}


@data_register('data.embedding', rewrite_func='forward_batch_input')
def build_embedding_corpus(
    inputs: List[dict],
    input_pos_key: str = 'pos',
    corpus_key: str = 'passage',
) -> List[dict]:
    all_passages = []

    for item in inputs:
        pos_list = item.get(input_pos_key, [])
        if isinstance(pos_list, list):
            all_passages.extend(pos_list)
        else:
            all_passages.append(pos_list)

        if corpus_key in item:
            all_passages.append(item[corpus_key])

    corpus = list(set(all_passages))
    LOG.info(f'Built corpus with {len(corpus)} unique passages.')

    return [{**item, '_corpus': corpus} for item in inputs]


@data_register('data.embedding', rewrite_func='forward_batch_input')
def build_embedding_corpus_from_list(
    inputs: List[dict],
    corpus: Optional[List[str]] = None,
) -> List[dict]:
    if corpus is None:
        corpus = []

    LOG.info(f'Using external corpus with {len(corpus)} passages.')
    return [{**item, '_corpus': corpus} for item in inputs]


class EmbeddingInitBM25(embedding):

    def __init__(self, language: str = 'zh', **kwargs):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.language = language
        self._setup_tokenizer(language)

    def _setup_tokenizer(self, language: str):
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

    def forward_batch_input(self, inputs: List[dict], **kwargs) -> List[dict]:
        if not inputs:
            return inputs

        corpus = inputs[0].get('_corpus') or []
        if not corpus:
            LOG.warning('No corpus found for BM25 initialization.')
            return [
                {**item, '_bm25': None, '_bm25_corpus': []}
                for item in inputs
            ]

        LOG.info(f'Initializing BM25 index for {len(corpus)} documents...')

        corpus_tokens = bm25s.tokenize(
            [self._tokenizer(doc) for doc in corpus],
            stopwords=self._stopwords,
            stemmer=self._stemmer,
        )

        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

        LOG.info('BM25 index initialized.')

        return [
            {
                **item,
                '_bm25': bm25_index,
                '_bm25_corpus': corpus,
                '_bm25_tokenizer': self._tokenizer,
                '_bm25_stopwords': self._stopwords,
                '_bm25_stemmer': self._stemmer,
            }
            for item in inputs
        ]


class EmbeddingInitSemantic(embedding):

    def __init__(
        self,
        embedding_serving: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.embedding_serving = embedding_serving

    def forward_batch_input(self, inputs: List[dict], **kwargs) -> List[dict]:
        if not inputs:
            return inputs

        # Verify all inputs share the same corpus for consistency
        corpus = inputs[0].get('_corpus') or []
        if not all(item.get('_corpus') == corpus for item in inputs):
            LOG.warning('Not all inputs share the same corpus. Using corpus from first item.')

        if not corpus or self.embedding_serving is None:
            LOG.warning(
                'No corpus or embedding_serving for semantic initialization.'
            )
            return [
                {
                    **item,
                    '_semantic_embeddings': None,
                    '_semantic_corpus': corpus or [],
                }
                for item in inputs
            ]

        LOG.info(f'Computing embeddings for {len(corpus)} documents...')
        embeddings = np.array(self.embedding_serving(corpus))
        LOG.info('Embeddings computed.')

        return [
            {
                **item,
                '_semantic_embeddings': embeddings,
                '_semantic_corpus': corpus,
            }
            for item in inputs
        ]


@data_register('data.embedding', rewrite_func='forward', _concurrency_mode='process')
def mine_bm25_negatives(
    data: dict,
    num_negatives: int = 7,
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    output_neg_key: str = 'neg',
) -> dict:
    bm25_index = data.get('_bm25')
    corpus = data.get('_bm25_corpus') or []
    tokenizer = data.get('_bm25_tokenizer', lambda t: t)
    stopwords = data.get('_bm25_stopwords')
    stemmer = data.get('_bm25_stemmer')

    if bm25_index is None:
        LOG.warning('BM25 index not initialized.')
        return {**data, output_neg_key: []}

    query = data.get(input_query_key, '')
    pos_samples = data.get(input_pos_key, [])

    if not query:
        return {**data, output_neg_key: []}

    pos_set = _normalize_pos_samples(pos_samples)

    tokenized_query = bm25s.tokenize(
        tokenizer(query),
        stopwords=stopwords,
        stemmer=stemmer,
    )

    k = min(
        len(corpus) if corpus else 0,
        num_negatives + len(pos_set) + 10,
    )

    indices, _ = bm25_index.retrieve(tokenized_query, k=k)

    negatives = []

    if not corpus:
        return {**data, output_neg_key: []}

    for idx in indices[0]:
        doc = corpus[idx]
        if doc not in pos_set:
            negatives.append(doc)
            if len(negatives) >= num_negatives:
                break

    return {**data, output_neg_key: negatives}


@data_register('data.embedding', rewrite_func='forward', _concurrency_mode='process')
def mine_random_negatives(
    data: dict,
    num_negatives: int = 7,
    seed: int = 42,
    input_query_key: str = 'query',
    input_pos_key: str = 'pos',
    output_neg_key: str = 'neg',
) -> dict:

    corpus = data.get('_corpus') or []
    if not corpus:
        return {**data, output_neg_key: []}

    query = data.get(input_query_key, '')
    pos_samples = data.get(input_pos_key, [])

    if not query:
        return {**data, output_neg_key: []}

    pos_set = _normalize_pos_samples(pos_samples)
    candidates = [doc for doc in corpus if doc not in pos_set]

    if len(candidates) <= num_negatives:
        negatives = candidates
    else:
        local_random = random.Random(f'{seed}_{query}')
        negatives = local_random.sample(
            candidates,
            num_negatives,
        )

    return {**data, output_neg_key: negatives}


class EmbeddingMineSemanticNegatives(embedding):

    def __init__(
        self,
        num_negatives: int = 7,
        embedding_serving: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.num_negatives = num_negatives
        self.embedding_serving = embedding_serving

    @staticmethod
    def _cosine_similarity(
        query_emb: np.ndarray,
        corpus_embs: np.ndarray,
    ) -> np.ndarray:
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        corpus_norms = np.linalg.norm(
            corpus_embs,
            axis=1,
            keepdims=True,
        )
        corpus_norms = np.where(corpus_norms > 0, corpus_norms, 1)
        corpus_normalized = corpus_embs / corpus_norms

        return np.dot(corpus_normalized, query_emb)

    def forward(
        self,
        data: dict,
        input_query_key: str = 'query',
        input_pos_key: str = 'pos',
        output_neg_key: str = 'neg',
        **kwargs,
    ) -> dict:
        corpus_embeddings = data.get('_semantic_embeddings')
        corpus = data.get('_semantic_corpus') or []

        if corpus_embeddings is None:
            LOG.warning('Semantic embeddings not initialized.')
            return {**data, output_neg_key: []}

        query = data.get(input_query_key, '')
        pos_samples = data.get(input_pos_key, [])

        if not query:
            return {**data, output_neg_key: []}

        if self.embedding_serving is None:
            return {**data, output_neg_key: []}

        pos_set = _normalize_pos_samples(pos_samples)

        query_embedding = np.array(
            self.embedding_serving([query])[0]
        )

        similarities = self._cosine_similarity(
            query_embedding,
            corpus_embeddings,
        )

        scored_docs = [
            (sim, doc)
            for sim, doc in zip(similarities, corpus)
            if doc not in pos_set
        ]

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        negatives = [
            doc for _, doc in scored_docs[: self.num_negatives]
        ]

        return {**data, output_neg_key: negatives}
