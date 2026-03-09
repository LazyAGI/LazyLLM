import json
import os
import shutil
import tempfile
import numpy as np
from lazyllm import config
from lazyllm.tools.data import embedding


class MockLLMServe:

    def __init__(self, return_value=None, raise_exc=False):
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.started = False

    def start(self):
        self.started = True
        return self

    def prompt(self, system_prompt):
        return self

    def formatter(self, formatter):
        return self

    def __call__(self, prompt):
        if self._raise_exc:
            raise RuntimeError('mock error')
        return self._return_value


class MockLLM:

    def __init__(self, answer_return=None, score_return=None,
                 answer_raise=False, score_raise=False):
        self.answer_serve = MockLLMServe(answer_return, answer_raise)
        self.score_serve = MockLLMServe(score_return, score_raise)
        self._share_count = 0

    def share(self, prompt=None, format=None, stream=None, history=None):
        # First share returns answer_serve, second returns score_serve
        if format is not None and self._share_count == 1:
            self.score_serve.formatter(format)
        if self._share_count == 0:
            self._share_count += 1
            return self.answer_serve
        else:
            return self.score_serve


class MockEmbeddingServing:

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def __call__(self, texts):
        self.call_count += 1
        # Return random embeddings for testing
        if isinstance(texts, str):
            texts = [texts]
        return np.random.randn(len(texts), self.embedding_dim).tolist()


class TestEmbeddingSynthesisOperators:
    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)
    # ========== Query Generator Operators ==========

    def test_embedding_generate_queries(self):
        mock_llm = MockLLM(answer_return=[
            {'query': 'What is AI?', 'type': 'factual'},
            {'query': 'How does AI work?', 'type': 'semantic'}
        ])
        op = embedding.EmbeddingGenerateQueries(llm=mock_llm, num_queries=2)
        result = op([{'passage': 'AI is a technology.'}])[0]
        assert '_query_response' in result

    def test_embedding_parse_queries(self):
        op = embedding.EmbeddingParseQueries()
        data = {
            'passage': 'AI is a technology.',
            '_query_response': json.dumps([
                {'query': 'What is AI?', 'type': 'factual'},
                {'query': 'How does AI work?', 'type': 'semantic'}
            ])
        }
        result = op([data])
        assert len(result) == 2
        assert 'query' in result[0]
        assert 'pos' in result[0]

    # ========== Hard Negative Miner Operators ==========

    def test_build_embedding_corpus(self):
        op = embedding.build_embedding_corpus()
        data = [
            {'query': 'Q1', 'pos': ['P1', 'P2']},
            {'query': 'Q2', 'pos': ['P2', 'P3']}
        ]
        result = op(data)
        assert '_corpus' in result[0]
        assert os.path.exists(result[0]['_corpus'])

    def test_embedding_init_bm25(self):
        # First build corpus
        corpus_op = embedding.build_embedding_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence', 'AI is a technology']}
        ]
        data_with_corpus = corpus_op(data)

        # Then init BM25
        op = embedding.EmbeddingInitBM25(language='en')
        result = op(data_with_corpus)
        assert '_bm25' in result[0]

    def test_mine_bm25_negatives(self):
        # Build corpus with more diverse content
        corpus_op = embedding.build_embedding_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']},
            {'query': 'What is ML?', 'pos': ['ML is machine learning']},
            {'query': 'What is Python?', 'pos': ['Python is a programming language']}
        ]
        data_with_corpus = corpus_op(data)

        bm25_op = embedding.EmbeddingInitBM25(language='en')
        data_with_bm25 = bm25_op(data_with_corpus)

        # Mine negatives
        func = embedding.mine_bm25_negatives(num_negatives=2)
        result = func([data_with_bm25[0]])[0]
        assert 'neg' in result  # May be empty list if no negatives found

    def test_mine_random_negatives(self):
        # Build corpus first - need more diverse corpus
        corpus_op = embedding.build_embedding_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']},
            {'query': 'What is ML?', 'pos': ['ML is machine learning']},
            {'query': 'What is DL?', 'pos': ['DL is deep learning']}
        ]
        data_with_corpus = corpus_op(data)

        # Mine random negatives - use original function directly
        func = embedding.mine_random_negatives(num_negatives=2)
        result = func(data_with_corpus[0])[0]
        assert 'neg' in result

    def test_embedding_init_semantic(self):
        # Build corpus first
        corpus_op = embedding.build_embedding_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']}
        ]
        data_with_corpus = corpus_op(data)

        # Init semantic with mock embedding serving
        mock_embedding = MockEmbeddingServing()
        op = embedding.EmbeddingInitSemantic(embedding_serving=mock_embedding)
        result = op.forward_batch_input(data_with_corpus)
        assert '_semantic_embeddings_path' in result[0]
        assert os.path.exists(result[0]['_semantic_embeddings_path'])

    def test_embedding_mine_semantic_negatives(self):
        # Build corpus and init semantic
        corpus_op = embedding.build_embedding_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence', 'ML is machine learning']}
        ]
        data_with_corpus = corpus_op(data)

        mock_embedding = MockEmbeddingServing()
        init_op = embedding.EmbeddingInitSemantic(embedding_serving=mock_embedding)
        data_with_semantic = init_op.forward_batch_input(data_with_corpus)

        # Mine semantic negatives
        op = embedding.EmbeddingMineSemanticNegatives(
            num_negatives=1,
            embedding_serving=mock_embedding
        )
        result = op([data_with_semantic[0]])[0]
        assert 'neg' in result

    # ========== Data Formatter Operators ==========

    def test_embedding_format_flag_embedding(self):
        op = embedding.EmbeddingFormatFlagEmbedding()
        data = {
            'query': 'What is AI?',
            'pos': ['AI is artificial intelligence'],
            'neg': ['ML is machine learning']
        }
        result = op([data])[0]
        assert result['query'] == 'What is AI?'
        assert 'pos' in result
        assert 'neg' in result

    def test_embedding_format_sentence_transformers(self):
        op = embedding.EmbeddingFormatSentenceTransformers()
        data = {
            'query': 'What is AI?',
            'pos': ['AI is artificial intelligence'],
            'neg': ['ML is machine learning']
        }
        result = op([data])
        assert len(result) == 1
        assert 'anchor' in result[0]
        assert 'positive' in result[0]
        assert 'negative' in result[0]

    def test_embedding_format_triplet(self):
        op = embedding.EmbeddingFormatTriplet()
        data = {
            'query': 'What is AI?',
            'pos': ['AI is artificial intelligence'],
            'neg': ['ML is machine learning']
        }
        result = op([data])
        assert len(result) == 1
        assert 'query' in result[0]
        assert 'positive' in result[0]
        assert 'negative' in result[0]

    def test_embedding_train_test_splitter(self):
        op = embedding.EmbeddingTrainTestSplitter(test_size=0.3, seed=42)
        data = [
            {'query': f'Q{i}', 'pos': [f'P{i}']}
            for i in range(10)
        ]
        result = op.forward_batch_input(data)
        assert len(result) == 10
        splits = [item['split'] for item in result]
        assert 'train' in splits
        assert 'test' in splits

    # ========== Data Augmentor Operators ==========

    def test_embedding_query_rewrite(self):
        mock_llm = MockLLM(answer_return=['rewritten query 1', 'rewritten query 2'])
        op = embedding.EmbeddingQueryRewrite(llm=mock_llm, num_augments=2)
        data = {'query': 'original query'}
        result = op([data])
        assert len(result) == 2
        assert result[0]['is_augmented'] is True

    def test_embedding_adjacent_word_swap(self):
        op = embedding.EmbeddingAdjacentWordSwap(num_augments=2)
        data = {'query': 'this is a test query'}
        result = op([data])
        # May return 0, 1, or 2 augmentations depending on word count
        assert len(result) <= 2
        if len(result) > 0:
            assert result[0]['is_augmented'] is True
