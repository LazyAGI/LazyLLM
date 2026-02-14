import json
import os
import shutil
import numpy as np
from lazyllm import config
# Import reranker_synthesis to register it first
from lazyllm.tools.data.operators import reranker_synthesis  # noqa: F401
from lazyllm.tools.data import reranker


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

    def __init__(self, return_value=None, raise_exc=False):
        self._return_value = return_value
        self._raise_exc = raise_exc

    def share(self, prompt=None, format=None, stream=None, history=None):
        serve = MockLLMServe(self._return_value, self._raise_exc)
        return serve


class MockEmbeddingServing:

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def __call__(self, texts):
        self.call_count += 1
        if isinstance(texts, str):
            texts = [texts]
        return np.random.randn(len(texts), self.embedding_dim).tolist()


class TestRerankerSynthesisOperators:

    def setup_method(self):
        self.root_dir = './test_reranker_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    # ========== Query Generator Operators ==========

    def test_reranker_generate_queries(self):
        mock_llm = MockLLM(return_value=[
            {'query': 'What is AI?', 'difficulty': 'easy'},
            {'query': 'How does AI work?', 'difficulty': 'medium'}
        ])
        op = reranker.RerankerGenerateQueries(llm_serving=mock_llm, num_queries=2)
        result = op([{'passage': 'AI is a technology.'}])[0]
        assert '_query_response' in result

    def test_reranker_parse_queries(self):
        op = reranker.RerankerParseQueries()
        data = {
            'passage': 'AI is a technology.',
            '_query_response': json.dumps([
                {'query': 'What is AI?', 'difficulty': 'easy'},
                {'query': 'How does AI work?', 'difficulty': 'medium'}
            ])
        }
        result = op([data])
        assert len(result) == 2
        assert 'query' in result[0]
        assert 'pos' in result[0]
        assert result[0]['pos'] == ['AI is a technology.']

    # ========== Corpus Builder Operators ==========

    def test_build_reranker_corpus(self):
        op = reranker.build_reranker_corpus()
        data = [
            {'query': 'Q1', 'pos': ['P1', 'P2']},
            {'query': 'Q2', 'pos': ['P2', 'P3']}
        ]
        result = op(data)
        assert '_corpus' in result[0]
        assert os.path.exists(result[0]['_corpus'])

    def test_reranker_init_bm25(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence', 'AI is a technology']}
        ]
        data_with_corpus = corpus_op(data)

        op = reranker.RerankerInitBM25(language='en')
        result = op(data_with_corpus)
        assert '_bm25' in result[0]
        assert '_bm25_corpus' in result[0]

    # ========== Hard Negative Miner Operators ==========

    def test_reranker_mine_random_negatives(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']},
            {'query': 'What is ML?', 'pos': ['ML is machine learning']},
            {'query': 'What is DL?', 'pos': ['DL is deep learning']}
        ]
        data_with_corpus = corpus_op(data)

        op = reranker.RerankerMineRandomNegatives(num_negatives=2)
        result = op(data_with_corpus)
        assert 'neg' in result[0]

    def test_reranker_mine_bm25_negatives(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']},
            {'query': 'What is ML?', 'pos': ['ML is machine learning']},
            {'query': 'What is Python?', 'pos': ['Python is a programming language']}
        ]
        data_with_corpus = corpus_op(data)

        bm25_op = reranker.RerankerInitBM25(language='en')
        data_with_bm25 = bm25_op(data_with_corpus)

        op = reranker.RerankerMineBM25Negatives(num_negatives=2)
        result = op(data_with_bm25)
        assert 'neg' in result[0]

    def test_reranker_init_semantic(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']}
        ]
        data_with_corpus = corpus_op(data)

        mock_embedding = MockEmbeddingServing()
        op = reranker.RerankerInitSemantic(embedding_serving=mock_embedding)
        result = op.forward_batch_input(data_with_corpus)
        assert '_semantic_embeddings_path' in result[0]
        assert os.path.exists(result[0]['_semantic_embeddings_path'])

    def test_reranker_mine_semantic_negatives(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence', 'ML is machine learning']}
        ]
        data_with_corpus = corpus_op(data)

        mock_embedding = MockEmbeddingServing()
        init_op = reranker.RerankerInitSemantic(embedding_serving=mock_embedding)
        data_with_semantic = init_op(data_with_corpus)

        op = reranker.RerankerMineSemanticNegatives(
            num_negatives=1,
            embedding_serving=mock_embedding
        )
        result = op(data_with_semantic)
        assert 'neg' in result[0]

    def test_reranker_mine_mixed_negatives(self):
        corpus_op = reranker.build_reranker_corpus()
        data = [
            {'query': 'What is AI?', 'pos': ['AI is artificial intelligence']},
            {'query': 'What is ML?', 'pos': ['ML is machine learning']}
        ]
        data_with_corpus = corpus_op(data)

        mock_embedding = MockEmbeddingServing()

        init_bm25_op = reranker.RerankerInitBM25(language='en')
        data_with_bm25 = init_bm25_op(data_with_corpus)

        init_semantic_op = reranker.RerankerInitSemantic(embedding_serving=mock_embedding)
        data_with_both = init_semantic_op(data_with_bm25)

        op = reranker.RerankerMineMixedNegatives(
            num_negatives=2,
            embedding_serving=mock_embedding,
            bm25_ratio=0.5
        )
        result = op(data_with_both)
        assert 'neg' in result[0]

    # ========== Data Formatter Operators ==========

    def test_validate_reranker_data(self):
        validate_op = reranker.validate_reranker_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        result = validate_op([data])[0]
        assert result['_is_valid'] is True

    def test_validate_reranker_data_missing_query(self):
        validate_op = reranker.validate_reranker_data()
        data = {'query': '', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        result = validate_op([data])[0]
        assert result['_is_valid'] is False

    def test_reranker_format_flag_reranker(self):
        validate_op = reranker.validate_reranker_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML', 'DL is DL']}
        validated = validate_op([data])[0]

        op = reranker.RerankerFormatFlagReranker(train_group_size=4)
        result = op([validated])
        assert len(result) == 1
        assert result[0]['query'] == 'What is AI?'
        assert 'pos' in result[0]
        assert 'neg' in result[0]

    def test_reranker_format_cross_encoder(self):
        validate_op = reranker.validate_reranker_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        validated = validate_op([data])[0]

        op = reranker.RerankerFormatCrossEncoder()
        result = op([validated])
        assert len(result) == 2
        assert result[0]['label'] == 1
        assert result[1]['label'] == 0

    def test_reranker_format_pairwise(self):
        validate_op = reranker.validate_reranker_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        validated = validate_op([data])[0]

        op = reranker.RerankerFormatPairwise()
        result = op([validated])
        assert len(result) == 1
        assert 'doc_pos' in result[0]
        assert 'doc_neg' in result[0]

    def test_reranker_train_test_splitter(self):
        op = reranker.RerankerTrainTestSplitter(test_size=0.3, seed=42)
        data = [
            {'query': f'Q{i}', 'pos': [f'P{i}']}
            for i in range(10)
        ]
        result = op.forward_batch_input(data)
        assert len(result) == 10
        splits = [item['split'] for item in result]
        assert 'train' in splits
        assert 'test' in splits

    # ========== Embedding Converter Operators ==========

    def test_validate_reranker_embedding_data(self):
        validate_op = reranker.validate_reranker_embedding_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        result = validate_op([data])[0]
        assert result['_is_valid'] is True

    def test_reranker_adjust_negatives(self):
        validate_op = reranker.validate_reranker_embedding_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML', 'DL is DL', 'RL is RL']}
        validated = validate_op([data])[0]

        op = reranker.RerankerAdjustNegatives(adjust_neg_count=5)
        result = op([validated])[0]
        assert len(result['_neg']) == 5

    def test_reranker_build_format(self):
        validate_op = reranker.validate_reranker_embedding_data()
        data = {'query': 'What is AI?', 'pos': ['AI is AI'], 'neg': ['ML is ML']}
        validated = validate_op([data])[0]

        op = reranker.RerankerBuildFormat()
        result = op([validated])[0]
        assert 'query' in result
        assert 'pos' in result
        assert 'neg' in result
