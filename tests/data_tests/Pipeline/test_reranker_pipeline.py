import os
import shutil
import pytest

from lazyllm import config
from lazyllm.tools.data.pipelines.reranker_pipelines import (
    build_reranker_dataformatter_pipeline,
    build_convert_from_embed_pipeline,
    build_reranker_hard_neg_pipeline,
    build_reranker_qa_generate_pipeline,
)


class MockLLMServe:
    def __init__(self, return_value=None):
        self._return_value = return_value or {'queries': [{'query': 'mock query', 'difficulty': 'medium'}]}

    def start(self):
        return self

    def prompt(self, system_prompt):
        return self

    def formatter(self, formatter):
        return self

    def __call__(self, prompt):
        return self._return_value


class MockLLM:
    def __init__(self, return_value=None):
        self._serve = MockLLMServe(return_value=return_value)

    def share(self, prompt=None, format=None, stream=None, history=None):
        return self._serve


class TestRerankerPipeline:
    def setup_method(self):
        self.root_dir = './test_reranker_pipeline'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_build_reranker_dataformatter_pipeline_flagreranker_run(self):
        ppl = build_reranker_dataformatter_pipeline(
            output_format='flagreranker',
            train_group_size=8,
        )
        data = [
            {'query': 'what is ML?', 'pos': ['doc1', 'doc2'], 'neg': ['n1', 'n2', 'n3']},
            {'query': 'what is DL?', 'pos': ['docA'], 'neg': ['nA', 'nB', 'nC', 'nD', 'nE', 'nF', 'nG', 'nH']},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) >= 1
        for item in res:
            assert 'query' in item and 'pos' in item and 'neg' in item
            assert isinstance(item['pos'], list) and isinstance(item['neg'], list)
            assert len(item['neg']) == 7

    def test_build_reranker_dataformatter_pipeline_cross_encoder_run(self):
        ppl = build_reranker_dataformatter_pipeline(output_format='cross_encoder')
        data = [{'query': 'q1', 'pos': ['doc1'], 'neg': ['doc2', 'doc3']}]
        res = ppl(data)
        assert isinstance(res, list)
        for item in res:
            assert 'query' in item and 'document' in item and 'label' in item
            assert item['label'] in (0, 1)

    def test_build_reranker_dataformatter_pipeline_pairwise_run(self):
        ppl = build_reranker_dataformatter_pipeline(output_format='pairwise')
        data = [{'query': 'q1', 'pos': ['p1'], 'neg': ['n1', 'n2']}]
        res = ppl(data)
        assert isinstance(res, list)
        for item in res:
            assert 'query' in item and 'doc_pos' in item and 'doc_neg' in item

    def test_build_reranker_dataformatter_pipeline_custom_keys(self):
        ppl = build_reranker_dataformatter_pipeline(
            output_format='flagreranker',
            input_query_key='q',
            input_pos_key='positive',
            input_neg_key='negative',
            train_group_size=4,
        )
        data = [{'q': 'question?', 'positive': ['p1'], 'negative': ['n1', 'n2', 'n3', 'n4', 'n5']}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) >= 1
        assert res[0]['query'] == 'question?'
        assert res[0]['pos'] == ['p1']
        assert len(res[0]['neg']) == 3

    def test_build_reranker_dataformatter_pipeline_invalid_data_filtered(self):
        ppl = build_reranker_dataformatter_pipeline(output_format='flagreranker', train_group_size=8)
        data = [
            {'query': 'valid?', 'pos': ['p1'], 'neg': ['n1']},
            {'query': '', 'pos': ['p2'], 'neg': ['n2']},
            {'pos': ['p3'], 'neg': ['n3']},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) <= 2

    def test_reranker_dataformatter_pipeline_unknown_format_raises(self):
        with pytest.raises(ValueError, match='Unknown output format'):
            build_reranker_dataformatter_pipeline(output_format='unknown')

    def test_build_convert_from_embed_pipeline_run(self):
        ppl = build_convert_from_embed_pipeline(
            input_query_key='query',
            input_pos_key='pos',
            input_neg_key='neg',
            adjust_neg_count=5,
            seed=42,
        )
        data = [
            {'query': 'q1', 'pos': ['p1'], 'neg': ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) >= 1
        for item in res:
            assert 'query' in item and 'pos' in item and 'neg' in item
            assert len(item['neg']) == 5

    def test_build_convert_from_embed_pipeline_invalid_filtered(self):
        ppl = build_convert_from_embed_pipeline(adjust_neg_count=3, seed=42)
        data = [
            {'query': 'ok', 'pos': ['p1'], 'neg': ['n1', 'n2']},
            {'query': '', 'pos': ['p2'], 'neg': ['n2']},
        ]
        res = ppl(data)
        valid = [r for r in res if r]
        assert len(valid) >= 1

    def test_build_reranker_hard_neg_pipeline_random_run(self):
        run = build_reranker_hard_neg_pipeline(
            mining_strategy='random',
            num_negatives=2,
            seed=42,
            corpus_dir=os.path.abspath(self.root_dir),
        )
        data = [
            {'query': 'what is ML?', 'pos': ['Machine learning is a branch of AI.']},
            {'query': 'what is DL?', 'pos': ['Deep learning uses neural networks.']},
        ]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) <= len(data)
        if len(res) > 0:
            for item in res:
                assert 'neg' in item
                assert isinstance(item['neg'], list)
                assert len(item['neg']) <= 2
                for doc in item['neg']:
                    assert doc not in (item.get('pos') or [])

    def test_build_reranker_hard_neg_pipeline_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match='Unknown mining strategy'):
            build_reranker_hard_neg_pipeline(mining_strategy='invalid')

    def test_reranker_qa_generate_pipeline_with_mock_llm_and_data(self):
        mock_llm = MockLLM(return_value={
            'queries': [
                {'query': 'Generated Q1?', 'difficulty': 'easy'},
                {'query': 'Generated Q2?', 'difficulty': 'medium'},
            ]
        })
        ppl = build_reranker_qa_generate_pipeline(
            llm_serving=mock_llm,
            input_key='passage',
            output_query_key='query',
            num_queries=2,
        )
        data = [{'passage': 'This is a sample passage about machine learning.'}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) >= 1
        for item in res:
            assert 'query' in item
            assert 'pos' in item

    def test_build_reranker_qa_generate_pipeline_custom_keys_and_difficulty(self):
        mock_llm = MockLLM(return_value={
            'queries': [
                {'query': 'Custom Q1?', 'difficulty': 'easy'},
                {'query': 'Custom Q2?', 'difficulty': 'hard'},
            ]
        })
        ppl = build_reranker_qa_generate_pipeline(
            llm_serving=mock_llm,
            input_key='passage',
            output_query_key='query',
            num_queries=2,
            difficulty_levels=['easy', 'hard'],
        )
        data = [{'passage': 'A long passage about machine learning and deep learning.'}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 2
        assert res[0]['query'] == 'Custom Q1?'
        assert res[0]['difficulty'] == 'easy'
        assert res[0]['pos'] == [data[0]['passage']]
        assert res[1]['query'] == 'Custom Q2?'
        assert res[1]['difficulty'] == 'hard'

    def test_reranker_qa_generate_pipeline_llm_none_drops_item(self):
        ppl = build_reranker_qa_generate_pipeline(llm_serving=None)
        data = [{'passage': 'some passage'}]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 0
