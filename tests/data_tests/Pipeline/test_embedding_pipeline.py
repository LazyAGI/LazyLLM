import os
import shutil
from lazyllm import config
from lazyllm.tools.data.pipelines.embedding_pipelines import (
    build_embedding_data_augmentation_pipeline,
    build_embedding_data_formatter_pipeline,
    build_embedding_hard_neg_pipeline,
    build_query_generation_pipeline,
)


class TestEmbeddingPipeline:

    def setup_method(self):
        self.root_dir = './test_embedding_pipeline'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_embedding_data_augmentation_pipeline_no_methods(self):
        run = build_embedding_data_augmentation_pipeline(
            keep_original=True,
            augment_methods=[],
        )
        data = [{'query': 'hello world'}, {'query': 'foo bar'}]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) == 2
        assert res == data

    def test_embedding_data_augmentation_pipeline_synonym_replace(self):
        run = build_embedding_data_augmentation_pipeline(
            keep_original=True,
            augment_methods=['synonym_replace'],
            num_augments=2,
        )
        data = [{'query': 'one two three four'}, {'query': 'a b c'}]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) >= 2

    def test_embedding_data_formatter_pipeline(self):
        run = build_embedding_data_formatter_pipeline(
            input_query_key='query',
            input_pos_key='pos',
            input_neg_key='neg',
            output_format='flagembedding',
        )
        data = [
            {'query': 'what is ML?', 'pos': ['Machine learning is...'], 'neg': []},
            {'query': 'what is DL?', 'pos': ['Deep learning is...'], 'neg': ['Something else']},
        ]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) == 2
        for item in res:
            assert 'query' in item and 'pos' in item and 'neg' in item

    def test_embedding_data_formatter_pipeline_triplet(self):
        run = build_embedding_data_formatter_pipeline(output_format='triplet')
        data = [{'query': 'q1', 'pos': ['p1'], 'neg': ['n1']}]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) >= 1
        for item in res:
            assert 'query' in item and 'positive' in item and 'negative' in item

    def test_embedding_hard_neg_pipeline_random(self):
        run = build_embedding_hard_neg_pipeline(
            mining_strategy='random',
            num_negatives=2,
            seed=42,
        )
        data = [
            {'query': 'what is ML?', 'pos': ['Machine learning is a field.', 'ML is cool.']},
            {'query': 'what is DL?', 'pos': ['Deep learning uses neural networks.']},
        ]
        res = run(data)
        assert isinstance(res, list)
        assert len(res) == 2
        for item in res:
            assert 'neg' in item and isinstance(item['neg'], list)

    def test_query_generation_pipeline(self):
        ppl = build_query_generation_pipeline(
            input_key='passage',
            output_query_key='query',
            num_queries=1,
            llm=None,
        )
        data = [
            {'passage': 'Machine learning is a subset of AI.'},
            {'passage': 'Deep learning uses neural networks.'},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) <= len(data) * 2
