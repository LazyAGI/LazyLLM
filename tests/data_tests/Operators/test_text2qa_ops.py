import os
import shutil

from lazyllm import config
from lazyllm.tools.data import Text2qa, EnQA


class MockModel:
    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self

class TestText2qaOperators:

    def setup_method(self):
        self.root_dir = './test_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_text2qa_text_to_chunks(self):
        # Example from data.operators.text2qa_ops.TextToChunks
        op = Text2qa.TextToChunks(input_key='content', output_key='chunk', chunk_size=10, tokenize=False)
        data = [{'content': 'line1\nline2\nline3\nline4'}]
        res = op(data)
        assert res == [
            {'content': 'line1\nline2\nline3\nline4', 'chunk': 'line1\nline2'},
            {'content': 'line1\nline2\nline3\nline4', 'chunk': 'line3\nline4'},
        ]

    def test_text2qa_empty_or_noise_filter(self):
        # Example from data.operators.text2qa_ops.empty_or_noise_filter
        op = Text2qa.empty_or_noise_filter(input_key='chunk')
        data = [{'chunk': 'hello'}, {'chunk': ''}, {'chunk': '\n'}]
        res = op(data)
        assert res == [{'chunk': 'hello'}]

    def test_text2qa_invalid_unicode_cleaner(self):
        # Example from data.operators.text2qa_ops.invalid_unicode_cleaner
        op = Text2qa.invalid_unicode_cleaner(input_key='chunk')
        data = [{'chunk': 'valid text\uFFFE tail'}]
        res = op(data)
        assert res == [{'chunk': 'valid text tail'}]

    def test_text2qa_chunk_to_qa(self):
        llm = MockModel({'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'})
        op = Text2qa.ChunkToQA(input_key='chunk', query_key='query', answer_key='answer', model=llm)
        data = [{'chunk': '今天是晴天！'}]
        res = op(data)
        assert len(res) == 1
        assert 'chunk' in res[0] and 'query' in res[0] and 'answer' in res[0]

    def test_text2qa_qa_scorer(self):
        # Example from data.operators.text2qa_ops.QAScorer
        llm = MockModel({'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！', 'score': 1})
        op = Text2qa.QAScorer(input_key='chunk', output_key='score', query_key='query', answer_key='answer', model=llm)
        data = [
            {'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'}
        ]
        res = op(data)
        assert len(res) == 1
        assert res[0].get('score') == 1


class TestEnQAOperators:

    def setup_method(self):
        self.root_dir = './test_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_query_rewriter(self):
        model = MockModel({'query': 'What is AI?', 'rewrite_querys': ['Ai is xxx', 'Ai is yyyy']})
        rewrite_num = 2
        op = EnQA.QueryRewriter(
            input_key='query',
            output_key='rewrite_querys',
            rewrite_num=rewrite_num,
            model=model
        )

        data = {'query': 'What is AI?'}

        res = op(data)

        assert 'rewrite_querys' in res[0]
        assert len(res[0]['rewrite_querys']) == 2

    def test_diversity_scorer(self):
        model = MockModel({'rewrite_querys': ['a', 'b'], 'diversity_querys': [1, 0]})

        op = EnQA.DiversityScorer(
            input_key='rewrite_querys',
            output_key='diversity_querys',
            model=model
        )

        data = {'rewrite_querys': ['a', 'b']}

        res = op(data)

        assert 'diversity_querys' in res[0]
        assert len(res[0]['diversity_querys']) == 2

    def test_post_processor(self):
        op = EnQA.post_processor(
            input_key='diversity_querys'
        )

        data = {
            'diversity_querys': [
                {
                    'rewritten_query': 'a',
                    'diversity_score': 1
                }
            ]
        }

        res = op(data)

        assert isinstance(res, list)

    def test_diversity_filter(self):
        op = EnQA.diversity_filter(
            input_key='score',
            min_score=1
        )

        data = {'score': 0}

        res = op(data)

        assert res == []
