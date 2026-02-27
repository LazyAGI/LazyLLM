import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data import genCot


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

class TestGenCotOperators:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_cot_generator(self):
        model = MockModel({'query': 'What is 2+2?', 'cot_answer': 'XXXXX \\boxed{4}'})

        op = genCot.CoTGenerator(
            input_key='query',
            output_key='cot_answer',
            model=model
        )

        data = {'query': 'What is 2+2?'}

        res = op(data)

        assert 'cot_answer' in res[0]

    def test_self_consistency(self):
        model = MockModel('XXXXX \\boxed{9}')

        op = genCot.SelfConsistencyCoTGenerator(
            input_key='query',
            output_key='cot_answer',
            num_samples=2,
            model=model
        )

        data = {'query': 'What is 3*3?'}

        res = op(data)

        assert 'cot_answer' in res[0]

    def test_answer_verify(self):
        op = genCot.answer_verify(
            answer_key='reference',
            infer_key='llm_extracted',
            output_key='is_equal'
        )

        data = {
            'reference': '1/2',
            'llm_extracted': '0.5'
        }

        res = op(data)

        assert 'is_equal' in res[0]
