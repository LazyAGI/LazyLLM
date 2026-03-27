import os
import shutil
import tempfile
from lazyllm import config
from lazyllm.tools.data.pipelines.math_pipelines import build_math_cot_pipeline


class MockModel:
    def __init__(self, mock_response):
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


class TestTextMathQAPipeline:

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

    def test_math_cot_pipeline(self):

        excepted_output = 'cot xxxx \\boxed{0.5}'

        model = MockModel(excepted_output)

        ppl = build_math_cot_pipeline(
            question_key='question',
            reference_key='reference',
            answer_key='answer',
            extracted_key='math_answer',
            verify_key='is_equal',
            model=model,
            num_samples=3
        )

        data = {
            'question': 'answer for 1 divided by 2?',
            'answer': 'to solve 1/2 we xxxxx is 0.5',
            'reference': '0.5'
        }

        res = ppl(data)
        print(res)
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]['output'] == 'cot xxxx \\boxed{0.5}'
        assert res[0]['instruction'] == 'answer for 1 divided by 2?'
        assert res[0]['input'] == '0.5'
