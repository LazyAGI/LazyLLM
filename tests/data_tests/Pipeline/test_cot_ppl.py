import os
import shutil
import tempfile
from lazyllm import config
from lazyllm.tools.data.pipelines.cot_pipelines import build_cot_pipeline


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


class TestCoTQAPipeline:

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

    def test_cot_pipeline(self):

        excepted_output = 'cot #### B'

        model = MockModel(excepted_output)

        ppl = build_cot_pipeline(
            input_key='question',
            reference_key='reference',
            cot_key='cot_answer',
            extracted_key='llm_extracted',
            verify_key='is_equal',
            model=model,
            use_self_consistency=True,
            num_samples=3,
            enable_verify=True,
            hash_answer=True,
            boxed_answer=False
        )

        data = {
            'task': 'date_understanding',
            'question': 'Today is xxx:(A) xx(B) xx\n(C) xx(D) x(E) x(F) xx',
            'reference': 'B'
        }

        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]['output'] == 'cot #### B'
        assert res[0]['instruction'] == 'Today is xxx:(A) xx(B) xx\n(C) xx(D) x(E) x(F) xx'
        assert res[0]['input'] == 'B'
