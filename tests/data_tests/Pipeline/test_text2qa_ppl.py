import os
import shutil
import tempfile
from lazyllm import config
from lazyllm.tools.data.pipelines.text_pipelines import build_text2qa_pipeline

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

class TestTextPipeline:

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

    def test_text2qa_pipeline(self):

        model = MockModel({
            'chunk': '今天是晴天！',
            'instruction': '今天的天气怎么样？',
            'output': '今天是晴天！',
            'score': 1
        })

        ppl = build_text2qa_pipeline(
            model=model,
            text_key='text',
            chunk_key='chunk',
            instruction_key='instruction',
            output_key='output',
            score_prompt=None,
            tokenizer=None,
            chunk_size=200,
            tokenize=False,
            qa_prompt=None,
            threshold=1
        )

        data = [{'text': '今天是晴天！'}]

        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 1
