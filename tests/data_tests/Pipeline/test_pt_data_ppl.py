import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.pt_data_ppl import build_structured_data_pipeline


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self


class TestStructuredDataPipeline:

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

    def test_structured_data_pipeline(self):
        mock_response = {
            'triples': [
                {'subject': 'Alice', 'predicate': 'works at', 'object': 'Company X'},
            ],
        }
        llm = MockModel(mock_response)

        ppl = build_structured_data_pipeline(
            llm,
            input_key='text',
            output_key='parsed',
        )

        data = [{'text': 'Alice works at Company X. Bob is a friend of Alice.'}]
        res = ppl(data)

        assert isinstance(res, list)
        assert len(res) == 1
        assert 'parsed' in res[0]
        assert res[0]['parsed'] == mock_response
        assert res[0]['parsed']['triples'][0]['subject'] == 'Alice'
