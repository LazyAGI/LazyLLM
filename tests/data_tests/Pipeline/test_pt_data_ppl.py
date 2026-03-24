import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.pt_data_ppl import build_long_context_pipeline


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def share(self):
        return self


class TestLongContextPtPipeline:

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

    def test_long_context_pt_pipeline(self):
        mock_llm = MockModel('Expanded context about the topic with more details.')
        ppl = build_long_context_pipeline(mock_llm, num_distractors=2, seed=42)
        data = [
            {'context': f'Short context {i}.', 'question': f'Q{i}', 'answer': f'A{i}'}
            for i in range(4)
        ]
        res = ppl(data)
        assert len(res) == 4
        for item in res:
            assert 'long_context' in item
            assert 'question' in item
            assert 'answer' in item
            passages = item['long_context'].split('\n\n')
            assert len(passages) == 3
        questions = {r['question'] for r in res}
        assert questions == {'Q0', 'Q1', 'Q2', 'Q3'}

    def test_pipeline_filters_empty_expansion(self):
        mock_llm = MockModel(None)
        ppl = build_long_context_pipeline(mock_llm, num_distractors=2, seed=0)
        data = [
            {'context': f'ctx {i}', 'question': f'Q{i}', 'answer': f'A{i}'}
            for i in range(3)
        ]
        res = ppl(data)
        assert res == []

    def test_pipeline_single_item(self):
        mock_llm = MockModel('Expanded single context.')
        ppl = build_long_context_pipeline(mock_llm, num_distractors=3, seed=1)
        data = [{'context': 'Only one.', 'question': 'Q?', 'answer': 'A.'}]
        res = ppl(data)
        assert len(res) == 1
        assert res[0]['long_context'] == 'Expanded single context.'
        assert res[0]['question'] == 'Q?'
        assert res[0]['answer'] == 'A.'
