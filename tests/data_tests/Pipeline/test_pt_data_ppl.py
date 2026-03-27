import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.pt_data_ppl import build_long_context_pipeline


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response
        self.last_prompt = None

    def __call__(self, string, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        self.last_prompt = prompt
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

    def test_long_context_pipeline(self):
        mock_llm = MockModel('Expanded payload.')
        custom_prompt = 'Please expand with richer background.'
        ppl = build_long_context_pipeline(
            llm=mock_llm,
            context_key='ctx',
            question_key='q',
            answer_key='a',
            expanded_key='expanded_ctx',
            long_context_key='long_ctx',
            expansion_prompt=custom_prompt,
            num_distractors=2,
            passage_sep=' || ',
            seed=42,
            expansion_concurrency_mode='single',
            reconstruction_concurrency_mode='single',
        )
        data = [
            {'ctx': f'ctx {i}', 'q': f'Q{i}', 'a': f'A{i}'}
            for i in range(4)
        ]
        res = ppl(data)
        assert len(res) == 4
        assert mock_llm.last_prompt == custom_prompt
        assert {item['q'] for item in res} == {'Q0', 'Q1', 'Q2', 'Q3'}
        for item in res:
            assert set(item.keys()) == {'context', 'long_ctx', 'q', 'a'}
            assert item['context'] == 'Expanded payload.'
            passages = item['long_ctx'].split(' || ')
            assert len(passages) == 3
