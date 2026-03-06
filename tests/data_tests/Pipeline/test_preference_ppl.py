import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.preference_pipelines import build_preference_pipeline


class MockModelCallable:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def __call__(self, x):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


class TestPreferencePipeline:

    class MockModel:
        def __init__(self, return_val=None):
            self.return_val = return_val

        def share(self): return self
        def prompt(self, system): return self
        def formatter(self, fmt): return self

        def __call__(self, x, **kwargs):
            if callable(self.return_val):
                return self.return_val(x)
            return self.return_val

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

    def test_preference_pipeline(self):
        responses = [
            {'intent': 'book a hotel'},  # IntentExtractor
            'I can help you book a hotel in Beijing.',  # PreferenceResponseGenerator #1
            'Here are some hotels for you in Beijing.',  # PreferenceResponseGenerator #2
            {'total_score': 8},  # ResponseEvaluator #1
            {'total_score': 5},  # ResponseEvaluator #2
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = build_preference_pipeline(
            model=mock_model,
            input_key='content',
            n=2,
            strategy='max_min',
            threshold=0.5
        )
        data = [{'content': 'I want to stay at a hotel in Beijing.'}]
        res = ppl(data)

        assert len(res) == 1
        assert 'chosen' in res[0]
        assert 'rejected' in res[0]
        assert 'instruction' in res[0]
