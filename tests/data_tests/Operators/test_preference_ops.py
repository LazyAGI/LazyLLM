import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.operators import preference_ops

class TestPreferenceOperators:
    class MockModel:
        def __init__(self, return_val=None):
            self.return_val = return_val

        def share(self): return self
        def prompt(self, system): return self
        def formatter(self, fmt): return self

        def __call__(self, x, **kwargs):
            if isinstance(x, list):
                return [self.return_val] * len(x)
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

    def test_intent_extractor(self):
        mockmodle = self.MockModel(return_val={'intent': 'book a hotel'})

        op = preference_ops.IntentExtractor(
            model=mockmodle,
            input_key='content',
            output_key='intent',
            _save_data=False
        )
        data = {'content': 'I want to stay at a hotel in Beijing.'}
        res = op([data])
        print('test_intent_extractor result:', res)
        assert 'intent' in res[0]['intent']

    def test_preference_response_generator(self):
        mockmodle = self.MockModel(return_val='I can help you book a hotel in Beijing.')

        op = preference_ops.PreferenceResponseGenerator(
            model=mockmodle,
            n=2,
            input_key='intent',
            output_key='responses',
            _save_data=False
        )
        data = {'intent': {'intent': 'book a hotel'}}
        res = op([data])
        print('test_preference_response_generator result:', res)
        assert len(res[0]['responses']) == 2
        assert isinstance(res[0]['responses'][0], str)

    def test_response_evaluator(self):
        mockmodle = self.MockModel(return_val={'total_score': 8})

        op = preference_ops.ResponseEvaluator(
            model=mockmodle,
            input_key='intent',
            response_key='responses',
            output_key='evaluation',
            _save_data=False
        )
        data = {
            'intent': {'intent': 'book a hotel'},
            'responses': ['I can help you book a hotel in Beijing.', 'Here are some hotels for you.']
        }
        res = op([data])
        print('test_response_evaluator result:', res)
        assert len(res[0]['evaluation']) == 2
        assert all(isinstance(score, (int, float)) for score in res[0]['evaluation'])

    def test_preference_pair_constructor(self):
        op = preference_ops.PreferencePairConstructor(
            strategy='max_min',
            instruction_key='intent',
            response_key='responses',
            score_key='evaluation',
            _save_data=False
        )

        data = {
            'intent': 'book a hotel',
            'responses': ['good response', 'bad response'],
            'evaluation': [9, 3]
        }
        res = op([data])
        print('test_preference_pair_constructor result:', res)
        assert len(res) == 1
        assert res[0]['chosen'] == 'good response'
        assert res[0]['rejected'] == 'bad response'
