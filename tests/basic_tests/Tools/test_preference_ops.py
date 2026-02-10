import os
import shutil
from lazyllm import config
from lazyllm.tools.data.operators import preference_ops
import pytest  # noqa: F401

class TestPreferenceOperators:

    def setup_method(self):
        self.root_dir = './test_preference_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_intent_extractor(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"intent": "book a hotel"}'

        op = preference_ops.IntentExtractor(
            model=MockModel(),
            input_key='content',
            output_key='intent',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'content': 'I want to stay at a hotel in Beijing.'}
        res = op([data])
        assert res[0]['intent']['intent'] == 'book a hotel'

    def test_preference_response_generator(self):
        class MockModel:
            def __call__(self, x, **kwargs):
                return f'Response to {x}'

        op = preference_ops.PreferenceResponseGenerator(
            model=MockModel(),
            n=2,
            input_key='intent',
            output_key='responses',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'intent': 'book a hotel'}
        res = op([data])
        assert len(res[0]['responses']) == 2
        assert res[0]['responses'][0] == 'Response to book a hotel'

    def test_response_evaluator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"total_score": 8}'

        op = preference_ops.ResponseEvaluator(
            model=MockModel(),
            input_key='intent',
            response_key='responses',
            output_key='evaluation',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'intent': 'book a hotel', 'responses': ['res1', 'res2']}
        res = op([data])
        assert res[0]['evaluation'] == [8, 8]

    def test_preference_pair_constructor(self):
        op = preference_ops.PreferencePairConstructor(
            strategy='max_min',
            instruction_key='intent',
            response_key='responses',
            score_key='evaluation',
            _concurrency_mode='single',
            _save_data=False
        )

        # Test valid pair
        data = {
            'intent': 'book a hotel',
            'responses': ['good response', 'bad response'],
            'evaluation': [9, 3]
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['chosen'] == 'good response'
        assert res[0]['rejected'] == 'bad response'

        # Test no valid pair (scores equal)
        data2 = {
            'intent': 'book a hotel',
            'responses': ['res1', 'res2'],
            'evaluation': [5, 5]
        }
        res2 = op([data2])
        assert res2 == []
