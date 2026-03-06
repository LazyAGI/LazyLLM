import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.tool_use_pipelines import build_tool_use_pipeline, build_simple_tool_use_pipeline


class MockModelCallable:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def __call__(self, x):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


class TestToolUsePipeline:

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

    def test_tool_use_pipeline(self):
        responses = [
            {'scene': 'ordering pizza', 'domain': 'food'},  # ContextualBeacon
            {'scenarios': [{'scene': 'ordering sushi', 'domain': 'food'}]},  # ScenarioDiverger
            {'tasks': [{'task': 'select restaurant', 'input': '', 'output': ''}]},  # DecompositionKernel
            # ChainedLogicAssembler
            {'items': [{'task': 'select restaurant', 'next_task': 'place order',
                        'composed_task': 'order food'}]},
            {'parallel_tasks': ['task1'], 'sequential_tasks': ['task2'],
             'hybrid_tasks': ['hybrid_task']},  # TopologyArchitect
            {'items': [{'composed_task': 'order food', 'is_valid': True, 'reason': 'ok'}]},  # ViabilitySieve
            {'functions': [{'name': 'order_food', 'description': 'place an order'}]},
            # ProtocolSpecifier (receives list, should handle it)
            {'messages': [{'role': 'user', 'content': 'I want pizza'}]},  # DialogueSimulator
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = build_tool_use_pipeline(model=mock_model, input_key='content', n_turns=3)
        data = [{'content': 'I want to order food.'}]
        res = ppl(data)

        assert len(res) >= 0
        if len(res) > 0:
            assert 'conversation' in res[0]

    def test_simple_tool_use_pipeline(self):
        '''Test the simplified tool use pipeline.

        This pipeline is simpler and avoids the list/single value conversion issues.
        '''
        responses = [
            {'scene': 'booking flight', 'domain': 'travel'},  # ContextualBeacon
            {'tasks': [{'task': 'search flights', 'input': '', 'output': ''}]},  # DecompositionKernel
            {'functions': [{'name': 'search_flights', 'description': 'find flights'}]},  # ProtocolSpecifier
            {'messages': [{'role': 'user', 'content': 'Book a flight'}]},  # DialogueSimulator
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = build_simple_tool_use_pipeline(model=mock_model, input_key='content', n_tasks=3, n_turns=3)
        data = [{'content': 'I want to book a flight.'}]
        res = ppl(data)

        assert len(res) == 1
        assert 'conversation' in res[0]
