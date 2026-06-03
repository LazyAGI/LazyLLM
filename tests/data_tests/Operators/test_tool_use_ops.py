import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.operators import tool_use_ops

class TestToolUseOperators:

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

    def test_scenario_extractor(self):
        mock_model = self.MockModel(return_val={'scene': 'ordering pizza', 'domain': 'food'})

        op = tool_use_ops.ContextualBeacon(
            model=mock_model,
            input_key='content',
            output_key='scenario',
            _save_data=False
        )
        data = {'content': 'I want to order a pepperoni pizza.'}
        res = op([data])
        assert res[0]['scenario']['scene'] == 'ordering pizza'
        assert res[0]['scenario']['domain'] == 'food'

    def test_scenario_expander(self):
        mock_model = self.MockModel(return_val={'scenarios': [{'scene': 'booking a flight'}]})

        op = tool_use_ops.ScenarioDiverger(
            model=mock_model,
            input_key='scenario',
            output_key='expanded',
            _save_data=False
        )
        data = {'scenario': {'scene': 'travel'}}
        res = op([data])
        assert len(res[0]['expanded']) == 1
        assert res[0]['expanded'][0]['scene'] == 'booking a flight'

    def test_atom_task_generator(self):
        mock_model = self.MockModel(return_val={'tasks': [{'task': 'check weather'}]})

        op = tool_use_ops.DecompositionKernel(
            model=mock_model,
            input_key='scenario',
            output_key='tasks',
            _save_data=False
        )
        data = {'scenario': 'outdoor activity'}
        res = op([data])
        assert res[0]['tasks'][0]['task'] == 'check weather'

    def test_sequential_task_generator(self):
        mock_model = self.MockModel(return_val={'items': [{'task': 't1', 'next_task': 't2',
                                                          'composed_task': 't1 then t2'}]})

        op = tool_use_ops.ChainedLogicAssembler(
            model=mock_model,
            input_key='atomic_tasks',
            output_key='seq_tasks',
            _save_data=False
        )
        data = {'atomic_tasks': ['t1', 't2']}
        res = op([data])
        assert res[0]['seq_tasks'][0]['task'] == 't1'
        assert res[0]['seq_tasks'][0]['next_task'] == 't2'

    def test_para_seq_task_generator(self):
        mock_model = self.MockModel(return_val={'parallel_tasks': ['task1'], 'sequential_tasks': ['task2'],
                                                'hybrid_tasks': []})

        op = tool_use_ops.TopologyArchitect(
            model=mock_model,
            input_key='atomic_tasks',
            output_key='complex',
            _save_data=False
        )
        data = {'atomic_tasks': ['task1', 'task2']}
        res = op([data])
        assert 'task1' in res[0]['complex']['parallel_tasks']
        assert 'task2' in res[0]['complex']['sequential_tasks']

    def test_composition_task_filter(self):
        mock_model = self.MockModel(return_val={'items': [{'composed_task': 'valid_task',
                                                          'is_valid': True, 'reason': 'ok'}]})

        op = tool_use_ops.ViabilitySieve(
            model=mock_model,
            input_composition_key='complex',
            output_key='filtered',
            _save_data=False
        )
        data = {'complex': ['valid_task', 'invalid_task']}
        res = op([data])
        assert 'valid_task' in res[0]['filtered']

    def test_function_generator(self):
        mock_model = self.MockModel(return_val={'functions': [{'name': 'get_weather',
                                                              'description': 'returns weather'}]})

        op = tool_use_ops.ProtocolSpecifier(
            model=mock_model,
            input_composition_key='task',
            output_key='funcs',
            _save_data=False
        )
        data = {'task': 'check weather'}
        res = op([data])
        assert res[0]['funcs'][0]['name'] == 'get_weather'

    def test_multi_turn_conversation_generator(self):
        mock_model = self.MockModel(return_val={'messages': [{'role': 'user', 'content': 'hello'}]})

        op = tool_use_ops.DialogueSimulator(
            model=mock_model,
            input_composition_key='task',
            output_key='conv',
            _save_data=False
        )
        data = {'task': 'greet'}
        res = op([data])
        assert res[0]['conv']['messages'][0]['role'] == 'user'

    def test_quality_filter_pass(self):
        mock_model = self.MockModel(return_val={'completeness': 4, 'feasibility': 4})

        op = tool_use_ops.ToolUseQualityFilter(
            model=mock_model,
            min_completeness_score=3,
            min_feasibility_score=3,
            _save_data=False
        )
        data = {'instruction': 'test instruction', 'input': 'test input', 'output': 'test output'}
        res = op([data])
        assert len(res) == 1
        assert res[0]['instruction'] == 'test instruction'

    def test_quality_filter_fail_completeness(self):
        mock_model = self.MockModel(return_val={'completeness': 2, 'feasibility': 4})

        op = tool_use_ops.ToolUseQualityFilter(
            model=mock_model,
            min_completeness_score=3,
            min_feasibility_score=3,
            _save_data=False
        )
        data = {'instruction': 'test instruction', 'input': 'test input', 'output': 'test output'}
        res = op([data])
        assert len(res) == 0

    def test_quality_filter_fail_feasibility(self):
        mock_model = self.MockModel(return_val={'completeness': 5, 'feasibility': 2})

        op = tool_use_ops.ToolUseQualityFilter(
            model=mock_model,
            min_completeness_score=3,
            min_feasibility_score=3,
            _save_data=False
        )
        data = {'instruction': 'test instruction', 'input': 'test input', 'output': 'test output'}
        res = op([data])
        assert len(res) == 0

    def test_quality_filter_empty_output(self):
        mock_model = self.MockModel(return_val={'completeness': 5, 'feasibility': 5})

        op = tool_use_ops.ToolUseQualityFilter(
            model=mock_model,
            min_completeness_score=3,
            min_feasibility_score=3,
            _save_data=False
        )
        data = {'instruction': 'test instruction', 'input': 'test input', 'output': ''}
        res = op([data])
        assert len(res) == 0
