import os
import shutil
from lazyllm import config
from lazyllm.tools.data.operators import tool_use_ops
import pytest  # noqa: F401

class TestToolUseOperators:

    def setup_method(self):
        self.root_dir = './test_tool_use_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_scenario_extractor(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"scene": "ordering pizza", "domain": "food"}'

        op = tool_use_ops.ScenarioExtractor(
            model=MockModel(),
            input_key='content',
            output_key='scenario',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'content': 'I want to order a pepperoni pizza.'}
        res = op([data])
        assert res[0]['scenario']['scene'] == 'ordering pizza'
        assert res[0]['scenario']['domain'] == 'food'

    def test_scenario_expander(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"scenarios": [{"scene": "booking a flight"}]}'

        op = tool_use_ops.ScenarioExpander(
            model=MockModel(),
            input_key='scenario',
            output_key='expanded',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'scenario': {'scene': 'travel'}}
        res = op([data])
        assert len(res[0]['expanded']) == 1
        assert res[0]['expanded'][0]['scene'] == 'booking a flight'

    def test_atom_task_generator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"tasks": [{"task": "check weather"}]}'

        op = tool_use_ops.AtomTaskGenerator(
            model=MockModel(),
            input_key='scenario',
            output_key='tasks',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'scenario': 'outdoor activity'}
        res = op([data])
        assert res[0]['tasks'][0]['task'] == 'check weather'

    def test_sequential_task_generator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"items": [{"task": "t1", "next_task": "t2", "composed_task": "t1 then t2"}]}'

        op = tool_use_ops.SequentialTaskGenerator(
            model=MockModel(),
            input_key='atomic_tasks',
            output_key='seq_tasks',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'atomic_tasks': ['t1', 't2']}
        res = op([data])
        assert res[0]['seq_tasks'][0]['task'] == 't1'
        assert res[0]['seq_tasks'][0]['next_task'] == 't2'
        assert res[0]['seq_tasks'][0]['composed_task'] == 't1 then t2'

    def test_para_seq_task_generator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"parallel_tasks": ["task1"], "sequential_tasks": ["task2"], "hybrid_tasks": []}'

        op = tool_use_ops.ParaSeqTaskGenerator(
            model=MockModel(),
            input_key='atomic_tasks',
            output_key='complex',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'atomic_tasks': ['task1', 'task2']}
        res = op([data])
        assert 'task1' in res[0]['complex']['parallel_tasks']
        assert 'task2' in res[0]['complex']['sequential_tasks']

    def test_composition_task_filter(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"items": [{"composed_task": "valid_task", "is_valid": true, "reason": "ok"}]}'

        op = tool_use_ops.CompositionTaskFilter(
            model=MockModel(),
            composition_key='complex',
            output_key='filtered',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'complex': ['valid_task', 'invalid_task']}
        res = op([data])
        assert 'valid_task' in res[0]['filtered']
        assert 'invalid_task' not in res[0]['filtered']

    def test_function_generator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"functions": [{"name": "get_weather", "description": "returns weather"}]}'

        op = tool_use_ops.FunctionGenerator(
            model=MockModel(),
            task_key='task',
            output_key='funcs',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'task': 'check weather'}
        res = op([data])
        assert res[0]['funcs'][0]['name'] == 'get_weather'

    def test_multi_turn_conversation_generator(self):
        class MockModel:
            def prompt(self, system):
                return lambda x: '{"messages": [{"role": "user", "content": "hello"}]}'

        op = tool_use_ops.MultiTurnConversationGenerator(
            model=MockModel(),
            task_key='task',
            output_key='conv',
            _concurrency_mode='single',
            _save_data=False
        )
        data = {'task': 'greet'}
        res = op([data])
        assert res[0]['conv']['messages'][0]['role'] == 'user'
