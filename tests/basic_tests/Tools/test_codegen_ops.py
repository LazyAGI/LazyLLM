import os
import shutil
import pytest  # noqa: F401
import json
from lazyllm import config
from lazyllm.tools.data.operators import codegen_ops

class MockModel:
    def __init__(self, return_value):
        self.return_value = return_value
        self.sys_prompt = None
        self._formatter = None

    def share(self):
        return self

    def prompt(self, template):
        self.sys_prompt = template
        return self

    def formatter(self, formatter_obj):
        self._formatter = formatter_obj
        return self

    def __call__(self, *args, **kwargs):
        if self._formatter:
            try:
                return json.loads(self.return_value)
            except Exception:
                return self.return_value
        return self.return_value

class TestCodeGenOps:

    def setup_method(self):
        self.root_dir = './test_codegen_ops'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_code_enhancement_instruction_generator(self):
        mock_response = 'Enhanced instruction\n```python\ndef solution():\n    pass\n```'
        model = MockModel(mock_response)
        op = codegen_ops.CodeEnhancementInstructionGenerator(model=model)
        data = {'messages': [{'role': 'user', 'content': 'test'}]}
        result = op.forward(data)
        assert result['generated_instruction'] == mock_response

    def test_code_instruction_to_code_generator(self):
        mock_response = "```python\nprint('hello')\n```"
        model = MockModel(mock_response)
        op = codegen_ops.CodeInstructionToCodeGenerator(model=model)
        data = {'instruction': 'print hello'}
        result = op.forward(data)
        assert "print('hello')" in result['generated_code']

    def test_code_quality_sample_evaluator(self):
        mock_data = {
            'score': 8,
            'feedback': 'Good code.'
        }
        mock_response = json.dumps(mock_data)
        model = MockModel(mock_response)
        op = codegen_ops.CodeQualitySampleEvaluator(model=model)
        data = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result = op.forward(data)
        assert result['quality_score'] == 8
        assert result['quality_feedback'] == 'Good code.'

    def test_code_quality_score_filter(self):
        mock_data_high = {
            'score': 8,
            'feedback': 'Good code.'
        }
        mock_response_high = json.dumps(mock_data_high)
        model = MockModel(mock_response_high)
        op = codegen_ops.CodeQualityScoreFilter(model=model, min_score=7)
        data = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result = op.forward(data)

        res_obj = result[0] if isinstance(result, list) else result
        assert res_obj['quality_score_filter_label'] == 1

        mock_data_low = {
            'score': 5,
            'feedback': 'Bad code.'
        }
        mock_response_low = json.dumps(mock_data_low)
        model_low = MockModel(mock_response_low)
        op_low = codegen_ops.CodeQualityScoreFilter(model=model_low, min_score=7)
        data_low = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result_low = op_low.forward(data_low)
        assert result_low == []
