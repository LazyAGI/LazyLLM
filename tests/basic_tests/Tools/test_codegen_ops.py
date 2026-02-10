import os
import shutil
import pytest  # noqa: F401
from lazyllm import config
from lazyllm.tools.data.operators import codegen_ops

class MockModel:
    def __init__(self, response):
        self.response = response

    def prompt(self, template):
        return lambda x: self.response

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
        assert result['generated_code'] == "print('hello')"

    def test_code_quality_sample_evaluator(self):
        mock_response = 'Score: 8\nFeedback: Good code.'
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
        mock_response = 'Score: 8\nFeedback: Good code.'
        model = MockModel(mock_response)
        op = codegen_ops.CodeQualityScoreFilter(model=model, min_score=7)
        data = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result = op.forward(data)
        assert result['quality_score_filter_label'] == 1

        # Test filtered out
        mock_response_low = 'Score: 5\nFeedback: Bad code.'
        model_low = MockModel(mock_response_low)
        op_low = codegen_ops.CodeQualityScoreFilter(model=model_low, min_score=7)
        data_low = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result_low = op_low.forward(data_low)
        assert result_low == []
