import pytest
from unittest.mock import patch
from lazyllm.tools.data.operators.codegen_ops import (
    CodeEnhancementInstructionGenerator,
    CodeInstructionToCodeGenerator,
    CodeQualitySampleEvaluator,
    CodeQualityScoreFilter,
    CodeSandboxSampleEvaluator
)

class MockModel:
    def __init__(self, response):
        self.response = response

    def prompt(self, template):
        return lambda x: self.response

class TestCodeGenOps:
    def test_code_enhancement_instruction_generator(self):
        mock_response = 'Enhanced instruction\n```python\ndef solution():\n    pass\n```'
        model = MockModel(mock_response)
        op = CodeEnhancementInstructionGenerator(model=model)
        data = {'messages': [{'role': 'user', 'content': 'test'}]}
        # Test forward directly
        result = op.forward(data)
        assert result['generated_instruction'] == mock_response

    def test_code_instruction_to_code_generator(self):
        mock_response = "```python\nprint('hello')\n```"
        model = MockModel(mock_response)
        op = CodeInstructionToCodeGenerator(model=model)
        data = {'instruction': 'print hello'}
        result = op.forward(data)
        assert result['generated_code'] == "print('hello')"

    def test_code_quality_sample_evaluator(self):
        mock_response = 'Score: 8\nFeedback: Good code.'
        model = MockModel(mock_response)
        op = CodeQualitySampleEvaluator(model=model)
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
        op = CodeQualityScoreFilter(model=model, min_score=7)
        data = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result = op.forward(data)
        assert result['quality_score_filter_label'] == 1

        # Test filtered out
        mock_response_low = 'Score: 5\nFeedback: Bad code.'
        model_low = MockModel(mock_response_low)
        op_low = CodeQualityScoreFilter(model=model_low, min_score=7)
        data_low = {
            'generated_instruction': 'print hello',
            'generated_code': "print('hello')"
        }
        result_low = op_low.forward(data_low)
        assert result_low == []

    def test_code_sandbox_sample_evaluator(self):
        # Mock PythonExecutor
        with patch('lazyllm.tools.data.operators.codegen_ops.PythonExecutor') as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.batch_apply.return_value = [
                ({'text': 'hello\n'}, 'Done'),
                ('Error message', 'Error: runtime error')
            ]

            # Ensure the module uses our mock even if it was None initially
            from lazyllm.tools.data.operators import codegen_ops
            original_executor = codegen_ops.PythonExecutor
            codegen_ops.PythonExecutor = MockExecutor

            try:
                op = CodeSandboxSampleEvaluator()
                inputs = [
                    {'generated_code': "print('hello')"},
                    {'generated_code': 'raise Exception()'}
                ]
                # Test calling the operator (which uses forward_batch_input)
                op(inputs)

                # Note: LazyLLMDataBase.__call__ might return a path to a jsonl file
                # if save_data is True (default).
                # For testing, we can check the behavior of forward_batch_input directly if needed,
                # or mock the store.

                # Let's test forward_batch_input directly to avoid file I/O complexity in basic test
                direct_results = op.forward_batch_input(inputs)

                assert len(direct_results) == 2
                assert direct_results[0]['sandbox_status'] == 'PASS'
                assert direct_results[0]['sandbox_log'] == 'hello\n'
                assert direct_results[1]['sandbox_status'] == 'FAIL'
                assert direct_results[1]['sandbox_log'] == 'Error: runtime error'
            finally:
                codegen_ops.PythonExecutor = original_executor

    def test_code_sandbox_missing_executor_error(self):
        from lazyllm.tools.data.operators import codegen_ops
        original_executor = codegen_ops.PythonExecutor
        codegen_ops.PythonExecutor = None
        try:
            with pytest.raises(ImportError, match='PythonExecutor not found'):
                CodeSandboxSampleEvaluator()
        finally:
            codegen_ops.PythonExecutor = original_executor

    def test_code_sandbox_real_executor(self):
        # Only run if PythonExecutor is available
        from lazyllm.tools.data.operators.codegen_ops import PythonExecutor
        if PythonExecutor is None:
            pytest.skip('PythonExecutor not available')

        op = CodeSandboxSampleEvaluator()
        inputs = [
            {'generated_code': "print('hello world')"},
            {'generated_code': "import time; time.sleep(0.1); print('delayed')"},
            {'generated_code': "raise ValueError('error')"}
        ]

        results = op.forward_batch_input(inputs)

        assert len(results) == 3
        assert results[0]['sandbox_status'] == 'PASS'
        assert 'hello world' in results[0]['sandbox_log']

        assert results[1]['sandbox_status'] == 'PASS'
        assert 'delayed' in results[1]['sandbox_log']

        assert results[2]['sandbox_status'] == 'FAIL'
        assert 'ValueError: error' in results[2]['sandbox_log']
