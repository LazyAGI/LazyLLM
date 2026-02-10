from typing import List, Tuple, Dict
from lazyllm import LOG
from ..base_data import data_register
from .python_executor import PythonExecutor
#————后续修改————

CodeGenOps = data_register.new_group('codegen_ops')

class CodeSandboxSampleEvaluator(CodeGenOps):

    def __init__(self, language: str = 'python', timeout_length: int = 15, use_process_isolation: bool = True,
                 input_code_key: str = 'generated_code', output_status_key: str = 'sandbox_status',
                 output_log_key: str = 'sandbox_log', **kwargs):
        super().__init__(**kwargs)
        self.language = language
        self.timeout_length = timeout_length
        self.use_process_isolation = use_process_isolation
        self.input_code_key = input_code_key
        self.output_status_key = output_status_key
        self.output_log_key = output_log_key

        LOG.info(f'Initializing {self.__class__.__name__}...')

        if PythonExecutor is None:
            raise ImportError("PythonExecutor not found. Please ensure 'python_executor.py' is in the same directory.")

        # Initialize the PythonExecutor here. It will be reused for all code snippets.
        self.executor = PythonExecutor(
            get_answer_from_stdout=True,  # Capture print statements as primary output
            timeout_length=timeout_length,
            use_process_isolation=use_process_isolation
        )
        self.score_name = 'SandboxValidationScore'
        LOG.info(f'{self.__class__.__name__} initialized.')


    def _execute_code_batch(self, code_list: List[str]) -> List[Tuple[str, str]]:
        results_with_reports = self.executor.batch_apply(code_list, messages=[])

        processed_results = []
        for (result, report) in results_with_reports:
            if report == 'Done':
                status = 'PASS'
                log = result.get('text', '') if isinstance(result, dict) else result
            else:
                status = 'FAIL'
                log = report

            processed_results.append((status, log))

        return processed_results

    def forward_batch_input(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        LOG.info(f'Evaluating {self.score_name}...')

        if not inputs:
            return []

        code_list = []
        for i, item in enumerate(inputs):
            if self.input_code_key not in item:
                raise ValueError(f"Missing required key '{self.input_code_key}' in item at index {i}")
            code_list.append(item[self.input_code_key])

        execution_results = self._execute_code_batch(code_list)

        results = []
        for item, (status, log) in zip(inputs, execution_results):
            new_item = item.copy()
            # Check for conflicts
            if self.output_status_key in new_item:
                raise ValueError(f"The key '{self.output_status_key}' already exists and would be overwritten.")
            if self.output_log_key in new_item:
                raise ValueError(f"The key '{self.output_log_key}' already exists and would be overwritten.")

            new_item[self.output_status_key] = status
            new_item[self.output_log_key] = log
            results.append(new_item)

        LOG.info('Evaluation complete!')
        return results
