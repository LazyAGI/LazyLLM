from ..base_data import data_register
import re
from typing import Tuple, List, Dict
from lazyllm import LOG

try:
    from .python_executor import PythonExecutor
except ImportError:
    PythonExecutor = None


CodeGenOps = data_register.new_group('codegen_ops')


def _extract_human_instruction(messages):
    if messages is None:
        return ''
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        content = messages.get('content', '')
        return content if isinstance(content, str) else str(content)
    if isinstance(messages, list):
        parts = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = (item.get('role') or '').lower()
            if role in {'human', 'user'}:
                content = item.get('content', '')
                return content if isinstance(content, str) else str(content)
            content = item.get('content', '')
            if content:
                parts.append(content if isinstance(content, str) else str(content))
        return '\n'.join(parts)
    return str(messages)


def _parse_code(response):
    if response is None:
        return ''
    if not isinstance(response, str):
        response = str(response)
    code_block_match = re.search(r'```(?:python\n)?(.*?)```', response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    return response.strip()


def _parse_score_and_feedback(response) -> Tuple[int, str]:
    if response is None:
        return 0, 'No response from LLM'
    if not isinstance(response, str):
        response = str(response)
    try:
        score_match = re.search(r'Score:\s*(\d+)', response)
        feedback_match = re.search(r'Feedback:\s*(.*)', response, re.DOTALL)
        score = int(score_match.group(1)) if score_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else 'No feedback provided.'
        return score, feedback
    except (AttributeError, ValueError, IndexError):
        return 0, 'Failed to parse LLM evaluation output.'


class CodeEnhancementInstructionGenerator(CodeGenOps):
    def __init__(self, model=None, prompt_template=None, input_key='messages', output_key='generated_instruction',
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = prompt_template or (
            'You are a code instruction standardization assistant.\n'
            'Rewrite the given instruction into a consistent format for Python code generation tasks.\n'
            'Output must be English and contain exactly two parts:\n'
            '1) A single concise instruction sentence in English.\n'
            '2) A Python code block in Markdown with a complete function skeleton.\n'
            'Do not add explanations, do not add extra sections.\n'
            'Example output format:\n'
            'Write a Python function that ...\n'
            '```python\n'
            'def solution(...):\n'
            '    \"\"\"...\"\"\"\n'
            '    ...\n'
            '```\n'
        )

    @staticmethod
    def get_desc(lang: str = 'en'):
        if lang == 'zh':
            return (
                '该算子用于增强人类指令，将不同输出格式的任务统一为生成完整函数。\n\n'
                '输入参数：\n'
                "- input_key: 包含原始指令消息的字段名 (默认: 'messages')\n"
                '输出参数：\n'
                "- output_key: 用于存储生成指令的字段名 (默认: 'generated_instruction')\n"
            )
        return (
            'This operator enhances human instructions by unifying tasks with different output formats '
            'into complete function generation tasks.\n\n'
            'Input Parameters:\n'
            "- input_key: Field name containing the original instruction messages (default: 'messages')\n"
            'Output Parameters:\n'
            "- output_key: Field name to store the enhanced instruction (default: 'generated_instruction')\n"
        )

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.input_key not in data:
            raise ValueError(f'Missing required key: {self.input_key}')
        if self.output_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_key}')
        raw_instruction = _extract_human_instruction(data.get(self.input_key))
        response = self.model.prompt(self.system_prompt)(raw_instruction)
        data[self.output_key] = response.strip() if isinstance(response, str) else response
        return data


class CodeInstructionToCodeGenerator(CodeGenOps):
    def __init__(self, model=None, prompt_template=None, input_key='instruction', output_key='generated_code', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = prompt_template or (
            'You are a senior Python engineer.\n'
            'Given a natural language instruction, generate the corresponding Python code.\n'
            'Return only the code. If you include a Markdown code block, use ```python ... ```.\n'
        )

    @staticmethod
    def get_desc(lang: str = 'en'):
        if lang == 'zh':
            return (
                '该算子根据给定的人类指令生成相应的代码片段。\n\n'
                '输入参数：\n'
                "- input_key: 包含人类指令的字段名 (默认: 'instruction')\n"
                '输出参数：\n'
                "- output_key: 用于存储生成代码的字段名 (默认: 'generated_code')\n"
            )
        return (
            'This operator generates a code snippet based on a given natural language instruction.\n\n'
            'Input Parameters:\n'
            "- input_key: Field name containing the human instruction (default: 'instruction')\n"
            'Output Parameters:\n'
            "- output_key: Field name to store the generated code (default: 'generated_code')\n"
        )

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.input_key not in data:
            raise ValueError(f'Missing required key: {self.input_key}')
        if self.output_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_key}')
        instruction = data.get(self.input_key, '')
        response = self.model.prompt(self.system_prompt)(instruction)
        data[self.output_key] = _parse_code(response)
        return data


class CodeQualitySampleEvaluator(CodeGenOps):
    def __init__(
        self,
        model=None,
        prompt_template=None,
        input_instruction_key='generated_instruction',
        input_code_key='generated_code',
        output_score_key='quality_score',
        output_feedback_key='quality_feedback',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.input_instruction_key = input_instruction_key
        self.input_code_key = input_code_key
        self.output_score_key = output_score_key
        self.output_feedback_key = output_feedback_key
        self.system_prompt = prompt_template or (
            'You are an automated code reviewer.\n'
            'Evaluate the generated Python code against the given instruction.\n'
            'Return the result strictly in the following format:\n'
            'Score: <0-10>\n'
            'Feedback: <your feedback>\n'
        )

    @staticmethod
    def get_desc(lang: str = 'en'):
        if lang == 'zh':
            return (
                '该算子用于评估生成的代码片段与其源指令的匹配质量，并输出分数和反馈。\n\n'
                '输入参数：\n'
                "- input_instruction_key: 包含人类指令的字段名 (默认: 'generated_instruction')\n"
                "- input_code_key: 包含生成代码的字段名 (默认: 'generated_code')\n"
                '输出参数：\n'
                "- output_score_key: 用于存储质量分数的字段名 (默认: 'quality_score')\n"
                "- output_feedback_key: 用于存储质量反馈的字段名 (默认: 'quality_feedback')\n"
            )
        return (
            'This operator evaluates the quality of a generated code snippet against its source instruction, '
            'providing a score and feedback.\n\n'
            'Input Parameters:\n'
            "- input_instruction_key: Field name containing the human instruction (default: 'generated_instruction')\n"
            "- input_code_key: Field name containing the generated code (default: 'generated_code')\n"
            'Output Parameters:\n'
            "- output_score_key: Field name to store the quality score (default: 'quality_score')\n"
            "- output_feedback_key: Field name to store the quality feedback (default: 'quality_feedback')\n"
        )

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.input_instruction_key not in data:
            raise ValueError(f'Missing required key: {self.input_instruction_key}')
        if self.input_code_key not in data:
            raise ValueError(f'Missing required key: {self.input_code_key}')
        if self.output_score_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_score_key}')
        if self.output_feedback_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_feedback_key}')
        instruction = data.get(self.input_instruction_key, '')
        code = data.get(self.input_code_key, '')
        user_input = f'Instruction:\n{instruction}\n\nCode:\n```python\n{code}\n```'
        response = self.model.prompt(self.system_prompt)(user_input)
        score, feedback = _parse_score_and_feedback(response)
        data[self.output_score_key] = score
        data[self.output_feedback_key] = feedback
        return data


class CodeQualityScoreFilter(CodeGenOps):
    def __init__(
        self,
        model=None,
        min_score: int = 7,
        max_score: int = 10,
        input_instruction_key: str = 'generated_instruction',
        input_code_key: str = 'generated_code',
        output_score_key: str = 'quality_score',
        output_feedback_key: str = 'quality_feedback',
        output_key: str = 'quality_score_filter_label',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.min_score = min_score
        self.max_score = max_score
        self.input_instruction_key = input_instruction_key
        self.input_code_key = input_code_key
        self.output_score_key = output_score_key
        self.output_feedback_key = output_feedback_key
        self.output_key = output_key
        self.scorer = CodeQualitySampleEvaluator(
            model=model,
            input_instruction_key=input_instruction_key,
            input_code_key=input_code_key,
            output_score_key=output_score_key,
            output_feedback_key=output_feedback_key,
        )

    @staticmethod
    def get_desc(lang: str = 'zh'):
        if lang == 'zh':
            return (
                '基于 LLM 生成的代码质量分数过滤代码样本，评估正确性、完整性、清晰度、最佳实践和效率。\n\n'
                '评估维度：\n'
                '- 正确性：代码语法和逻辑是否正确\n'
                '- 完整性：代码是否完整实现功能\n'
                '- 清晰度：代码是否清晰易懂\n'
                '- 最佳实践：是否遵循编程最佳实践\n'
                '- 效率：代码执行效率如何\n\n'
                '输入参数：\n'
                "- input_code_key: 输入代码字段名 (默认: 'generated_code')\n"
                "- input_instruction_key: 输入指令字段名 (默认: 'generated_instruction')\n"
                "- output_score_key: 输出打分字段名 (默认: 'quality_score')\n"
                "- output_feedback_key: 输出反馈字段名 (默认: 'quality_feedback')\n"
                "- output_key: 输出过滤标签字段名 (默认: 'quality_score_filter_label')\n"
                '- min_score: 最小质量分数阈值 (默认: 7)\n'
                '- max_score: 最大质量分数阈值 (默认: 10)\n\n'
                '输出：\n'
                '- 仅保留质量分数在指定范围内的样本；同时在保留样本写入过滤标签字段。\n'
                '- 特殊规则：score=0（解析失败）会被保留。\n'
            )
        return (
            'Filter code samples based on LLM-generated quality scores evaluating correctness, completeness, '
            'clarity, best practices, and efficiency.\n\n'
            'Input Parameters:\n'
            "- input_code_key: Input code field name (default: 'generated_code')\n"
            "- input_instruction_key: Input instruction field name (default: 'generated_instruction')\n"
            "- output_score_key: Output score field name (default: 'quality_score')\n"
            "- output_feedback_key: Output feedback field name (default: 'quality_feedback')\n"
            "- output_key: Output filter label field name (default: 'quality_score_filter_label')\n"
            '- min_score: Minimum quality score threshold (default: 7)\n'
            '- max_score: Maximum quality score threshold (default: 10)\n\n'
            'Output:\n'
            '- Keeps only samples whose scores fall within the range; writes a filter label on kept samples.\n'
            '- Special rule: score=0 (failed parsing) will be kept.\n'
        )

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.output_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_key}')

        if self.output_score_key not in data:
            data = self.scorer.forward(data)

        score = data.get(self.output_score_key, 0)
        try:
            score_int = int(score)
        except Exception:
            score_int = 0
        pass_filter = (self.min_score <= score_int <= self.max_score) or score_int == 0
        data[self.output_key] = 1 if pass_filter else 0
        if pass_filter:
            return data
        return []


class CodeSandboxSampleEvaluator(CodeGenOps):
    '''
    CodeSandboxSampleEvaluator is an operator that executes code snippets in a secure,
    isolated environment to verify their correctness. It leverages a robust
    PythonExecutor to handle process isolation, timeouts, and capturing results.
    This is the final validation step in the data synthesis pipeline.
    '''

    def __init__(self, language: str = 'python', timeout_length: int = 15, use_process_isolation: bool = True,
                 input_code_key: str = 'generated_code', output_status_key: str = 'sandbox_status',
                 output_log_key: str = 'sandbox_log', **kwargs):
        '''
        Initializes the operator and the underlying PythonExecutor.

        Args:
            timeout_length: Maximum execution time in seconds for each code snippet.
            use_process_isolation: Whether to run code in a separate process for security. Recommended to keep True.
            input_code_key: Field name containing the code to be executed (default: 'generated_code').
            output_status_key: Field name to store the execution status ('PASS' or 'FAIL') (default: 'sandbox_status').
            output_log_key: Field name to store the execution log or error message (default: 'sandbox_log').
        '''
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

    @staticmethod
    def get_desc(lang: str = 'en'):
        '''
        Provides a description of the operator's function and parameters.
        '''
        if lang == 'zh':
            return (
                '该算子在一个安全的沙箱环境中执行代码片段以验证其正确性。\n\n'
                '输入参数：\n'
                "- input_code_key: 包含待执行代码的字段名 (默认: 'generated_code')\n"
                '输出参数：\n'
                "- output_status_key: 用于存储执行状态 ('PASS' 或 'FAIL') 的字段名 (默认: 'sandbox_status')\n"
                "- output_log_key: 用于存储执行日志或错误信息的字段名 (默认: 'sandbox_log')\n"
            )
        else:  # Default to English
            return (
                'This operator executes code snippets in a secure sandbox environment to verify their correctness.\n\n'
                'Input Parameters：\n'
                "- input_code_key: Field name containing the code to be executed (default: 'generated_code')\n"
                'Output Parameters：\n'
                "- output_status_key: Field name to store the execution status ('PASS' or 'FAIL') "
                "(default: 'sandbox_status')\n"
                "- output_log_key: Field name to store the execution log or error message (default: 'sandbox_log')\n"
            )

    def _execute_code_batch(self, code_list: List[str]) -> List[Tuple[str, str]]:
        '''
        Execute a batch of code snippets using the PythonExecutor.

        Args:
            code_list: A list of strings, where each string is a code snippet.

        Returns:
            A list of tuples, where each tuple contains (status, log).
            Status can be 'PASS' or 'FAIL', log contains execution output or error message.
        '''
        # The executor's batch_apply takes a list of code strings and a 'messages' context.
        # For our simple validation, the context can be an empty list.
        results_with_reports = self.executor.batch_apply(code_list, messages=[])

        processed_results = []
        for (result, report) in results_with_reports:
            # The executor's report tells us about success or failure.
            # "Done" indicates success. Anything else (e.g., "Error: ...", "Timeout Error") indicates failure.
            if report == 'Done':
                status = 'PASS'
                # The 'result' can be a dict with 'text' and/or 'images'. We just need the text log.
                log = result.get('text', '') if isinstance(result, dict) else result
            else:
                status = 'FAIL'
                # The report itself is the most informative log on failure.
                log = report

            processed_results.append((status, log))

        return processed_results

    def forward_batch_input(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        '''
        Execute code snippets and return statuses and logs for a batch of inputs.
        Args:
            inputs (List[Dict]): List of input dictionaries
        Returns:
            List[Dict]: List of updated dictionaries
        '''
        LOG.info(f'Evaluating {self.score_name}...')

        # Validate input keys exist
        if not inputs:
            return []

        # Extract code list
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

    def forward(self, data: Dict, **kwargs) -> Dict:
        '''
        Execute code for a single input dictionary.
        '''
        if not isinstance(data, dict):
            raise TypeError('Input must be a dictionary.')

        result = self.forward_batch_input([data], **kwargs)
        return result[0] if result else data
