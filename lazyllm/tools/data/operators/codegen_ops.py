from ..base_data import data_register
from lazyllm.components.formatter import JsonFormatter
import re
from typing import Tuple


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
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = prompt_template or (
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
        self.model = model.share().prompt(sys_prompt)

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.input_key not in data:
            raise ValueError(f'Missing required key: {self.input_key}')
        if self.output_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_key}')
        raw_instruction = _extract_human_instruction(data.get(self.input_key))
        response = self.model(raw_instruction)
        data[self.output_key] = response.strip() if isinstance(response, str) else response
        return data


class CodeInstructionToCodeGenerator(CodeGenOps):
    def __init__(self, model=None, prompt_template=None, input_key='instruction', output_key='generated_code', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = prompt_template or (
            'You are a senior Python engineer.\n'
            'Given a natural language instruction, generate the corresponding Python code.\n'
            'Return only the code. If you include a Markdown code block, use ```python ... ```.\n'
        )
        self.model = model.share().prompt(sys_prompt)

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.input_key not in data:
            raise ValueError(f'Missing required key: {self.input_key}')
        if self.output_key in data:
            raise ValueError(f'The following key already exists and would be overwritten: {self.output_key}')
        instruction = data.get(self.input_key, '')
        response = self.model(instruction)
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
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_instruction_key = input_instruction_key
        self.input_code_key = input_code_key
        self.output_score_key = output_score_key
        self.output_feedback_key = output_feedback_key
        sys_prompt = prompt_template or (
            'You are an automated code reviewer.\n'
            'Evaluate the generated Python code against the given instruction.\n'
            'Please provide a score (0-10) and feedback.\n'
            'Output must be in JSON format:\n'
            '{\n'
            '  "score": <0-10>,\n'
            '  "feedback": "..."\n'
            '}'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

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
        res = self.model(user_input)

        if isinstance(res, dict):
            score = res.get('score', 0)
            feedback = res.get('feedback', 'No feedback provided.')
        else:
            from lazyllm import LOG
            LOG.warning(f'Failed to extract JSON from response: {res}')
            score, feedback = 0, 'Failed to parse LLM evaluation output.'

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
        except (ValueError, TypeError):
            score_int = 0
        pass_filter = (self.min_score <= score_int <= self.max_score)
        data[self.output_key] = 1 if pass_filter else 0
        if pass_filter:
            return data
        return []
