import json
from typing import List

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass

from ...base_data import data_register
from ...prompts import (
    AtomicTaskGeneratorGetIdentifierPrompt,
    AtomicTaskGeneratorGetConclusionPrompt,
    AtomicTaskGeneratorQuestionPrompt,
    AtomicTaskGeneratorCleanQAPrompt,
    AtomicTaskGeneratorAnswerPrompt,
    AtomicTaskGeneratorRecallScorePrompt,
    AtomicTaskGeneratorOptionalAnswerPrompt,
    AtomicTaskGeneratorGoldenDocAnswerPrompt,
)

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


def _clean_json_block(item: str) -> str:
    return (
        item.strip()
        .removeprefix('```json')
        .removeprefix('```')
        .removesuffix('```')
        .strip()
    )


def _call_llm_single(llm, prompt: str, system_prompt: str = '') -> str:
    if llm is None:
        raise ValueError('LLM is not configured')
    llm_serve = llm.share(prompt=system_prompt)
    llm_serve.start()
    return llm_serve(prompt)


class AgenticRAGGetIdentifier(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.input_key = input_key
        self.prompt_template = AtomicTaskGeneratorGetIdentifierPrompt()

    def forward(self, data: dict) -> dict:
        '''Extract identifier from a single document.'''
        content = data.get(self.input_key, '')
        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            cleaned = _clean_json_block(result)
            identifier_obj = json.loads(cleaned)
            data['identifier'] = identifier_obj.get('content_identifier', '')
        except Exception as e:
            LOG.warning(f'Failed to extract identifier: {e}')
            data['identifier'] = ''

        return data


class AgenticRAGGetConclusion(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.input_key = input_key
        self.prompt_template = AtomicTaskGeneratorGetConclusionPrompt()

    def forward(self, data: dict) -> dict:
        content = data.get(self.input_key, '')
        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            data['raw_conclusion'] = result
        except Exception as e:
            LOG.warning(f'Failed to extract conclusion: {e}')
            data['raw_conclusion'] = ''

        return data


class AgenticRAGExpandConclusions(agenticrag):

    def __init__(self, max_per_task: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_per_task = max_per_task

    def forward(self, data: dict) -> List[dict]:
        conclusion_str = data.get('raw_conclusion', '')
        identifier = data.get('identifier', '')

        if not conclusion_str:
            return []

        try:
            parsed = json.loads(_clean_json_block(conclusion_str))
            if isinstance(parsed, list):
                parsed = parsed[:self.max_per_task]
            else:
                return []
        except Exception as e:
            LOG.warning(f'Failed to parse conclusion JSON: {e}')
            return []

        expanded_rows = []
        for item in parsed:
            if isinstance(item, dict) and 'conclusion' in item and 'R' in item:
                new_row = data.copy()
                new_row['candidate_tasks_str'] = json.dumps(item, ensure_ascii=False)
                new_row['identifier'] = str(identifier)
                expanded_rows.append(new_row)

        return expanded_rows


class AgenticRAGGenerateQuestion(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.prompt_template = AtomicTaskGeneratorQuestionPrompt()

    def forward(self, data: dict):
        candidate_str = data.get('candidate_tasks_str', '')
        identifier = data.get('identifier', '')

        try:
            task_item = json.loads(_clean_json_block(candidate_str))
            conclusion = task_item.get('conclusion', '')
            relation = task_item.get('R', '')

            system_prompt = self.prompt_template.build_system_prompt()
            user_prompt = self.prompt_template.build_prompt(
                identifier, conclusion, relation
            )

            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            parsed = json.loads(_clean_json_block(result))

            if isinstance(parsed, dict) and 'Q' in parsed:
                data['question'] = str(parsed['Q'])
                data['answer'] = str(conclusion)
                return data
        except Exception as e:
            LOG.warning(f'Failed to generate question: {e}')

        return []


class AgenticRAGCleanQA(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.prompt_template = AtomicTaskGeneratorCleanQAPrompt()

    def forward(self, data: dict) -> dict:
        question = data.get('question', '')
        answer = data.get('answer', '')

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(
            {'question': question, 'original_answer': answer}
        )

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            parsed = json.loads(_clean_json_block(result))
            data['refined_answer'] = str(parsed.get('refined_answer', ''))
        except Exception as e:
            LOG.warning(f'Failed to clean QA: {e}')
            data['refined_answer'] = ''

        return data


class AgenticRAGLLMVerify(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.prompt_template = AtomicTaskGeneratorAnswerPrompt()
        self.score_template = AtomicTaskGeneratorRecallScorePrompt()

    def forward(self, data: dict):
        question = data.get('question', '')
        refined_answer = data.get('refined_answer', '')

        user_prompt = self.prompt_template.build_prompt(question)
        try:
            llm_answer = _call_llm_single(self.llm, user_prompt, '')
            data['llm_answer'] = llm_answer
        except Exception as e:
            LOG.warning(f'Failed to get LLM answer: {e}')
            return []

        system_prompt = self.score_template.build_system_prompt()
        score_prompt = self.score_template.build_prompt(
            refined_answer, llm_answer
        )

        try:
            score_result = _call_llm_single(
                self.llm, score_prompt, system_prompt
            )
            score_dict = json.loads(_clean_json_block(score_result))
            score = score_dict.get('answer_score', 0)
            data['llm_score'] = score

            if score >= 1:
                return []
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            data['llm_score'] = 0

        return data


class AgenticRAGGoldenDocAnswer(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.input_key = input_key
        self.prompt_template = AtomicTaskGeneratorGoldenDocAnswerPrompt()
        self.score_template = AtomicTaskGeneratorRecallScorePrompt()

    def forward(self, data: dict):
        golden_doc = data.get(self.input_key, '')
        question = data.get('question', '')
        refined_answer = data.get('refined_answer', '')

        user_prompt = self.prompt_template.build_prompt(
            golden_doc, question
        )
        try:
            golden_doc_answer = _call_llm_single(self.llm, user_prompt, '')
            data['golden_doc_answer'] = golden_doc_answer
        except Exception as e:
            LOG.warning(f'Failed to get golden doc answer: {e}')
            return []

        system_prompt = self.score_template.build_system_prompt()
        score_prompt = self.score_template.build_prompt(
            refined_answer, golden_doc_answer
        )

        try:
            score_result = _call_llm_single(
                self.llm, score_prompt, system_prompt
            )
            score_dict = json.loads(_clean_json_block(score_result))
            score = score_dict.get('answer_score', 0)
            data['golden_doc_score'] = score

            if score < 1:
                return []
        except Exception as e:
            LOG.warning(f'Failed to calculate golden doc score: {e}')
            return []

        return data


class AgenticRAGOptionalAnswers(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.prompt_template = AtomicTaskGeneratorOptionalAnswerPrompt()

    def forward(self, data: dict) -> dict:
        refined_answer = data.get('refined_answer', '')

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(refined_answer)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            parsed = json.loads(_clean_json_block(result))

            if isinstance(parsed, list):
                data['optional_answer'] = parsed
            else:
                data['optional_answer'] = [refined_answer]
        except Exception as e:
            LOG.warning(f'Failed to generate optional answers: {e}')
            data['optional_answer'] = [refined_answer]

        return data


class AgenticRAGGroupAndLimit(agenticrag):

    def __init__(
        self,
        input_key: str = 'prompts',
        max_question: int = 10,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.input_key = input_key
        self.max_question = max_question

    def forward_batch_input(self, data: List[dict]) -> List[dict]:
        grouped_data = {}

        for item in data:
            key_value = item.get(self.input_key, '')
            grouped_data.setdefault(key_value, [])

            if len(grouped_data[key_value]) < self.max_question:
                grouped_data[key_value].append(item)

        result_list = []
        for items in grouped_data.values():
            result_list.extend(items)

        LOG.info(f'Grouped and limited to {len(result_list)} QA pairs')
        return result_list
