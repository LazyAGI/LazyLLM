'''AgenticRAG Depth QA Generator Operators'''
import json
from typing import List, Optional
from lazyllm import LOG
from ...base_data import data_register
from ...prompts import (
    DepthQAGeneratorGetIdentifierPrompt,
    DepthQAGeneratorBackwardTaskPrompt,
    DepthQAGeneratorSupersetCheckPrompt,
    DepthQAGeneratorQuestionPrompt,
    DepthQAGeneratorAnswerPrompt,
    DepthQAGeneratorRecallScorePrompt
)
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


def _clean_json_block(item: str) -> str:

    return item.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()


def _call_llm_single(llm, prompt: str, system_prompt: str = '') -> str:

    if llm is None:
        raise ValueError('LLM is not configured')
    llm_serve = llm.share(prompt=system_prompt)
    llm_serve.start()
    return llm_serve(prompt)


class DepthQAGGetIdentifier(agenticrag):
    
    def __init__(self, llm=None, input_key: str = 'question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.input_key = input_key
        self.prompt_template = DepthQAGeneratorGetIdentifierPrompt()

    def forward(self, data: dict) -> dict:

        # Skip if identifier already exists
        if 'identifier' in data:
            return data

        content = data.get(self.input_key, '')
        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            data['identifier'] = result
        except Exception as e:
            LOG.warning(f'Failed to get identifier: {e}')
            data['identifier'] = ''

        return data


class DepthQAGBackwardTask(agenticrag):

    def __init__(self, llm=None, identifier_key: str = 'identifier',
                 new_identifier_key: str = 'new_identifier', relation_key: str = 'relation', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.identifier_key = identifier_key
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.prompt_template = DepthQAGeneratorBackwardTaskPrompt()

    def forward(self, data: dict) -> dict:

        identifier = data.get(self.identifier_key, '')

        user_prompt = self.prompt_template.build_prompt(identifier)

        try:
            result = _call_llm_single(self.llm, user_prompt, '')
            parsed = self._parse_backward_result(result)
            if parsed is not None:
                data[self.new_identifier_key] = parsed['identifier']
                data[self.relation_key] = parsed['relation']
                return data
        except Exception as e:
            LOG.warning(f'Failed to generate backward task: {e}')

        return []

    def _parse_backward_result(self, result) -> Optional[dict]:

        try:
            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
            if isinstance(result, dict) and 'identifier' in result and 'relation' in result:
                return result
            LOG.warning('[Skipped]: Invalid backward result')
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse backward result: {e}')
        return None


class DepthQAGCheckSuperset(agenticrag):

    def __init__(self, llm=None, new_identifier_key: str = 'new_identifier',
                 relation_key: str = 'relation', identifier_key: str = 'identifier', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.identifier_key = identifier_key
        self.prompt_template = DepthQAGeneratorSupersetCheckPrompt()

    def forward(self, data: dict) -> dict:

        new_identifier = data.get(self.new_identifier_key, '')
        relation = data.get(self.relation_key, '')
        identifier = data.get(self.identifier_key, '')

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(new_identifier, relation, identifier)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            if self._is_valid_superset(result):
                return data
        except Exception as e:
            LOG.warning(f'Failed to check superset: {e}')

        return []

    def _is_valid_superset(self, result) -> bool:

        try:
            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
            return isinstance(result, dict) and result.get('new_query') == 'valid'
        except Exception as e:
            LOG.warning(f'[Error]: Failed to check superset: {e}')
            return False


class DepthQAGGenerateQuestion(agenticrag):

    def __init__(self, llm=None, new_identifier_key: str = 'new_identifier',
                 relation_key: str = 'relation', identifier_key: str = 'identifier',
                 question_key: str = 'depth_question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.identifier_key = identifier_key
        self.question_key = question_key
        self.prompt_template = DepthQAGeneratorQuestionPrompt()

    def forward(self, data: dict) -> dict:

        new_identifier = data.get(self.new_identifier_key, '')
        relation = data.get(self.relation_key, '')
        identifier = data.get(self.identifier_key, '')

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(new_identifier, relation, identifier)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            parsed = self._parse_question_result(result)
            if parsed is not None:
                data[self.question_key] = parsed
                return data
        except Exception as e:
            LOG.warning(f'Failed to generate question: {e}')

        return []

    def _parse_question_result(self, result) -> Optional[str]:

        try:
            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
            if isinstance(result, dict) and 'new_query' in result:
                return result['new_query']
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse question: {e}')
        return None


class DepthQAGVerifyQuestion(agenticrag):

    def __init__(self, llm=None, question_key: str = 'depth_question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.question_key = question_key
        self.answer_template = DepthQAGeneratorAnswerPrompt()
        self.score_template = DepthQAGeneratorRecallScorePrompt()

    def forward(self, data: dict) -> dict:
        question = data.get(self.question_key, '')

        # Ensure refined_answer exists
        if 'refined_answer' not in data and 'answer' in data:
            data['refined_answer'] = data['answer']

        refined_answer = data.get('refined_answer', '')

        # Generate LLM answer
        user_prompt = self.answer_template.build_prompt(question)
        try:
            llm_answer = _call_llm_single(self.llm, user_prompt, '')
            data['llm_answer'] = llm_answer
        except Exception as e:
            LOG.warning(f'Failed to get LLM answer: {e}')
            return []

        # Calculate recall score
        system_prompt = self.score_template.build_system_prompt()
        score_prompt = self.score_template.build_prompt(refined_answer, llm_answer)

        try:
            score_result = _call_llm_single(self.llm, score_prompt, system_prompt)
            score_dict = json.loads(_clean_json_block(score_result))
            score = score_dict.get('answer_score', 0)
            data['llm_score'] = score

            # Filter out easy questions (score >= 1)
            if score >= 1:
                data.pop('llm_answer', None)
                data.pop('llm_score', None)
                return []

            # Clean up temporary fields
            data.pop('llm_answer', None)
            data.pop('llm_score', None)
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            return []

        return data
