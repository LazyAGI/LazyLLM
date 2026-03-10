from typing import Optional
from lazyllm import LOG
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts import (
    RAGDepthQueryIdPrompt,
    RAGDepthBackwardSupersetPrompt,
    RAGDepthSupersetValidationPrompt,
    RAGDepthQuestionFromContextPrompt,
    RAGDepthSolverPrompt,
    RAGDepthConsistencyScoringPrompt,
)
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')

class DepthQAGGetIdentifier(agenticrag):

    def __init__(self, llm=None, input_key: str = 'question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompt_template = RAGDepthQueryIdPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt)
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        # Skip if identifier already exists
        if 'identifier' in data:
            return data

        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        content = data.get(self.input_key, '')
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = self._llm_serve(user_prompt)
            data['identifier'] = result
        except Exception as e:
            LOG.warning(f'Failed to get identifier: {e}')
            data['identifier'] = ''

        return data


class DepthQAGBackwardTask(agenticrag):

    def __init__(self, llm=None, identifier_key: str = 'identifier',
                 new_identifier_key: str = 'new_identifier', relation_key: str = 'relation', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.identifier_key = identifier_key
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.prompt_template = RAGDepthBackwardSupersetPrompt()

        if llm is not None:
            self._llm_serve = llm.share().formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        identifier = data.get(self.identifier_key, '')
        user_prompt = self.prompt_template.build_prompt(identifier)

        try:
            result = self._llm_serve(user_prompt)
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
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.identifier_key = identifier_key
        self.prompt_template = RAGDepthSupersetValidationPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        new_identifier = data.get(self.new_identifier_key, '')
        relation = data.get(self.relation_key, '')
        identifier = data.get(self.identifier_key, '')

        user_prompt = self.prompt_template.build_prompt(new_identifier, relation, identifier)

        try:
            result = self._llm_serve(user_prompt)
            if self._is_valid_superset(result):
                return data
        except Exception as e:
            LOG.warning(f'Failed to check superset: {e}')

        return []

    def _is_valid_superset(self, result) -> bool:
        try:
            if isinstance(result, dict):
                return result.get('new_query') == 'valid'
        except Exception as e:
            LOG.warning(f'[Error]: Failed to check superset: {e}')
        return False


class DepthQAGGenerateQuestion(agenticrag):

    def __init__(self, llm=None, new_identifier_key: str = 'new_identifier',
                 relation_key: str = 'relation', identifier_key: str = 'identifier',
                 question_key: str = 'depth_question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.new_identifier_key = new_identifier_key
        self.relation_key = relation_key
        self.identifier_key = identifier_key
        self.question_key = question_key
        self.prompt_template = RAGDepthQuestionFromContextPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        new_identifier = data.get(self.new_identifier_key, '')
        relation = data.get(self.relation_key, '')
        identifier = data.get(self.identifier_key, '')

        user_prompt = self.prompt_template.build_prompt(new_identifier, relation, identifier)

        try:
            result = self._llm_serve(user_prompt)
            parsed = self._parse_question_result(result)
            if parsed is not None:
                data[self.question_key] = parsed
                return data
        except Exception as e:
            LOG.warning(f'Failed to generate question: {e}')

        return []

    def _parse_question_result(self, result) -> Optional[str]:
        try:
            if isinstance(result, dict) and 'new_query' in result:
                return result['new_query']
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse question: {e}')
        return None


class DepthQAGVerifyQuestion(agenticrag):

    def __init__(self, llm=None, question_key: str = 'depth_question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.question_key = question_key
        self.answer_template = RAGDepthSolverPrompt()
        self.score_template = RAGDepthConsistencyScoringPrompt()

        if llm is not None:
            self._llm_answer_serve = llm.share()
            self._llm_answer_serve.start()

            score_system_prompt = self.score_template.build_system_prompt()
            self._llm_score_serve = llm.share().prompt(score_system_prompt).formatter(JsonFormatter())
            self._llm_score_serve.start()
        else:
            self._llm_answer_serve = None
            self._llm_score_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_answer_serve is None or self._llm_score_serve is None:
            raise ValueError('LLM is not configured')

        question = data.get(self.question_key, '')

        if 'refined_answer' not in data and 'answer' in data:
            data['refined_answer'] = data['answer']

        refined_answer = data.get('refined_answer', '')

        user_prompt = self.answer_template.build_prompt(question)
        try:
            llm_answer = self._llm_answer_serve(user_prompt)
            data['llm_answer'] = llm_answer
        except Exception as e:
            LOG.warning(f'Failed to get LLM answer: {e}')
            return []

        score_prompt = self.score_template.build_prompt(refined_answer, llm_answer)

        try:
            score_result = self._llm_score_serve(score_prompt)
            if isinstance(score_result, dict):
                score = score_result.get('answer_score', 0)
            else:
                score = 0
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
