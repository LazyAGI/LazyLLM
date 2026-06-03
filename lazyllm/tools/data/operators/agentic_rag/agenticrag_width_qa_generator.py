from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from lazyllm import LOG
from lazyllm.components.formatter import JsonFormatter
from ...base_data import data_register
from ...prompts import (
    RAGWidthQuestionSynthesisPrompt,
    RAGWidthDecompositionCheckPrompt,
    RAGWidthVerificationPrompt,
    RAGWidthConsistencyScoringPrompt,
)
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')

class WidthQAGMergePairs(agenticrag):

    def __init__(
        self,
        llm=None,
        pair_stride: int = 1,
        max_merge_pairs: Optional[int] = None,
        merge_max_workers: int = 8,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.prompt_template = RAGWidthQuestionSynthesisPrompt()
        self._pair_stride = max(1, int(pair_stride))
        self._max_merge_pairs = max_merge_pairs
        self._merge_max_workers = int(merge_max_workers)

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def _build_prompts(self, data: List[dict]) -> list:
        user_prompts = []
        for i in range(0, len(data) - 1, self._pair_stride):
            pair = [data[i], data[i + 1]]
            user_prompts.append(self.prompt_template.build_prompt(pair))
        if self._max_merge_pairs is not None:
            cap = max(0, int(self._max_merge_pairs))
            user_prompts = user_prompts[:cap]
        return user_prompts

    def _parse_merge_result(self, result, idx: int, input_batch: List[dict]) -> Optional[dict]:
        try:
            # LLM/JsonFormatter 可能返回 list，取首元素
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if not isinstance(result, dict) or 'question' not in result or 'index' not in result:
                LOG.warning(f'[Skipped]: Invalid merge result at index {idx}')
                return None

            indices = result['index'] if isinstance(result['index'], list) else [result['index']]
            group_items = [input_batch[i] for i in indices if i < len(input_batch)]

            if not group_items:
                return None

            return {
                'question': result['question'],
                'content_identifier': result.get('content_identifier', ''),
                'qa_index': indices,
                'index': idx,
                'original_answer': [item['golden_answer'] for item in group_items],
                'original_question': [item['question'] for item in group_items],
            }
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse merge result at index {idx}: {e}')
            return None

    def forward_batch_input(self, data: List[dict], **kwargs) -> List[dict]:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        if len(data) < 2:
            LOG.warning('Need at least 2 items to merge.')
            return []

        user_prompts = self._build_prompts(data)
        LOG.info(
            f'Merging {len(data)} items into width questions '
            f'({len(user_prompts)} LLM calls, stride={self._pair_stride}, workers={self._merge_max_workers})...'
        )

        if not user_prompts:
            return []

        if self._merge_max_workers > 1:
            workers = min(self._merge_max_workers, len(user_prompts))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                merge_results = list(pool.map(self._llm_serve, user_prompts))
        else:
            merge_results = [self._llm_serve(p) for p in user_prompts]

        merged_data_list = []
        for idx, result in enumerate(merge_results):
            parsed = self._parse_merge_result(result, idx, data)
            if parsed is not None:
                merged_data_list.append(parsed)

        LOG.info(f'Generated {len(merged_data_list)} merged questions.')
        return merged_data_list


class WidthQAGCheckDecomposition(agenticrag):

    def __init__(self, llm=None, output_question_key: str = 'generated_width_task',
                 require_state_one: bool = True, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_question_key = output_question_key
        self.require_state_one = require_state_one
        self.prompt_template = RAGWidthDecompositionCheckPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def _build_check_input(self, item: dict) -> dict:
        ori_q = item.get('original_question', [])
        return {
            'index': item.get('index', 0),
            'complex_question': item.get('question', ''),
            'original_questions': ori_q if isinstance(ori_q, list) else [ori_q]
        }

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        check_input = self._build_check_input(data)
        user_prompt = self.prompt_template.build_prompt(check_input)

        try:
            result = self._llm_serve(user_prompt)

            # LLM/JsonFormatter 可能返回 list（与 prompt 示例一致），取首元素
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if isinstance(result, dict):
                raw_state = result.get('state', None)
                try:
                    state = int(raw_state) if raw_state is not None else 0
                except (TypeError, ValueError):
                    state = 0
                complex_question = result.get('complex_question', data.get('question'))

                if state == 1 or (not self.require_state_one and complex_question):
                    data['state'] = state
                    data[self.output_question_key] = complex_question
                    return data
                else:
                    return []
            else:
                LOG.warning('[Skipped]: Invalid check result')
                return []
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse check result: {e}')
            return []


class WidthQAGVerifyQuestion(agenticrag):

    def __init__(self, llm=None, output_question_key: str = 'generated_width_task', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.output_question_key = output_question_key
        self.prompt_template = RAGWidthVerificationPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def _parse_verify_result(self, result) -> Optional[str]:
        try:
            # prompt 示例为 [{ "llm_answer": "..." }]，LLM/JsonFormatter 可能返回 list
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if isinstance(result, dict):
                return result.get('llm_answer') or result.get('answer') or None
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse verification result: {e}')
        return None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        question = data.get(self.output_question_key, '')

        verify_input = {
            'index': data.get('index', 0),
            'complex_question': question
        }

        user_prompt = self.prompt_template.build_prompt(verify_input)

        try:
            result = self._llm_serve(user_prompt)
            llm_answer = self._parse_verify_result(result)
            data['llm_answer'] = llm_answer if llm_answer is not None else ''
            return data
        except Exception as e:
            LOG.warning(f'Failed to verify question: {e}')
            return []


class WidthQAGFilterByScore(agenticrag):

    def __init__(self, llm=None, filter_threshold: Optional[int] = 1, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.filter_threshold = filter_threshold
        self.score_template = RAGWidthConsistencyScoringPrompt()

        if llm is not None:
            system_prompt = self.score_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        golden_answer = data.get('original_answer', [])
        llm_answer = data.get('llm_answer', '') or ''

        # llm_answer 为空时（如 Verify 解析失败）：filter_threshold=None 时仍保留该条
        if not golden_answer:
            return []
        if not llm_answer:
            data.pop('llm_answer', None)
            data.pop('state', None)
            if self.filter_threshold is not None:
                return []
            return data

        user_prompt = self.score_template.build_prompt(golden_answer, llm_answer)

        try:
            score_result = self._llm_serve(user_prompt)

            if isinstance(score_result, dict):
                raw_score = score_result.get('answer_score', 0)
                try:
                    score = int(raw_score) if raw_score is not None else 0
                except (TypeError, ValueError):
                    score = 0
            else:
                score = 0

            data['llm_score'] = score

            if self.filter_threshold is not None and score >= self.filter_threshold:
                data.pop('llm_answer', None)
                data.pop('llm_score', None)
                data.pop('state', None)
                return []

            data.pop('llm_answer', None)
            data.pop('llm_score', None)
            data.pop('state', None)
            return data
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            return []
