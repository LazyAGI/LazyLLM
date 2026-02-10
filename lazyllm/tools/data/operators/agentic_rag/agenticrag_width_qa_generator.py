'''AgenticRAG Width QA Generator Operators'''
import json
from typing import List, Optional
from lazyllm import LOG
from ...base_data import data_register
from ...prompts import (
    WidthQAGeneratorMergePrompt,
    WidthQAGeneratorOriginCheckPrompt,
    WidthQAGeneratorQuestionVerifyPrompt,
    WidthQAGeneratorRecallScorePrompt
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


class WidthQAGMergePairs(agenticrag):
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.llm = llm
        self.prompt_template = WidthQAGeneratorMergePrompt()

    def _build_prompts(self, data: List[dict]) -> tuple:

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompts = []
        for i in range(len(data) - 1):
            pair = [data[i], data[i + 1]]
            user_prompts.append(self.prompt_template.build_prompt(pair))
        return system_prompt, user_prompts

    def _parse_merge_result(self, result, idx: int, input_batch: List[dict]) -> Optional[dict]:

        try:
            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
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

    def forward_batch_input(self, data: List[dict]) -> List[dict]:
     
        if len(data) < 2:
            LOG.warning('Need at least 2 items to merge.')
            return []

        LOG.info(f'Merging {len(data)} items into width questions...')
        system_prompt, user_prompts = self._build_prompts(data)

        if not user_prompts:
            return []

        # Call LLM for all pairs
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        merge_results = []
        for prompt in user_prompts:
            merge_results.append(llm_serve(prompt))

        merged_data_list = []
        for idx, result in enumerate(merge_results):
            parsed = self._parse_merge_result(result, idx, data)
            if parsed is not None:
                merged_data_list.append(parsed)

        LOG.info(f'Generated {len(merged_data_list)} merged questions.')
        return merged_data_list


class WidthQAGCheckDecomposition(agenticrag):
  
    def __init__(self, llm=None, output_question_key: str = 'generated_width_task', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.output_question_key = output_question_key
        self.prompt_template = WidthQAGeneratorOriginCheckPrompt()

    def _build_check_input(self, item: dict) -> dict:

        ori_q = item.get('original_question', [])
        return {
            'index': item.get('index', 0),
            'complex_question': item.get('question', ''),
            'original_questions': ori_q if isinstance(ori_q, list) else [ori_q]
        }

    def forward(self, data: dict) -> dict:

        check_input = self._build_check_input(data)

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(check_input)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)

            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

            if isinstance(result, dict):
                state = result.get('state', None)
                complex_question = result.get('complex_question', data.get('question'))

                if state == 1:
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
        self.llm = llm
        self.output_question_key = output_question_key
        self.prompt_template = WidthQAGeneratorQuestionVerifyPrompt()

    def _parse_verify_result(self, result) -> Optional[str]:

        try:
            if isinstance(result, str):
                result = json.loads(_clean_json_block(result))
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

            if isinstance(result, dict):
                return result.get('llm_answer', None)
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse verification result: {e}')
        return None

    def forward(self, data: dict) -> dict:

        question = data.get(self.output_question_key, '')

        verify_input = {
            'index': data.get('index', 0),
            'complex_question': question
        }

        system_prompt = self.prompt_template.build_system_prompt()
        user_prompt = self.prompt_template.build_prompt(verify_input)

        try:
            result = _call_llm_single(self.llm, user_prompt, system_prompt)
            llm_answer = self._parse_verify_result(result)
            data['llm_answer'] = llm_answer
            return data
        except Exception as e:
            LOG.warning(f'Failed to verify question: {e}')
            return []


class WidthQAGFilterByScore(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.llm = llm
        self.score_template = WidthQAGeneratorRecallScorePrompt()

    def forward(self, data: dict) -> dict:
        golden_answer = data.get('original_answer', [])
        llm_answer = data.get('llm_answer', '')

        if not golden_answer or not llm_answer:
            return []

        system_prompt = self.score_template.build_system_prompt()
        user_prompt = self.score_template.build_prompt(golden_answer, llm_answer)

        try:
            score_result = _call_llm_single(self.llm, user_prompt, system_prompt)
            score_dict = json.loads(_clean_json_block(score_result))
            score = score_dict.get('answer_score', 0)
            data['llm_score'] = score

            # Filter out easy questions (score >= 1)
            if score >= 1:
                data.pop('llm_answer', None)
                data.pop('llm_score', None)
                data.pop('state', None)
                return []

            # Clean up temporary fields
            data.pop('llm_answer', None)
            data.pop('llm_score', None)
            data.pop('state', None)
            return data
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            return []
