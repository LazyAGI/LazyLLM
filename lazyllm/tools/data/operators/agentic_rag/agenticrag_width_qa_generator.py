import json
from typing import List, Optional
from lazyllm import LOG
from lazyllm.components.formatter import JsonFormatter
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


class WidthQAGMergePairs(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.prompt_template = WidthQAGeneratorMergePrompt()
       
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def _build_prompts(self, data: List[dict]) -> list:
        user_prompts = []
        for i in range(len(data) - 1):
            pair = [data[i], data[i + 1]]
            user_prompts.append(self.prompt_template.build_prompt(pair))
        return user_prompts

    def _parse_merge_result(self, result, idx: int, input_batch: List[dict]) -> Optional[dict]:
        try:
            if isinstance(result, dict):
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
            elif isinstance(result, str):
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
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')
            
        if len(data) < 2:
            LOG.warning('Need at least 2 items to merge.')
            return []

        LOG.info(f'Merging {len(data)} items into width questions...')
        user_prompts = self._build_prompts(data)

        if not user_prompts:
            return []

        merge_results = []
        for prompt in user_prompts:
            merge_results.append(self._llm_serve(prompt))

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
        self.output_question_key = output_question_key
        self.prompt_template = WidthQAGeneratorOriginCheckPrompt()
        
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
        self.output_question_key = output_question_key
        self.prompt_template = WidthQAGeneratorQuestionVerifyPrompt()
        
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def _parse_verify_result(self, result) -> Optional[str]:
        try:
            if isinstance(result, dict):
                return result.get('llm_answer', None)
            elif isinstance(result, str):
                result = json.loads(_clean_json_block(result))
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                if isinstance(result, dict):
                    return result.get('llm_answer', None)
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
            data['llm_answer'] = llm_answer
            return data
        except Exception as e:
            LOG.warning(f'Failed to verify question: {e}')
            return []


class WidthQAGFilterByScore(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.score_template = WidthQAGeneratorRecallScorePrompt()
        
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
        llm_answer = data.get('llm_answer', '')

        if not golden_answer or not llm_answer:
            return []

        user_prompt = self.score_template.build_prompt(golden_answer, llm_answer)

        try:
            score_result = self._llm_serve(user_prompt)
            
            if isinstance(score_result, dict):
                score = score_result.get('answer_score', 0)
            elif isinstance(score_result, str):
                score_dict = json.loads(_clean_json_block(score_result))
                score = score_dict.get('answer_score', 0)
            else:
                score = 0
                
            data['llm_score'] = score

            if score >= 1:
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


class AgenticRAGWidthQAGenerator(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.llm = llm

    def forward_batch_input(
            self,
            data: List[dict],
            input_question_key: str = 'question',
            input_identifier_key: str = 'identifier',
            input_answer_key: str = 'answer',
            output_question_key: str = 'generated_width_task',
    ) -> List[dict]:
        from lazyllm import pipeline

        assert isinstance(data, list), 'Input data must be a list'

        LOG.info('Preparing input batch...')
        input_batch = []
        for i, item in enumerate(data):
            input_batch.append({
                'index': i,
                'question': item.get(input_question_key, ''),
                'content_identifier': item.get(input_identifier_key, ''),
                'golden_answer': item.get(input_answer_key, '')
            })

        if len(input_batch) < 2:
            LOG.warning('Need at least 2 items to merge. Returning empty list.')
            return []

        LOG.info('Merging adjacent QA pairs...')
        merge_op = WidthQAGMergePairs(llm=self.llm)
        merged_data_list = merge_op(input_batch)

        if not merged_data_list:
            LOG.warning('No valid merged questions generated.')
            return []

        LOG.info(f'{len(merged_data_list)} questions passed merge.')

        LOG.info('Processing merged questions with pipeline...')
        with pipeline() as ppl_process:
            ppl_process.check = WidthQAGCheckDecomposition(llm=self.llm, output_question_key=output_question_key)
            ppl_process.verify = WidthQAGVerifyQuestion(llm=self.llm, output_question_key=output_question_key)
            ppl_process.filter = WidthQAGFilterByScore(llm=self.llm)

        result_list = ppl_process(merged_data_list)

        LOG.info(f'Width QA generation completed! Final count: {len(result_list)}')
        return result_list
