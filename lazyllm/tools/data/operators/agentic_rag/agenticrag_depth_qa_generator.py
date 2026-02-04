'''AgenticRAG Depth QA Generator operator'''
import json
from typing import List, Optional
from lazyllm import LOG
from ...base_data import data_register
from ...prompts.agenticrag import (
    DepthQAGeneratorGetIdentifierPrompt,
    DepthQAGeneratorBackwardTaskPrompt,
    DepthQAGeneratorSupersetCheckPrompt,
    DepthQAGeneratorQuestionPrompt,
    DepthQAGeneratorAnswerPrompt,
    DepthQAGeneratorRecallScorePrompt
)
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 获取或创建 agenticrag 组（确保所有模块共享同一个组）
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


class AgenticRAGDepthQAGenerator(agenticrag):
    '''
    Operator for generating deeper questions based on existing QA pairs.
    该算子以已有问答生成更深度的问题。
    '''

    def __init__(
            self,
            llm=None,
            n_rounds: int = 2,
            **kwargs
    ):
        super().__init__()
        self.n_rounds = n_rounds
        self.llm = llm

    def _generate_from_llm(self, user_prompts, system_prompt=''):
        '''Helper to call LLM serving'''
        if self.llm is None:
            raise ValueError('LLM is not configured')
        llm_serve = self.llm.share(prompt=system_prompt)
        llm_serve.start()
        results = []
        for prompt in user_prompts:
            results.append(llm_serve(prompt))
        return results

    @staticmethod
    def get_desc(lang: str = 'zh'):
        if lang == 'zh':
            return (
                '该算子以已有问答生成更深度的问题。\n\n'
                '输入参数：\n'
                '- input_key: 输入字段名（默认值：\'question\'）\n'
                '- output_key: 输出字段名（默认值：\'depth_question\'）\n'
            )
        elif lang == 'en':
            return (
                'This operator generates deeper questions based on existing QA pairs.\n'
                'Input Parameters:\n'
                '- input_key: Field name for the input (default: \'question\')\n'
                '- output_key: Field name for the output (default: \'depth_question\')\n'
            )
        else:
            return 'DepthQAGenerator generate deeper questions based on existing QA pairs.'

    def _clean_json_block(self, item: str) -> str:
        '''Remove JSON code block markers'''
        return item.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()

    def recall_score(self, data_list: List[dict]) -> List[float]:
        '''Calculate recall scores for generated questions'''
        prompt_template = DepthQAGeneratorRecallScorePrompt()
        golden_answers = [item.get('refined_answer', '') for item in data_list]
        llm_answers = [item.get('llm_answer', '') for item in data_list]
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt(golden_answer, llm_answer)
            for golden_answer, llm_answer in zip(golden_answers, llm_answers)
        ]

        recall_scores = self._generate_from_llm(user_prompts, system_prompt)
        valid_scores = []

        for score_str in recall_scores:
            if score_str is not None:
                try:
                    score_dict = json.loads(self._clean_json_block(score_str))
                    valid_scores.append(score_dict['answer_score'])
                except (json.JSONDecodeError, KeyError) as e:
                    LOG.warning(f'Failed to parse score: {score_str}, Error: {e}')
                    valid_scores.append(0)
            else:
                valid_scores.append(0)

        return valid_scores

    # === Step methods for forward_batch_input ===
    def _step_get_identifiers(self, data_list: List[dict], input_key: str) -> None:
        '''Get identifiers if not present'''
        if not data_list or 'identifier' in data_list[0]:
            return

        LOG.info('Getting identifiers...')
        prompt_template = DepthQAGeneratorGetIdentifierPrompt()
        input_prompts = [item.get(input_key, '') for item in data_list]
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
        identifiers = self._generate_from_llm(user_prompts, system_prompt)

        for item, identifier in zip(data_list, identifiers):
            item['identifier'] = identifier

    def _step_backward_task(self, data_list: List[dict], identifier_key: str,
                            new_identifier_key: str, relation_key: str) -> List[dict]:
        '''Generate backward task (relation and superset)'''
        prompt_template = DepthQAGeneratorBackwardTaskPrompt()
        input_prompts = [item.get(identifier_key, '') for item in data_list]
        user_prompts = [prompt_template.build_prompt(p) for p in input_prompts]
        backward_results = self._generate_from_llm(user_prompts, '')

        new_data_list = []
        for item, result in zip(data_list, backward_results):
            parsed = self._parse_backward_result(result)
            if parsed is not None:
                new_item = item.copy()
                new_item[new_identifier_key] = parsed['identifier']
                new_item[relation_key] = parsed['relation']
                new_data_list.append(new_item)
        return new_data_list

    def _parse_backward_result(self, result) -> Optional[dict]:
        '''Parse backward task result'''
        try:
            if isinstance(result, str):
                result = json.loads(self._clean_json_block(result))
            if isinstance(result, dict) and 'identifier' in result and 'relation' in result:
                return result
            LOG.warning('[Skipped]: Invalid backward result')
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse backward result: {e}')
        return None

    def _step_check_superset(self, data_list: List[dict], new_identifier_key: str,
                             relation_key: str, identifier_key: str) -> List[dict]:
        '''Check if superset is valid'''
        prompt_template = DepthQAGeneratorSupersetCheckPrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt(
                item.get(new_identifier_key, ''),
                item.get(relation_key, ''),
                item.get(identifier_key, '')
            )
            for item in data_list
        ]
        check_results = self._generate_from_llm(user_prompts, system_prompt)

        new_data_list = []
        for item, result in zip(data_list, check_results):
            if self._is_valid_superset(result):
                new_data_list.append(item)
        return new_data_list

    def _is_valid_superset(self, result) -> bool:
        '''Check if result indicates valid superset'''
        try:
            if isinstance(result, str):
                result = json.loads(self._clean_json_block(result))
            return isinstance(result, dict) and result.get('new_query') == 'valid'
        except Exception as e:
            LOG.warning(f'[Error]: Failed to check superset: {e}')
            return False

    def _step_generate_questions(self, data_list: List[dict], new_identifier_key: str,
                                 relation_key: str, identifier_key: str,
                                 question_key: str) -> List[dict]:
        '''Generate new questions'''
        prompt_template = DepthQAGeneratorQuestionPrompt()
        system_prompt = prompt_template.build_system_prompt()
        user_prompts = [
            prompt_template.build_prompt(
                item.get(new_identifier_key, ''),
                item.get(relation_key, ''),
                item.get(identifier_key, '')
            )
            for item in data_list
        ]
        question_results = self._generate_from_llm(user_prompts, system_prompt)

        new_data_list = []
        for item, result in zip(data_list, question_results):
            parsed = self._parse_question_result(result)
            if parsed is not None:
                new_item = item.copy()
                new_item[question_key] = parsed
                new_data_list.append(new_item)
        return new_data_list

    def _parse_question_result(self, result) -> Optional[str]:
        '''Parse question generation result'''
        try:
            if isinstance(result, str):
                result = json.loads(self._clean_json_block(result))
            if isinstance(result, dict) and 'new_query' in result:
                return result['new_query']
        except Exception as e:
            LOG.warning(f'[Error]: Failed to parse question: {e}')
        return None

    def _step_verify_questions(self, data_list: List[dict], question_key: str) -> List[dict]:
        '''Verify with LLM and filter out easy questions'''
        # Ensure refined_answer exists
        for item in data_list:
            if 'refined_answer' not in item and 'answer' in item:
                item['refined_answer'] = item['answer']

        # Generate LLM answers
        prompt_template = DepthQAGeneratorAnswerPrompt()
        temp_questions = [item.get(question_key, '') for item in data_list]
        user_prompts = [prompt_template.build_prompt(q) for q in temp_questions]
        llm_answer_results = self._generate_from_llm(user_prompts, '')

        for item, llm_answer in zip(data_list, llm_answer_results):
            item['llm_answer'] = llm_answer

        # Calculate recall scores and filter
        llm_scores = self.recall_score(data_list)
        new_data_list = []
        for item, score in zip(data_list, llm_scores):
            item['llm_score'] = score
            if score < 1:
                item.pop('llm_answer', None)
                item.pop('llm_score', None)
                new_data_list.append(item)
        return new_data_list

    def _process_single_round(self, data_list: List[dict], round_id: int,
                              output_key: str) -> List[dict]:
        '''Process a single round of depth question generation'''
        LOG.info(f'=== Iteration Round {round_id} ===')

        identifier_key = 'identifier' if round_id == 1 else f'new_identifier_{round_id - 1}'
        new_identifier_key = f'new_identifier_{round_id}'
        relation_key = f'relation_{round_id}'
        question_key = f'{output_key}_{round_id}'

        # Step 1: Generate backward task
        LOG.info(f'Generating backward tasks (round {round_id})...')
        data_list = self._step_backward_task(data_list, identifier_key, new_identifier_key, relation_key)
        if not data_list:
            LOG.warning(f'No valid data after backward task generation in round {round_id}')
            return []

        # Step 2: Check superset validity
        LOG.info(f'Checking superset validity (round {round_id})...')
        data_list = self._step_check_superset(data_list, new_identifier_key, relation_key, identifier_key)
        if not data_list:
            LOG.warning(f'No valid data after superset check in round {round_id}')
            return []

        # Step 3: Generate new questions
        LOG.info(f'Generating new questions (round {round_id})...')
        data_list = self._step_generate_questions(
            data_list, new_identifier_key, relation_key, identifier_key, question_key
        )
        if not data_list:
            LOG.warning(f'No valid questions generated in round {round_id}')
            return []

        # Step 4: Verify with LLM
        LOG.info(f'Verifying questions with LLM (round {round_id})...')
        data_list = self._step_verify_questions(data_list, question_key)
        if not data_list:
            LOG.warning(f'No data left after LLM verification in round {round_id}. All questions were too easy.')
            return []

        LOG.info(f'Round {round_id} completed. Remaining items: {len(data_list)}')
        return data_list

    def forward_batch_input(
            self,
            data: List[dict],
            input_key: str = 'question',
            output_key: str = 'depth_question',
    ) -> List[dict]:
        '''
        Process data to generate depth QA pairs.

        Args:
            data: List of dict containing questions
            input_key: Key for input questions
            output_key: Key for output depth questions

        Returns:
            List of dict with generated depth QA pairs
        '''
        assert isinstance(data, list), 'Input data must be a list'
        data_list = data.copy()

        # Get identifiers if not present
        self._step_get_identifiers(data_list, input_key)

        # Iterative depth question generation
        for round_id in range(1, self.n_rounds + 1):
            data_list = self._process_single_round(data_list, round_id, output_key)
            if not data_list:
                break

        LOG.info(f'Depth QA generation completed! Final count: {len(data_list)}')
        return data_list
