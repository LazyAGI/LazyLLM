import json
from typing import List, Optional

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import JsonFormatter

from ...base_data import data_register
from ...prompts import (
    RAGContentIdExtractorPrompt,
    RAGFactsConclusionPrompt,
    RAGTaskToQuestionPrompt,
    RAGQARefinementPrompt,
    RAGTaskSolverPrompt,
    RAGConsistencyScoringPrompt,
    RAGAnswerVariantsPrompt,
    RAGDocGroundedAnswerPrompt,
)

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


def _extract_json_content(item: str) -> str:
    return (
        item.strip()
        .removeprefix('```json')
        .removeprefix('```')
        .removesuffix('```')
        .strip()
    )


def _extract_json_array_from_text(text: str):  # noqa: C901
    if not text or not isinstance(text, str):
        return None
    text = _extract_json_content(text)
    # 先尝试整体解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and 'conclusion' in parsed and 'R' in parsed:
            return [parsed]
        return None
    except json.JSONDecodeError:
        pass
    # 查找第一个 '[' 与匹配的 ']'，再解析
    start = text.find('[')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class AgenticRAGGetIdentifier(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompt_template = RAGContentIdExtractorPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')

        content = data.get(self.input_key, '')
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = self._llm_serve(user_prompt)
            if isinstance(result, dict):
                data['identifier'] = result.get('content_identifier', '')
            else:
                data['identifier'] = ''
        except Exception as e:
            LOG.warning(f'Failed to extract identifier: {e}')
            data['identifier'] = ''

        return data


class AgenticRAGGetConclusion(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompt_template = RAGFactsConclusionPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')
        content = data.get(self.input_key, '')
        user_prompt = self.prompt_template.build_prompt(content)

        try:
            result = self._llm_serve(user_prompt)
            data['raw_conclusion'] = result
        except Exception as e:
            LOG.warning(f'Failed to extract conclusion: {e}')
            data['raw_conclusion'] = ''

        return data


class AgenticRAGExpandConclusions(agenticrag):

    def __init__(self, max_per_task: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_per_task = max_per_task

    def forward(self, data: dict) -> List[dict]:  # noqa: C901
        raw_conclusion = data.get('raw_conclusion', '')
        identifier = data.get('identifier', '')

        if raw_conclusion is None or (isinstance(raw_conclusion, str) and not raw_conclusion.strip()):
            return []

        parsed = None
        # 1) 已是 list（GetConclusion 使用 JsonFormatter 时）
        if isinstance(raw_conclusion, list):
            parsed = raw_conclusion[:self.max_per_task]
        # 2) 已是 dict：单条结论或包装结构
        elif isinstance(raw_conclusion, dict):
            if 'conclusion' in raw_conclusion and 'R' in raw_conclusion:
                parsed = [raw_conclusion]
            else:
                # 常见包装键
                for key in ('conclusions', 'items', 'data', 'result'):
                    if key in raw_conclusion and isinstance(raw_conclusion[key], list):
                        parsed = raw_conclusion[key][:self.max_per_task]
                        break
                if parsed is None:
                    parsed = []
        # 3) 字符串：先整体解析，失败则从文本中抽取 JSON 数组
        elif isinstance(raw_conclusion, str) and raw_conclusion.strip():
            try:
                parsed = json.loads(_extract_json_content(raw_conclusion))
                if isinstance(parsed, list):
                    parsed = parsed[:self.max_per_task]
                elif isinstance(parsed, dict) and 'conclusion' in parsed and 'R' in parsed:
                    parsed = [parsed]
                else:
                    parsed = []
            except json.JSONDecodeError:
                parsed = _extract_json_array_from_text(raw_conclusion)
                if parsed is not None:
                    parsed = parsed[:self.max_per_task]
                else:
                    LOG.warning('Failed to parse conclusion JSON: no valid array found in raw_conclusion')
                    return []

        if not parsed:
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
        self.prompt_template = RAGTaskToQuestionPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict):
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')
        candidate_str = data.get('candidate_tasks_str', '')
        identifier = data.get('identifier', '')
        try:
            task_item = json.loads(_extract_json_content(candidate_str))
            conclusion = task_item.get('conclusion', '')
            relation = task_item.get('R', '')
            user_prompt = self.prompt_template.build_prompt(
                identifier, conclusion, relation
            )

            result = self._llm_serve(user_prompt)
            if isinstance(result, dict) and 'Q' in result:
                data['question'] = str(result['Q'])
                data['answer'] = str(conclusion)
                return data
        except Exception as e:
            LOG.warning(f'Failed to generate question: {e}')

        return []


class AgenticRAGCleanQA(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = RAGQARefinementPrompt()
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')
        question = data.get('question', '')
        answer = data.get('answer', '')

        user_prompt = self.prompt_template.build_prompt(
            {'question': question, 'original_answer': answer}
        )

        try:
            result = self._llm_serve(user_prompt)
            if isinstance(result, dict):
                data['refined_answer'] = str(result.get('refined_answer', ''))
            else:
                data['refined_answer'] = ''
        except Exception as e:
            LOG.warning(f'Failed to clean QA: {e}')
            data['refined_answer'] = ''

        return data


class AgenticRAGLLMVerify(agenticrag):

    def __init__(self, llm=None, filter_threshold: Optional[int] = 1, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.filter_threshold = filter_threshold
        self.prompt_template = RAGTaskSolverPrompt()
        self.score_template = RAGConsistencyScoringPrompt()
        if llm is not None:
            self._llm_answer_serve = llm.share()
            self._llm_answer_serve.start()
            score_system_prompt = self.score_template.build_system_prompt()
            self._llm_score_serve = llm.share().prompt(score_system_prompt).formatter(JsonFormatter())
            self._llm_score_serve.start()
        else:
            self._llm_answer_serve = None
            self._llm_score_serve = None

    def forward(self, data: dict):
        if self._llm_answer_serve is None or self._llm_score_serve is None:
            raise ValueError('LLM is not configured')
        question = data.get('question', '')
        refined_answer = data.get('refined_answer', '')

        user_prompt = self.prompt_template.build_prompt(question)
        try:
            llm_answer = self._llm_answer_serve(user_prompt)
            data['llm_answer'] = llm_answer
        except Exception as e:
            LOG.warning(f'Failed to get LLM answer: {e}')
            return []

        score_prompt = self.score_template.build_prompt(
            refined_answer, llm_answer
        )

        try:
            score_result = self._llm_score_serve(score_prompt)
            if isinstance(score_result, dict):
                raw_score = score_result.get('answer_score', 0)
                try:
                    score = int(raw_score) if raw_score is not None else 0
                except (TypeError, ValueError):
                    score = 0
                data['llm_score'] = score

                # filter_threshold=None 表示不过滤；否则 score >= filter_threshold 时过滤（默认 1，与 DataFlow 一致）
                if self.filter_threshold is not None and score >= self.filter_threshold:
                    return []
            else:
                data['llm_score'] = 0
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            data['llm_score'] = 0

        return data


class AgenticRAGGoldenDocAnswer(agenticrag):

    def __init__(self, llm=None, input_key: str = 'prompts', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompt_template = RAGDocGroundedAnswerPrompt()
        self.score_template = RAGConsistencyScoringPrompt()
        if llm is not None:
            self._llm_answer_serve = llm.share()
            self._llm_answer_serve.start()
            score_system_prompt = self.score_template.build_system_prompt()
            self._llm_score_serve = llm.share().prompt(score_system_prompt).formatter(JsonFormatter())
            self._llm_score_serve.start()
        else:
            self._llm_answer_serve = None
            self._llm_score_serve = None

    def forward(self, data: dict):
        if self._llm_answer_serve is None or self._llm_score_serve is None:
            raise ValueError('LLM is not configured')
        golden_doc = data.get(self.input_key, '')
        question = data.get('question', '')
        refined_answer = data.get('refined_answer', '')

        user_prompt = self.prompt_template.build_prompt(
            golden_doc, question
        )
        try:
            golden_doc_answer = self._llm_answer_serve(user_prompt)
            data['golden_doc_answer'] = golden_doc_answer
        except Exception as e:
            LOG.warning(f'Failed to get golden doc answer: {e}')
            return []

        score_prompt = self.score_template.build_prompt(
            refined_answer, golden_doc_answer
        )

        try:
            score_result = self._llm_score_serve(score_prompt)
            if isinstance(score_result, dict):
                score = score_result.get('answer_score', 0)
                data['golden_doc_score'] = score

                if score < 1:
                    return []
            else:
                return []
        except Exception as e:
            LOG.warning(f'Failed to calculate golden doc score: {e}')
            return []

        return data


class AgenticRAGOptionalAnswers(agenticrag):

    def __init__(self, llm=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = RAGAnswerVariantsPrompt()
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
            self._llm_serve.start()
        else:
            self._llm_serve = None

    def forward(self, data: dict) -> dict:
        if self._llm_serve is None:
            raise ValueError('LLM is not configured')
        refined_answer = data.get('refined_answer', '')

        user_prompt = self.prompt_template.build_prompt(refined_answer)

        try:
            result = self._llm_serve(user_prompt)
            if isinstance(result, list):
                data['optional_answer'] = result
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

    def forward_batch_input(self, data: List[dict], **kwargs) -> List[dict]:
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
