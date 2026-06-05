import json
import re
from typing import Any, List, Optional

import json5

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.formatter import EmptyFormatter, JsonFormatter

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


def _assistant_content_str(result: Any) -> str:
    if isinstance(result, dict):
        c = result.get('content', '')
        return c if isinstance(c, str) else ''
    if isinstance(result, str):
        return result
    return str(result) if result is not None else ''


def _truncate_conclusion_garbage_suffix(text: str) -> str:
    if not text:
        return text
    t = text
    markers = (
        '<|im_start|>',
        '<|im_sep|>',
        '<|im_end|>',
        '<|assistant|>',
        '<|user|>',
        '<|system|>',
        '</s>',
        '\x00',
    )
    cut = len(t)
    for m in markers:
        p = t.find(m)
        if p >= 0:
            cut = min(cut, p)
    if cut < len(t):
        t = t[:cut]
    # Repeated filler often seen when model derails
    spam = '元素元素元素'
    p = t.find(spam)
    if p >= 0:
        t = t[:p]
    return t.rstrip()


def _loads_json_lenient(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json5.loads(raw)
        except (ValueError, TypeError):
            return None


def _repair_split_r_attributes(text: str) -> str:
    if not text or '"R"' not in text:
        return text
    t = text
    for _ in range(128):
        m = re.search(r'"R"\s*:\s*"([^"]*)"\s*;\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*', t)
        if not m:
            break
        rest = t[m.end():]
        vm = re.match(r'([^"]*)"', rest)
        if vm:
            val = vm.group(1)
            consumed = vm.end()
        else:
            vm2 = re.match(
                r'((?:[^"]|"(?![,\}\]]))+?)(?=\s*";[a-zA-Z_][a-zA-Z0-9_]*\s*=|\s*"|$)',
                rest,
            )
            if not vm2:
                break
            val = vm2.group(1).strip()
            consumed = vm2.end()
        merged = f'"R":"{m.group(1)};{m.group(2)}={val}"'
        t = t[:m.start()] + merged + t[m.end() + consumed:]
    return t


def _normalize_to_conclusion_list(parsed: Any) -> Optional[List[Any]]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and 'conclusion' in parsed and 'R' in parsed:
        return [parsed]
    return None


def _find_matching_array_end(text: str, start: int) -> Optional[int]:
    if start >= len(text) or text[start] != '[':
        return None
    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _parse_embedded_json_array(cleaned: str) -> Optional[List[Any]]:
    start = cleaned.find('[')
    if start < 0:
        return None
    end = _find_matching_array_end(cleaned, start)
    if end is None:
        return None
    segment = cleaned[start:end + 1]
    parsed = _loads_json_lenient(segment)
    if isinstance(parsed, list):
        return parsed
    return None


def _extract_json_array_from_text(text: str) -> Optional[List[Any]]:
    if not text or not isinstance(text, str):
        return None
    cleaned = _repair_split_r_attributes(
        _extract_json_content(_truncate_conclusion_garbage_suffix(text))
    )
    parsed = _loads_json_lenient(cleaned)
    if parsed is not None:
        return _normalize_to_conclusion_list(parsed)
    try:
        from lazyllm.thirdparty import json_repair

        parsed = json_repair.loads(cleaned)
        norm = _normalize_to_conclusion_list(parsed)
        if norm is not None:
            return norm
    except Exception:
        pass
    return _parse_embedded_json_array(cleaned)


def _parse_optional_answer_variants(raw: Any, max_variants: int) -> List[str]:
    cap = max(1, int(max_variants))
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw[:cap]:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    s = _assistant_content_str(raw)
    if not s.strip():
        return []
    arr = _extract_json_array_from_text(s)
    if not arr:
        return []
    out = []
    for x in arr[:cap]:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _dict_with_conclusions_list(obj: dict) -> Optional[List[Any]]:
    cons = obj.get('conclusions')
    return cons if isinstance(cons, list) else None


def _parse_raw_conclusion_to_list(raw: Any, max_per_task: int) -> List[Any]:  # noqa: C901
    cap = max_per_task
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw[:cap]
    if isinstance(raw, dict):
        if 'conclusion' in raw and 'R' in raw:
            return [raw][:cap]
        wrapped = _dict_with_conclusions_list(raw)
        if wrapped is not None:
            return wrapped[:cap]
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        cleaned = _repair_split_r_attributes(
            _extract_json_content(_truncate_conclusion_garbage_suffix(s))
        )
        parsed = _loads_json_lenient(cleaned)
        if isinstance(parsed, list):
            return parsed[:cap]
        if isinstance(parsed, dict):
            if 'conclusion' in parsed and 'R' in parsed:
                return [parsed][:cap]
            wrapped = _dict_with_conclusions_list(parsed)
            if wrapped is not None:
                return wrapped[:cap]
        try:
            from lazyllm.thirdparty import json_repair

            parsed = json_repair.loads(cleaned)
            if isinstance(parsed, list):
                return parsed[:cap]
            if isinstance(parsed, dict):
                if 'conclusion' in parsed and 'R' in parsed:
                    return [parsed][:cap]
                wrapped = _dict_with_conclusions_list(parsed)
                if wrapped is not None:
                    return wrapped[:cap]
        except Exception:
            pass
        fallback = _extract_json_array_from_text(s)
        if fallback is not None:
            return fallback[:cap]
        LOG.warning(
            'Failed to parse raw_conclusion: expected a JSON array of objects with '
            '"conclusion" and "R" (see RAGFactsConclusionPrompt).'
        )
        return []
    return []


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
            # Raw text: models often append chat tokens / spam after JSON; JsonFormatter then fails.
            self._llm_serve = llm.share().prompt(system_prompt).formatter(EmptyFormatter())
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
            raw = _assistant_content_str(result)
            data['raw_conclusion'] = _truncate_conclusion_garbage_suffix(raw)
        except Exception as e:
            LOG.warning(f'Failed to extract conclusion: {e}')
            data['raw_conclusion'] = ''

        return data


class AgenticRAGExpandConclusions(agenticrag):

    def __init__(self, max_per_task: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_per_task = max_per_task

    def forward(self, data: dict) -> List[dict]:
        raw_conclusion = data.get('raw_conclusion', '')
        identifier = data.get('identifier', '')

        parsed = _parse_raw_conclusion_to_list(raw_conclusion, self.max_per_task)
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

    def __init__(self, llm=None, max_variants: int = 20, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.prompt_template = RAGAnswerVariantsPrompt()
        self._max_variants = max(1, int(max_variants))
        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            # Raw text + tolerant parse: JsonFormatter wrongly rejected valid JSON when array
            # strings contained unmatched `{`/`}`; use the same lenient path as conclusions.
            self._llm_serve = llm.share().prompt(system_prompt).formatter(EmptyFormatter())
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
            variants = _parse_optional_answer_variants(result, self._max_variants)
            if variants:
                data['optional_answer'] = variants
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
