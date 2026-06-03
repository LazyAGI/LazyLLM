import json
from typing import Any, List, Optional

from lazyllm import LOG
from lazyllm.components.formatter import EmptyFormatter, JsonFormatter

try:
    import json5
except ImportError:
    json5 = None
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


def _unwrap_first_list(result: Any) -> Any:
    if isinstance(result, list):
        if len(result) == 0:
            return None
        return result[0]
    return result


def _normalize_content_identifier_value(val: Any) -> str:
    if val is None:
        return ''
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, list):
        parts: list[str] = []
        for x in val:
            if isinstance(x, str) and x.strip():
                parts.append(x.strip())
            elif isinstance(x, (int, float, bool)):
                parts.append(str(x))
        if not parts:
            LOG.warning(
                'DepthQAGGetIdentifier: content_identifier list contained no string/scalar items; '
                'expected a single string per prompt.'
            )
            return ''
        if len(parts) > 1:
            LOG.warning(
                'DepthQAGGetIdentifier: content_identifier was a list of multiple items; '
                'using the first string only.'
            )
        return parts[0]
    if isinstance(val, dict):
        LOG.warning(
            'DepthQAGGetIdentifier: content_identifier must be a plain string, not an object; ignoring.'
        )
        return ''
    LOG.warning(
        'DepthQAGGetIdentifier: unsupported type for content_identifier: %s',
        type(val).__name__,
    )
    return ''


def _assistant_content_str(result: Any) -> str:
    if isinstance(result, dict):
        c = result.get('content', '')
        return c if isinstance(c, str) else ''
    if isinstance(result, str):
        return result
    return str(result) if result is not None else ''


def _strip_optional_json_lang_prefix(piece: str) -> str:
    t = piece.lstrip('\n\r ')
    if len(t) >= 4 and t[:4].lower() == 'json' and (len(t) == 4 or t[4] in '\n\r \t'):
        return t[4:].lstrip('\n\r ')
    return piece


def _chunks_from_triple_backtick_split(s: str) -> List[str]:
    parts = s.split('```')
    if len(parts) < 2:
        return []
    chunks: List[str] = []
    for piece in parts[1:]:
        if not piece.strip():
            continue
        body = _strip_optional_json_lang_prefix(piece)
        chunks.append(body.strip())
    return chunks


def _parse_backward_json_object(text: str) -> Optional[dict]:  # noqa: C901
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    chunks = _chunks_from_triple_backtick_split(s)
    if not chunks:
        chunks.append(s)

    from lazyllm.thirdparty import json_repair

    def _try_parse(chunk: str) -> Optional[dict]:
        for fn in (json.loads, json5.loads if json5 else None, json_repair.loads):
            if fn is None:
                continue
            try:
                obj = fn(chunk)
                if isinstance(obj, dict) and 'identifier' in obj and 'relation' in obj:
                    return obj
            except Exception:
                continue
        start = chunk.find('{')
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(chunk)):
            c = chunk[i]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    sub = chunk[start:i + 1]
                    for fn in (json.loads, json5.loads if json5 else None, json_repair.loads):
                        if fn is None:
                            continue
                        try:
                            obj = fn(sub)
                            if isinstance(obj, dict) and 'identifier' in obj and 'relation' in obj:
                                return obj
                        except Exception:
                            continue
                    break
        return None

    # Prefer later fenced blocks: models often emit a broken JSON first, then a corrected one.
    for chunk in reversed(chunks):
        parsed = _try_parse(chunk)
        if parsed is not None:
            return parsed
    try:
        obj = json_repair.loads(s)
        if isinstance(obj, dict) and 'identifier' in obj and 'relation' in obj:
            return obj
    except Exception:
        pass
    return None


def _parse_depth_identifier_llm_result(result: Any) -> str:
    result = _unwrap_first_list(result)
    if result is None:
        return ''

    if isinstance(result, dict):
        if 'content_identifier' not in result:
            LOG.warning(
                'DepthQAGGetIdentifier: JSON missing "content_identifier"; '
                'expected {{"content_identifier": "<string>"}}. Keys: %s',
                list(result.keys()),
            )
            return ''
        return _normalize_content_identifier_value(result['content_identifier'])

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, (int, float, bool)):
        return str(result)

    LOG.warning(
        'DepthQAGGetIdentifier: unexpected LLM result type after unwrap: %s',
        type(result).__name__,
    )
    return ''


class DepthQAGGetIdentifier(agenticrag):

    def __init__(self, llm=None, input_key: str = 'question', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.prompt_template = RAGDepthQueryIdPrompt()

        if llm is not None:
            system_prompt = self.prompt_template.build_system_prompt()
            self._llm_serve = llm.share().prompt(system_prompt).formatter(JsonFormatter())
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
            data['identifier'] = _parse_depth_identifier_llm_result(result)
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
            system_prompt = self.prompt_template.build_system_prompt()
            # Raw text + tolerant parse: models often wrap JSON in ```json``` or add commentary.
            self._llm_serve = llm.share().prompt(system_prompt).formatter(EmptyFormatter())
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
                return {
                    'identifier': str(result['identifier']),
                    'relation': str(result['relation']),
                }
            text = _assistant_content_str(result)
            obj = _parse_backward_json_object(text)
            if obj is not None:
                return {
                    'identifier': str(obj.get('identifier', '')),
                    'relation': str(obj.get('relation', '')),
                }
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
                new_query = result.get('new_query')
                if new_query is None:
                    return False
                if not isinstance(new_query, str):
                    LOG.warning(
                        'DepthQAGCheckSuperset: new_query must be a string (e.g. "valid"), got %s',
                        type(new_query).__name__,
                    )
                    return False
                return new_query.lower() == 'valid'
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

    def __init__(self, llm=None, question_key: str = 'depth_question',
                 filter_threshold: Optional[int] = 1, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.question_key = question_key
        self.filter_threshold = filter_threshold
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
                return []

            # Clean up temporary fields
            data.pop('llm_answer', None)
            data.pop('llm_score', None)
        except Exception as e:
            LOG.warning(f'Failed to calculate recall score: {e}')
            return []

        return data
