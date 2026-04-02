import hashlib
import json
import re
import random
from typing import Optional, List, Dict, Any

from lazyllm import LOG
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.tools.data.prompts import (
    DomainFinetuneExtractionPrompt,
    DomainFinetuneFieldMappingPrompt,
)

from ..base_data import data_register

if 'data' in LazyLLMRegisterMetaClass.all_clses and 'domain_finetune' in LazyLLMRegisterMetaClass.all_clses['data']:
    domain_finetune = LazyLLMRegisterMetaClass.all_clses['data']['domain_finetune'].base
else:
    domain_finetune = data_register.new_group('domain_finetune')


@data_register('data.domain_finetune', rewrite_func='forward', _concurrency_mode='process')
def rename_key(
    data: dict,
    input_key: str = 'content',
    output_key: str = 'cleaned_content',
    remove_input: bool = True,
):
    assert isinstance(data, dict)
    if input_key not in data:
        return data
    if remove_input:
        data[output_key] = data.pop(input_key)
    else:
        data[output_key] = data[input_key]
    return data


@data_register('data.domain_finetune', rewrite_func='forward', _concurrency_mode='process')
def prepare_load_path(data: dict) -> dict:
    assert isinstance(data, dict)
    t = data.get('_type', '')
    if t == 'text':
        data['_path_to_load'] = data.get('_raw_path', '')
    elif t in ('html', 'pdf'):
        data['_path_to_load'] = data.get('_markdown_path', '')
    else:
        data['_path_to_load'] = ''
    return data


@data_register('data.domain_finetune', rewrite_func='forward', _concurrency_mode='process')
def normalize_text(
    data: dict,
    input_key: str = 'content',
    fix_chinese_punct: bool = False,
    strip_whitespace: bool = True,
) -> dict:
    assert isinstance(data, dict)
    text = data.get(input_key)
    if not isinstance(text, str):
        return data
    if strip_whitespace:
        text = text.strip()
    if fix_chinese_punct:
        replacements = {
            '\uff0c': ',', '\u3002': '.', '\uff01': '!', '\uff1f': '?',
            '\uff1b': ';', '\uff1a': ':', '\u201c': '"', '\u201d': '"',
            '\u2018': "'", '\u2019': "'", '\uff08': '(', '\uff09': ')',
            '\u3010': '[', '\u3011': ']', '\u300a': '<', '\u300b': '>',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
    data[input_key] = text
    return data


@data_register('data.domain_finetune', rewrite_func='forward', _concurrency_mode='process')
def extract_content_text(
    data: dict,
    input_key: str = 'content',
    output_key: str = '_filter_text',
) -> dict:
    assert isinstance(data, dict)
    content = data.get(input_key)
    if isinstance(content, dict):
        if 'messages' in content:
            text = ' '.join(m.get('content', '') for m in content['messages'] if isinstance(m, dict))
        else:
            parts = []
            if content.get('instruction'):
                parts.append(str(content['instruction']))
            if content.get('input'):
                parts.append(str(content['input']))
            if content.get('output'):
                parts.append(str(content['output']))
            text = ' '.join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = ''
    data[output_key] = text
    return data


@data_register('data.domain_finetune', rewrite_func='forward', _concurrency_mode='process')
def merge_context_and_question(
    data: dict,
    question_key: str = 'question',
    context_key: str = 'context',
    target_key: str = 'question',
    context_label: str = 'Context',
    question_label: str = 'Question',
    drop_context: bool = True,
) -> dict:
    assert isinstance(data, dict)
    q = (data.get(question_key) or '').strip()
    ctx = (data.get(context_key) or '').strip()

    if ctx:
        if q:
            merged = f'{context_label}: {ctx}\\n\\n{question_label}: {q}'
        else:
            merged = f'{context_label}: {ctx}'
    else:
        merged = q

    if merged:
        data[target_key] = merged
    if drop_context and context_key in data:
        data.pop(context_key, None)
    return data


class DatasetFormatNormalizer(domain_finetune):
    _HUMAN_ROLES = frozenset({'human', 'user', 'Human', 'User', 'HUMAN', 'USER', 'question'})
    _ASSISTANT_ROLES = frozenset({'gpt', 'assistant', 'bot', 'GPT', 'Assistant', 'Bot', 'ASSISTANT', 'answer', 'output'})
    _SYSTEM_ROLES = frozenset({'system', 'System', 'SYSTEM'})

    def __init__(
        self,
        output_key: str = 'content',
        text_key: str = '_filter_text',
        instruction: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        keep_system: bool = True,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.output_key = output_key
        self.text_key = text_key
        self.instruction = instruction or 'You are a helpful assistant.'
        self.field_mapping = field_mapping or {}
        self.keep_system = keep_system

    def _apply_field_mapping(self, data: dict) -> dict:
        if not self.field_mapping:
            return data
        mapped = dict(data)
        for src, dst in self.field_mapping.items():
            if src in mapped:
                mapped[dst] = mapped.pop(src)
        return mapped

    def _extract_from_turns(self, turns: list) -> dict:
        if not turns:
            return {}
        if (len(turns) == 2
                and turns[0].get('role') == 'user'
                and turns[1].get('role') == 'assistant'):
            return {
                'instruction': self.instruction,
                'input': turns[0]['content'],
                'output': turns[1]['content'],
            }
        return {'messages': [{'role': 'system', 'content': self.instruction}] + turns}

    def _from_conversations(self, conversations: list) -> dict:
        system = self.instruction
        turns = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = turn.get('from') or turn.get('role', '')
            value = turn.get('value') or turn.get('content', '')
            if role in self._SYSTEM_ROLES:
                if self.keep_system:
                    system = value
            elif role in self._HUMAN_ROLES:
                turns.append({'role': 'user', 'content': value})
            elif role in self._ASSISTANT_ROLES:
                turns.append({'role': 'assistant', 'content': value})
        self.instruction = system
        return self._extract_from_turns(turns)

    def _from_messages(self, messages: list) -> dict:
        system = self.instruction
        turns = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                if self.keep_system:
                    system = content
            elif role in ('user', 'assistant'):
                turns.append({'role': role, 'content': content})
        self.instruction = system
        return self._extract_from_turns(turns)

    def _to_filter_text(self, result: dict) -> str:
        if 'messages' in result:
            return ' '.join(m.get('content', '') for m in result['messages'] if isinstance(m, dict))
        parts = []
        for k in ('instruction', 'input', 'output'):
            v = result.get(k, '')
            if v:
                parts.append(str(v))
        return ' '.join(parts)

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        data = self._apply_field_mapping(data)

        # Skip if already normalized
        if self.output_key in data and isinstance(data[self.output_key], dict):
            if self.text_key and self.text_key not in data:
                data[self.text_key] = self._to_filter_text(data[self.output_key])
            return data

        result: Dict[str, Any] = {}

        # Format 1: OpenAI messages
        if 'messages' in data and isinstance(data['messages'], list):
            result = self._from_messages(data['messages'])

        # Format 2: ShareGPT conversations
        elif 'conversations' in data and isinstance(data['conversations'], list):
            result = self._from_conversations(data['conversations'])

        # Format 3: Alpaca
        elif 'instruction' in data and ('output' in data or 'response' in data):
            result = {
                'instruction': data.get('instruction', self.instruction),
                'input': data.get('input', ''),
                'output': data.get('output', data.get('response', '')),
            }

        # Format 4a: question / answer|response
        elif 'question' in data and ('answer' in data or 'response' in data):
            result = {
                'instruction': self.instruction,
                'input': data['question'],
                'output': data.get('answer', data.get('response', '')),
            }

        # Format 4b: query / answer|response
        elif 'query' in data and ('answer' in data or 'response' in data):
            result = {
                'instruction': self.instruction,
                'input': data['query'],
                'output': data.get('answer', data.get('response', '')),
            }

        # Format 4c: prompt / response
        elif 'prompt' in data and 'response' in data:
            result = {
                'instruction': self.instruction,
                'input': data['prompt'],
                'output': data['response'],
            }

        # Format 5: simple input / output
        elif 'input' in data and 'output' in data:
            result = {
                'instruction': self.instruction,
                'input': data['input'],
                'output': data['output'],
            }

        # Format 6: plain text
        elif 'text' in data:
            result = {
                'instruction': self.instruction,
                'input': '',
                'output': str(data['text']),
            }

        # Format 7: content as plain string
        elif self.output_key in data and isinstance(data[self.output_key], str):
            result = {
                'instruction': self.instruction,
                'input': '',
                'output': data[self.output_key],
            }

        if result:
            data[self.output_key] = result
            if self.text_key:
                data[self.text_key] = self._to_filter_text(result)

        return data


class HashDeduplicator(domain_finetune):
    __reg_overwrite__ = 'forward_batch_input'

    def __init__(
        self,
        input_key: str = 'content',
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.input_key = input_key

    def _get_hash(self, data: dict) -> str:
        content = data.get(self.input_key, '')
        if isinstance(content, dict):
            text = str(sorted(content.items()))
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def forward_batch_input(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        assert isinstance(inputs, list)
        seen: set = set()
        result = []
        for item in inputs:
            h = self._get_hash(item)
            if h not in seen:
                seen.add(h)
                result.append(item)
        return result


class ConversationListExpander(domain_finetune):
    __reg_overwrite__ = 'forward_batch_input'

    def __init__(
        self,
        list_key: str = 'data',
        question_prefix: str = '问：',
        answer_prefix: str = '答：',
        min_question_chars: int = 8,
        min_answer_chars: int = 50,
        output_input_key: str = 'input',
        output_output_key: str = 'output',
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.list_key = list_key
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.min_question_chars = min_question_chars
        self.min_answer_chars = min_answer_chars
        self.output_input_key = output_input_key
        self.output_output_key = output_output_key

    def _strip_prefix(self, text: str, prefix: str) -> str:
        return text[len(prefix):].strip() if text.startswith(prefix) else text.strip()

    def _parse_pairs(self, data_list: list) -> List[Dict[str, str]]:
        pairs = []
        i = 0
        while i + 1 < len(data_list):
            q = self._strip_prefix(str(data_list[i]), self.question_prefix)
            a = self._strip_prefix(str(data_list[i + 1]), self.answer_prefix)
            if len(q) >= self.min_question_chars and len(a) >= self.min_answer_chars:
                pairs.append({self.output_input_key: q, self.output_output_key: a})
            i += 2
        return pairs

    def forward_batch_input(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        assert isinstance(inputs, list)
        expanded: List[Dict[str, Any]] = []
        skipped_empty = 0
        total_turns = 0

        for record in inputs:
            data_list = record.get(self.list_key) or []
            if not data_list or len(data_list) < 2:
                skipped_empty += 1
                continue
            pairs = self._parse_pairs(data_list)
            total_turns += len(pairs)
            expanded.extend(pairs)

        LOG.info(
            f'ConversationListExpander: {len(inputs)} 条记录 → {total_turns} 个QA轮次，'
            f'保留 {len(expanded)} 条（跳过 {skipped_empty} 条空记录 + '
            f'{total_turns - len(expanded)} 条过短QA）'
        )
        return expanded

class DomainFormatAlpaca(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        output_key: str = 'formatted_text',
        instruction: Optional[str] = None,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.instruction = instruction or (
            'You are a helpful assistant. Please answer the following question.'
        )

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            return data

        if isinstance(content, dict):
            if 'messages' in content:
                messages = content['messages']
                system = next(
                    (m['content'] for m in messages if isinstance(m, dict) and m.get('role') == 'system'),
                    self.instruction,
                )
                user_turns = [m['content'] for m in messages if isinstance(m, dict) and m.get('role') == 'user']
                assistant_turns = [m['content'] for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
                instruction_text = system
                input_text = user_turns[-1] if user_turns else ''
                output_text = assistant_turns[-1] if assistant_turns else ''
            else:
                instruction_text = content.get('instruction', self.instruction)
                input_text = content.get('input', '')
                output_text = content.get('output', content.get('response', ''))
        else:
            instruction_text = self.instruction
            input_text = ''
            output_text = str(content)

        data[self.output_key] = {
            'instruction': instruction_text,
            'input': input_text,
            'output': output_text,
        }
        return data


class DomainFormatShareGPT(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        output_key: str = 'formatted_text',
        instruction: Optional[str] = None,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.instruction = instruction or (
            'You are a helpful assistant. Please answer the following question.'
        )

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            return data

        if isinstance(content, dict):
            if 'messages' in content:
                messages = content['messages']
                if not messages or messages[0].get('role') != 'system':
                    messages = [{'role': 'system', 'content': self.instruction}] + list(messages)
            else:
                messages = [
                    {
                        'role': 'system',
                        'content': content.get('instruction', self.instruction),
                    },
                    {'role': 'user', 'content': content.get('input', '')},
                    {
                        'role': 'assistant',
                        'content': content.get('output', content.get('response', '')),
                    },
                ]
        else:
            messages = [
                {'role': 'system', 'content': self.instruction},
                {'role': 'user', 'content': str(content)},
                {'role': 'assistant', 'content': ''},
            ]

        data[self.output_key] = {'messages': messages}
        return data


class DomainFormatRaw(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        output_key: str = 'formatted_text',
        instruction: Optional[str] = None,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.instruction = instruction or (
            'You are a helpful assistant. Please answer the following question.'
        )

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            return data
        data[self.output_key] = {
            'system': self.instruction,
            'content': content,
        }
        return data


class DomainFormatChatML(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        output_key: str = 'formatted_text',
        instruction: Optional[str] = None,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.instruction = instruction or 'You are a helpful assistant.'

    @staticmethod
    def _build_chatml(messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            parts.append(f'<|im_start|>{role}\n{content}<|im_end|>')
        return '\n'.join(parts)

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            return data

        if isinstance(content, dict):
            if 'messages' in content:
                messages = list(content['messages'])
                if not messages or messages[0].get('role') != 'system':
                    messages = [{'role': 'system', 'content': self.instruction}] + messages
            else:
                messages = [
                    {'role': 'system', 'content': content.get('instruction', self.instruction)},
                    {'role': 'user', 'content': content.get('input', '')},
                    {'role': 'assistant', 'content': content.get('output', content.get('response', ''))},
                ]
        else:
            messages = [
                {'role': 'system', 'content': self.instruction},
                {'role': 'user', 'content': str(content)},
                {'role': 'assistant', 'content': ''},
            ]

        data[self.output_key] = {'text': self._build_chatml(messages)}
        return data


class TrainValTestSplitter(domain_finetune):
    __reg_overwrite__ = 'forward_batch_input'

    def __init__(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratify_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        assert (
            abs(train_ratio + validation_ratio + test_ratio - 1.0) < 1e-6
        ), 'Ratios must sum to 1.0'
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify_key = stratify_key

    def forward_batch_input(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        assert isinstance(inputs, list), 'inputs must be a list'
        if not inputs:
            return {'train': [], 'validation': [], 'test': []}

        random.seed(self.seed)
        shuffled = inputs.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.validation_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        return {
            'train': shuffled[:n_train],
            'validation': shuffled[n_train: n_train + n_val],
            'test': shuffled[n_train + n_val:],
        }


class LLMDataExtractor(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        output_key: str = '_extracted_samples',
        llm=None,
        num_samples: int = 3,
        extract_format: str = 'qa',
        lang: str = 'zh',
        max_input_chars: int = 3000,
        instruction: str = 'You are a helpful assistant.',
        _concurrency_mode: str = 'thread',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.num_samples = num_samples
        self.extract_format = extract_format
        self.lang = lang
        self.max_input_chars = max_input_chars
        self.instruction = instruction
        self._llm_serve = None
        self._prompt_template = DomainFinetuneExtractionPrompt(
            lang=self.lang,
            extract_format=self.extract_format,
            num_samples=self.num_samples,
        )

        if llm is not None:
            try:
                system_prompt = self._prompt_template.build_system_prompt()
                self._llm_serve = llm.share().prompt(system_prompt)
                self._llm_serve.start()
            except Exception as e:
                LOG.warning(f'LLMDataExtractor: failed to initialize llm serve: {e}')
                self._llm_serve = None

    def _get_text(self, data: dict) -> str:
        content = data.get(self.input_key, '')
        if isinstance(content, dict):
            if 'messages' in content:
                return ' '.join(
                    m.get('content', '') for m in content['messages'] if isinstance(m, dict)
                )
            parts = [str(v) for k, v in content.items() if v and not k.startswith('_')]
            return ' '.join(parts)
        return str(content) if content else ''

    def _build_prompt(self, text: str) -> str:
        return self._prompt_template.build_prompt(text=text, max_input_chars=self.max_input_chars)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return []
        try:
            parsed = json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            return []

        if self.extract_format == 'qa':
            pairs = parsed.get('qa_pairs', [])
            return [
                {
                    'instruction': self.instruction,
                    'input': str(p.get('question', '')),
                    'output': str(p.get('answer', '')),
                }
                for p in pairs
                if isinstance(p, dict) and p.get('question') and p.get('answer')
            ]
        else:
            samples = parsed.get('samples', [])
            return [
                {
                    'instruction': str(s.get('instruction', '')) or self.instruction,
                    'input': str(s.get('input', '')),
                    'output': str(s.get('output', '')),
                }
                for s in samples
                if isinstance(s, dict) and s.get('output')
            ]

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)
        data[self.output_key] = []

        if self._llm_serve is None:
            LOG.warning('LLMDataExtractor: llm is None or not initialized, skipping extraction.')
            return data

        text = self._get_text(data)
        if not text.strip():
            return data

        try:
            prompt = self._build_prompt(text)
            response = self._llm_serve(prompt)
            data[self.output_key] = self._parse_response(response)
        except Exception as e:
            LOG.warning(f'LLMDataExtractor: extraction failed: {e}')

        return data


class LLMFieldMapper(domain_finetune):
    def __init__(
        self,
        output_key: str = 'content',
        text_key: str = '_filter_text',
        llm=None,
        lang: str = 'zh',
        exclude_keys: Optional[List[str]] = None,
        max_data_chars: int = 2000,
        _concurrency_mode: str = 'thread',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.output_key = output_key
        self.text_key = text_key
        self.lang = lang
        self.max_data_chars = max_data_chars
        self.exclude_keys = set(exclude_keys or []) | {
            '_type', '_raw_path', '_markdown_path', '_path_to_load',
            '_filter_text', '_extracted_samples',
        }
        self._llm_serve = None
        self._prompt_template = DomainFinetuneFieldMappingPrompt(lang=self.lang)

        if llm is not None:
            try:
                system_prompt = self._prompt_template.build_system_prompt()
                self._llm_serve = llm.share().prompt(system_prompt)
                self._llm_serve.start()
            except Exception as e:
                LOG.warning(f'LLMFieldMapper: failed to initialize llm serve: {e}')
                self._llm_serve = None

    def _build_prompt(self, data: dict) -> str:
        filtered = {
            k: v for k, v in data.items()
            if k not in self.exclude_keys and not k.startswith('_')
        }
        return self._prompt_template.build_prompt(record=filtered, max_data_chars=self.max_data_chars)

    def _parse_response(self, response: str) -> Optional[Dict[str, str]]:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            return None
        try:
            parsed = json.loads(json_match.group())
            if 'output' in parsed:
                return {
                    'instruction': str(parsed.get('instruction', '')),
                    'input': str(parsed.get('input', '')),
                    'output': str(parsed.get('output', '')),
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _to_filter_text(self, result: dict) -> str:
        parts = [str(result.get(k, '')) for k in ('instruction', 'input', 'output') if result.get(k)]
        return ' '.join(parts)

    def forward(self, data: dict, **kwargs) -> dict:
        assert isinstance(data, dict)

        if self.output_key in data and isinstance(data[self.output_key], dict):
            return data

        if self._llm_serve is None:
            LOG.warning('LLMFieldMapper: llm is None or not initialized, skipping field mapping.')
            return data

        try:
            prompt = self._build_prompt(data)
            response = self._llm_serve(prompt)
            result = self._parse_response(response)
            if result:
                data[self.output_key] = result
                if self.text_key:
                    data[self.text_key] = self._to_filter_text(result)
        except Exception as e:
            LOG.warning(f'LLMFieldMapper: field mapping failed: {e}')

        return data


class OutputContentFilter(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        min_output_chars: int = 80,
        output_field: str = 'output',
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.min_output_chars = min_output_chars
        self.output_field = output_field

    def _get_output_text(self, content) -> str:
        if isinstance(content, dict):
            if 'messages' in content:
                parts = [
                    m.get('content', '') for m in content['messages']
                    if isinstance(m, dict) and m.get('role') == 'assistant'
                ]
                return ' '.join(parts)
            return str(content.get(self.output_field, ''))
        if isinstance(content, str):
            return content
        return ''

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        content = data.get(self.input_key)
        output_text = self._get_output_text(content)
        if len(output_text.strip()) < self.min_output_chars:
            return None
        return data


class InputOutputRatioFilter(domain_finetune):
    def __init__(
        self,
        input_key: str = 'content',
        min_ratio: float = 0.3,
        _concurrency_mode: str = 'process',
        **kwargs,
    ):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.input_key = input_key
        self.min_ratio = min_ratio

    def _extract_io(self, content):
        if isinstance(content, dict):
            if 'messages' in content:
                user_parts = [
                    m.get('content', '') for m in content['messages']
                    if isinstance(m, dict) and m.get('role') == 'user'
                ]
                asst_parts = [
                    m.get('content', '') for m in content['messages']
                    if isinstance(m, dict) and m.get('role') == 'assistant'
                ]
                return ' '.join(user_parts), ' '.join(asst_parts)
            return str(content.get('input', '')), str(content.get('output', ''))
        if isinstance(content, str):
            return '', content
        return '', ''

    def forward(self, data: dict, **kwargs):
        assert isinstance(data, dict)
        content = data.get(self.input_key)
        input_text, output_text = self._extract_io(content)
        input_len = len(input_text.strip())
        output_len = len(output_text.strip())
        if input_len > 0 and output_len / input_len < self.min_ratio:
            return None
        return data


class SampleExpander(domain_finetune):
    __reg_overwrite__ = 'forward_batch_input'

    def __init__(
        self,
        samples_key: str = '_extracted_samples',
        output_key: str = 'content',
        keep_original_keys: Optional[List[str]] = None,
        drop_empty_extraction: bool = False,
        **kwargs,
    ):
        super().__init__(rewrite_func='forward_batch_input', **kwargs)
        self.samples_key = samples_key
        self.output_key = output_key
        self.keep_original_keys = keep_original_keys or []
        self.drop_empty_extraction = drop_empty_extraction

    def forward_batch_input(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        assert isinstance(inputs, list)
        expanded_items: List[Dict[str, Any]] = []
        for record in inputs:
            samples = record.get(self.samples_key, [])
            if isinstance(samples, list) and samples:
                for sample in samples:
                    new_record = {k: record[k] for k in self.keep_original_keys if k in record}
                    new_record[self.output_key] = sample
                    expanded_items.append(new_record)
            elif not self.drop_empty_extraction:
                expanded_items.append(record)
        return expanded_items
