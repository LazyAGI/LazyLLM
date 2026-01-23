import re
import difflib
from typing import List, Optional, Dict, Any

import lazyllm

from lazyllm import AutoModel, warp, package
from ....module import LLMBase

DEFAULT_INSTRUCTION = '纠正输入句子中的语法错误，并只输出正确的句子，绝对不允许输出其他内容，' \
                      '例如：输入"我喜欢编程成"，输出"我喜欢编程"输入句子为：{sentence}'
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_TEMPERATURE = 0.6


def get_errors(corrected_text, origin_text):  # noqa: C901
    errors = []
    unk_tokens = set([' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', ''])

    def add_error(orig_char, corr_char, pos):
        if orig_char not in unk_tokens and corr_char not in unk_tokens:
            errors.append((orig_char, corr_char, pos))

    matcher = difflib.SequenceMatcher(None, origin_text, corrected_text)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        origin_part = origin_text[i1:i2]
        corrected_part = corrected_text[j1:j2]

        min_len = min(len(origin_part), len(corrected_part))

        for idx in range(min_len):
            add_error(origin_part[idx], corrected_part[idx], i1 + idx)

        for idx in range(min_len, len(origin_part)):
            add_error(origin_part[idx], '', i1 + idx)

        insert_pos = i1 + len(origin_part) if tag == 'replace' else i1
        for idx in range(min_len, len(corrected_part)):
            add_error('', corrected_part[idx], insert_pos)

    return sorted(errors, key=lambda x: x[2])


class ChineseCorrector:
    def __init__(self, llm: Optional[LLMBase] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, api_key: Optional[str] = 'null',
                 source: str = 'openai', **_: Any):
        if llm:
            base_llm = llm
        else:
            base_llm = AutoModel(source=source, model=model)
        self.base_llm = base_llm.prompt(lazyllm.AlpacaPrompter(DEFAULT_INSTRUCTION))

    def _predict(self, sentences: List[str], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, **kwargs) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        llm_kwargs = {
            'max_tokens': max_tokens or DEFAULT_MAX_TOKENS,
            'temperature': temperature if temperature is not None else DEFAULT_TEMPERATURE
        }

        llm_kwargs.update(kwargs)

        results: List[Dict[str, Any]] = []
        for sentence in sentences:
            try:
                response = self.base_llm(
                    dict(sentence=sentence),
                    stream_output=False,
                    **llm_kwargs,
                )
                response = self._post_process(response, sentence)
                errors = get_errors(response, sentence)
            except Exception as e:
                lazyllm.LOG.error(
                    f'Error predicting sentence {sentence[:50]}{"..." if len(sentence) > 50 else ""}. '
                    f'with max_tokens: {max_tokens}, temperature: {temperature}'
                    f'Error: {e}'
                )
                response = ''
                errors = []
            results.append(
                {
                    'source': sentence,
                    'target': response,
                    'errors': errors,
                }
            )

        return results

    def correct(self, sentence: str, **kwargs) -> Dict[str, Any]:
        results = self._predict([sentence], **kwargs)
        return results[0] if results else {'source': sentence, 'target': sentence, 'errors': []}

    def correct_batch(self, sentences: List[str], batch_size: int = DEFAULT_BATCH_SIZE,
                      concurrency: Optional[int] = 2, **kwargs) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        def process_sentence(sent: str) -> Dict[str, Any]:
            try:
                res = self._predict([sent], **kwargs)
                return res[0] if res else {'source': sent, 'target': sent, 'errors': []}
            except Exception as e:
                lazyllm.LOG.error(f'Error processing sentence: {e}')
                return {'source': sent, 'target': sent, 'errors': []}

        try:
            results_package = warp(process_sentence, _concurrent=concurrency)(package(sentences))
            results = list(results_package)
            return results
        except Exception as e:
            lazyllm.LOG.error(f'Error in warp processing: {e}')
            return [{'source': sent, 'target': sent, 'errors': []} for sent in sentences]

    def _post_process(self, response: str, origin: str) -> str:
        response = response.strip()
        match = re.search(r'</think\s*>(.*)', response, re.DOTALL)
        if match:
            response = match.group(1).strip()
        else:
            response = re.sub(r'^<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        sentence_endings = ['。', '！', '？', '；', '：', '，', '、', '.', ',', '?', '!', ':']
        origin_ending = origin[-1] if origin[-1] in sentence_endings else None
        response_ending = response[-1] if response[-1] in sentence_endings else None
        if origin_ending and not response_ending:
            response += origin_ending
        elif not origin_ending and response_ending:
            response = response[:-1]
        return response
