import re
import difflib
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import lazyllm

from ...review import configs
from lazyllm import AutoModel
from ....module import LLMBase

DEFAULT_SYSTEM_PROMPT = '你是一个中文纠错专家。请根据用户提供的原始文本，生成纠正后的文本。'
DEFAULT_INSTRUCTION = '纠正输入句子中的语法错误，并输出正确的句子，输入句子为：{sentence}'
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_TEMPERATURE = 0.6


def get_errors(corrected_text, origin_text):  # noqa: C901
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    s = difflib.SequenceMatcher(None, origin_text, corrected_text)

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            origin_part = origin_text[i1:i2]
            corrected_part = corrected_text[j1:j2]

            for idx, (orig_char, corr_char) in enumerate(zip(origin_part, corrected_part)):
                if orig_char not in unk_tokens and corr_char not in unk_tokens:
                    errors.append((orig_char, corr_char, i1 + idx))

            for idx in range(len(corrected_part), len(origin_part)):
                orig_char = origin_part[idx]
                if orig_char not in unk_tokens:
                    errors.append((orig_char, '', i1 + idx))

            for idx in range(len(origin_part), len(corrected_part)):
                corr_char = corrected_part[idx]
                if corr_char not in unk_tokens:
                    errors.append(('', corr_char, i1 + len(origin_part)))

        elif tag == 'delete':
            for idx, char in enumerate(origin_text[i1:i2]):
                if char not in unk_tokens:
                    errors.append((char, '', i1 + idx))

        elif tag == 'insert':
            for _, char in enumerate(corrected_text[j1:j2]):
                if char not in unk_tokens:
                    errors.append(('', char, i1))

    errors = sorted(errors, key=lambda x: x[2])
    return errors


class ChineseCorrector:
    def __init__(self, llm: Optional[LLMBase] = None, base_url: Optional[str] = configs.CORRECTOR_URL,
                 model: Optional[str] = configs.CORRECTOR_MODEL_NAME, api_key: Optional[str] = 'null',
                 source: str = 'openai', system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
                 **_: Any):
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
            'max_tokens': max_tokens or DEFAULT_MAX_LENGTH,
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
            except Exception as e:
                lazyllm.LOG.error(
                    f'Error predicting sentence (length={len(sentence)}): '
                    f'{sentence[:50]}{"..." if len(sentence) > 50 else ""}. '
                    f'Error: {e}'
                )
                response = ''

            errors = get_errors(response, sentence)
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
                      concurrency: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        if not sentences:
            return []

        concurrency = concurrency or batch_size
        results: List[Optional[Dict[str, Any]]] = [None] * len(sentences)

        def _worker(idx: int, sent: str) -> None:
            try:
                res = self._predict([sent], **kwargs)
                results[idx] = res[0] if res else {'source': sent, 'target': sent, 'errors': []}
            except Exception as e:
                lazyllm.LOG.error(f'Error in correct_batch for index {idx}: {e}')
                results[idx] = {'source': sent, 'target': sent, 'errors': []}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_worker, idx, sent): idx
                for idx, sent in enumerate(sentences)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    lazyllm.LOG.error(f'Unhandled exception in correct_batch at index {idx}: {e}')

        return [r for r in results if r is not None]

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
