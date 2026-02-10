import re
from lazyllm import LOG, TrainableModule
from lazyllm.tools.data import data_register
from lazyllm.thirdparty import transformers
import regex
from lazyllm.components.formatter import JsonFormatter

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'
Text2qa = data_register.new_group('Text2qa')

def boxed_res_extractor(text):
    if not isinstance(text, str):
        return None
    pattern = r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None

class TextToChunks(Text2qa):
    def __init__(self,
                 input_key='content',
                 output_key='chunk',
                 chunk_size=10,
                 tokenize=True,
                 tokenizer=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        if tokenize and tokenizer is None:
            LOG.warning(
                f'tokenize=True but tokenizer is None, '
                f'loading tokenizer from default model: {DEFAULT_TOKENIZER}'
            )
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    DEFAULT_TOKENIZER,
                    trust_remote_code=True
                )
                self.tokenize = True
            except Exception as e:
                LOG.warning(
                    f'failed to load tokenizer from {DEFAULT_TOKENIZER}, '
                    f'falling back to char count, error: {e}'
                )
                self.tokenize = False
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            self.tokenize = tokenize

    def _get_len(self, text: str):
        if self.tokenize:
            return len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
        return len(text)

    def forward(self, data: dict):
        text = data.get(self.input_key, '')
        if not text:
            return []

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        chunks = []
        cur_parts = []
        cur_len = 0

        for line in lines:
            l_len = self._get_len(line)
            if cur_len + l_len <= self.chunk_size:
                cur_parts.append(line)
                cur_len += l_len
            else:
                if cur_parts:
                    chunks.append('\n'.join(cur_parts))
                cur_parts = [line]
                cur_len = l_len

        if cur_parts:
            chunks.append('\n'.join(cur_parts))

        results = []
        for c in chunks:
            item = data.copy()
            item[self.output_key] = c
            results.append(item)

        return results

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def empty_or_noise_filter(data: dict, input_key='chunk'):
    text = data.get(input_key, '')
    if not text:
        return []

    if not re.search(r'[\w\u4e00-\u9fff]', text):
        return []

    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def invalid_unicode_cleaner(data: dict, input_key='chunk'):
    text = data.get(input_key, '')
    if not text:
        return data

    text = re.sub(
        r'[\uFDD0-\uFDEF\uFFFE\uFFFF'
        r'\U0001FFFE\U0001FFFF'
        r'\U0002FFFE\U0002FFFF'
        r'\U0003FFFE\U0003FFFF'
        r'\U0004FFFE\U0004FFFF'
        r'\U0005FFFE\U0005FFFF'
        r'\U0006FFFE\U0006FFFF'
        r'\U0007FFFE\U0007FFFF'
        r'\U0008FFFE\U0008FFFF'
        r'\U0009FFFE\U0009FFFF'
        r'\U000AFFFE\U000AFFFF'
        r'\U000BFFFE\U000BFFFF'
        r'\U000CFFFE\U000CFFFF'
        r'\U000DFFFE\U000DFFFF'
        r'\U000EFFFE\U000EFFFF'
        r'\U000FFFFE\U000FFFFF'
        r'\U0010FFFE\U0010FFFF]',
        '',
        text
    )

    data[input_key] = text
    return data

class ChunkToQA(Text2qa):
    def __init__(self,
                 input_key='chunk',
                 query_key='query',
                 answer_key='answer',
                 model=None,
                 output_structure=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.query_key = query_key
        self.answer_key = answer_key

        if output_structure is None:
            output_structure = f'''
            输出格式要求：
            {{
                "{self.query_key}": "生成的问题",
                "{self.answer_key}": "答案"
            }}
            '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model
        self.model = self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data: dict, user_prompt=None):
        assert self.input_key in data
        chunk = data.get(self.input_key, '')

        if not chunk:
            data[self.query_key] = ''
            data[self.answer_key] = ''
            return data

        if user_prompt is None:
            user_prompt = '根据下面文本生成一个 QA 对：\n'

        inp = f'{user_prompt}\n{chunk}'

        qa = self.model(inp)

        data[self.query_key] = qa.get(self.query_key, '')
        data[self.answer_key] = qa.get(self.answer_key, '')
        return data

class QAScorer(Text2qa):
    def __init__(self,
                 input_key='chunk',
                 output_key='score',
                 query_key='query',
                 answer_key='answer',
                 model=None,
                 output_structure=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key

        if output_structure is None:
            output_structure = f'''
            输出格式要求：
            {{
                "{self.output_key}": 0 or 1
            }}
            '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.prompt(output_structure)
        self.model = self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data: dict, user_prompt=None):
        assert self.input_key in data
        assert self.query_key in data
        assert self.answer_key in data

        chunk = data.get(self.input_key, '')
        query = data.get(self.query_key, '')
        answer = data.get(self.answer_key, '')

        if not (chunk and query and answer):
            data[self.output_key] = 0
            return data

        if user_prompt is None:
            user_prompt = f'''
        请根据下面内容对 QA 打分：

        原文：
        {chunk}

        问题：
        {query}

        答案：
        {answer}

        规则：
        - 严格基于原文 → 1
        - 否则 → 0
        '''

        res = self.model(user_prompt)

        data[self.output_key] = res.get(self.output_key, 0)
        return data
