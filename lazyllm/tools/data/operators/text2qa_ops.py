import re
from lazyllm import LOG, TrainableModule
from lazyllm.tools.data import data_register
from lazyllm.thirdparty import transformers
from lazyllm.components.formatter import JsonFormatter

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'
Text2qa = data_register.new_group('Text2qa')

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
                 user_prompt=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.query_key}": "问题",
            "{self.answer_key}": "答案"
        }}
        '''

        self.default_prompt = '''
        任务：根据给定文本构造一个用于监督微调（SFT）的问答对。

        约束条件：
        - 提问需要具体，不可以出现文中xxx、今天是xxx、这种莫能两可的问题，
        - 问题必须可以仅通过原文回答。
        - 答案必须忠实于原文表达。
        - 不允许添加任何外部知识。
        - 不允许推测或扩展。
        - 问题应覆盖文本中的核心事实或关键观点。
        - 只生成一个问答对。

        输出要求：
        - 仅输出 JSON。
        - 不要输出解释说明。
        - 不要输出多余内容。
        '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.share()
        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data: dict):
        assert self.input_key in data
        chunk = data.get(self.input_key, '')

        if not chunk:
            data[self.query_key] = ''
            data[self.answer_key] = ''
            return data

        if self.user_prompt is None:
            user_prompt = self.default_prompt
        else:
            user_prompt = self.user_prompt

        inp = f'{user_prompt}\n原文：{chunk}\n'

        inp += f'''
        输出格式要求：
        {{
            "{self.query_key}": "问题",
            "{self.answer_key}": "答案"
        }}
        '''

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
                 user_prompt=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.output_key}": 0 or 1
        }}
        '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.share()
        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data: dict):
        assert self.input_key in data
        assert self.query_key in data
        assert self.answer_key in data

        chunk = data.get(self.input_key, '')
        query = data.get(self.query_key, '')
        answer = data.get(self.answer_key, '')

        if not (chunk and query and answer):
            data[self.output_key] = 0
            return data

        qa = f'问题{query}; 答案{answer}'
        if self.user_prompt is None:
            user_prompt = f'''
        请根据下面内容对 QA 打分：

        原文：
        {chunk}

        {qa}

        评分规则：
        - 如果问题和答案都严格基于原文内容，且答案可以从原文中直接或明确推断得到，打 1。
        - 如果问题或答案与原文无关、编造内容、无法从原文得到依据、或语义混乱，打 0。
        - 如果 QA 无意义或不构成有效问答，打 0。
        输出格式示例：
        {{"{self.output_key}": "0"}}
        '''
        else:
            user_prompt = self.user_prompt + qa
        res = self.model(user_prompt)

        data[self.output_key] = float(res.get(self.output_key, 0))
        return data

@data_register('data.Text2qa', rewrite_func='forward')
def qa_score_filter(data, input_key, min_score):
    score = data.get(input_key, 0)
    if score >= min_score:
        return None
    return []

@data_register('data.Text2qa', rewrite_func='forward')
def to_alpaca_sft(
        data,
        query_key='query',
        context_key='context',
        answer_key='output'
):
    instruction = data.get(query_key)
    context = data.get(context_key, '')
    answer = data.get(answer_key)

    if not instruction or not answer:
        return []

    return {
        'instruction': instruction,
        'input': context if context else '',
        'output': answer
    }

@data_register('data.Text2qa', rewrite_func='forward')
def to_chat_sft(
        data,
        query_key='query',
        context_key='context',
        answer_key='output'
):

    query = data.get(query_key)
    context = data.get(context_key, '')
    answer = data.get(answer_key)

    if not query or not answer:
        return None

    if context:
        user_content = f'{context}\n\n问题：{query}'
    else:
        user_content = query

    return {
        'messages': [
            {
                'role': 'user',
                'content': user_content
            },
            {
                'role': 'assistant',
                'content': answer
            }
        ]
    }

'''
Adapted from the implementation in Tianyi Lab's Cherry_LLM project:
https://github.com/tianyi-lab/Cherry_LLM/blob/main/cherry_seletion/data_by_IFD_vic.py
'''
class IFDScorer(Text2qa):
    def __init__(self,
                 model,
                 tokenizer,
                 input_key='chunk',
                 output_key='IFD_score',
                 query_key='query',
                 answer_key='answer',
                 max_length=512,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _get_token_loss(self, context: str, answer: str):
        from torch.nn import CrossEntropyLoss
        import torch

        text = context + answer
        input_ids = self.tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=self.max_length)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            vocab_size = logits.shape[-1]
            loss_fct = CrossEntropyLoss(reduction='none')
            token_loss = loss_fct(logits.view(-1, vocab_size), target_ids.view(-1))
        return token_loss.cpu().numpy()

    def _calc_loss_part(self, context: str, answer: str):
        import numpy as np
        full_text = context + answer
        input_ids = self.tokenizer.encode(full_text, return_tensors='pt', truncation=True, max_length=self.max_length)
        start_idx = len(self.tokenizer.encode(context)) if context else 0
        end_idx = input_ids.shape[1]

        token_loss_full = self._get_token_loss(context, answer)
        token_loss_answer = token_loss_full[start_idx:end_idx]
        if len(token_loss_answer) == 0:
            return None
        return float(np.mean(token_loss_answer))

    def forward(self, data: dict):
        instruction = data.get(self.query_key, '')
        answer = data.get(self.answer_key, '')

        if not (instruction and answer):
            return []

        # DAS: answer without instruction
        das = self._calc_loss_part('', answer)
        # CAS: answer with instruction
        cas = self._calc_loss_part(instruction, answer)

        if das is None or cas is None or das == 0:
            ifd_score = None
        else:
            ifd_score = cas / das

        data[self.output_key] = ifd_score
        data['CAS'] = cas
        data['DAS'] = das
        return data
