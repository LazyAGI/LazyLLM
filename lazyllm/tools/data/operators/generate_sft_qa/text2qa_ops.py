import re
import json5
from lazyllm import LOG, TrainableModule
from lazyllm.tools.data import data_register
from lazyllm.thirdparty import transformers
from datasketch import MinHash, MinHashLSH

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL = "qwen2.5-0.5B-instruct"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
Text2qa = data_register.new_group('Text2qa')


class TextToChunks(Text2qa):
    def __init__(self,
                 input_key='content',
                 output_key='chunk',
                 chunk_size=10,
                 tokenize=True,
                 tokenizer=None,
                 **kwargs):
        super().__init__(_concurrency_mode="thread", **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        print("加载Tokenizer")
        if tokenize and tokenizer is None:
            LOG.warning(
                f"tokenize=True but tokenizer is None, "
                f"loading tokenizer from default model: {DEFAULT_TOKENIZER}"
            )
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    DEFAULT_TOKENIZER,
                    trust_remote_code=True
                )
                self.tokenize = True
            except Exception as e:
                LOG.warning(
                    f"failed to load tokenizer from {DEFAULT_TOKENIZER}, "
                    f"falling back to char count, error: {e}"
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
        """
        将文本切分为 chunk
        """
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
                    chunks.append("\n".join(cur_parts))
                cur_parts = [line]
                cur_len = l_len

        if cur_parts:
            chunks.append("\n".join(cur_parts))

        results = []
        for c in chunks:
            item = data.copy()
            item[self.output_key] = c
            results.append(item)

        return results

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def html_tag_cleaner(data: dict, input_key='chunk'):
    """
    移除 HTML 标签，如 <p> <br> <div> 等
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = re.sub(r'<[^>]+>', '', text)
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def html_entity_cleaner(data: dict, input_key='chunk'):
    """
    移除 HTML 实体，如 &nbsp; &amp;
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = re.sub(r'&\w+;', ' ', text)
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def whitespace_cleaner(data: dict, input_key='chunk'):
    """
    合并多余空白字符并 strip
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = re.sub(r'\s+', ' ', text).strip()
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def control_char_cleaner(data: dict, input_key='chunk'):
    """
    移除控制字符（\x00-\x1f, \x7f）
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def unicode_space_cleaner(data: dict, input_key='chunk'):
    """
    统一 Unicode 空白字符
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def punctuation_dedup_cleaner(data: dict, input_key='chunk'):
    """
    压缩重复标点
    """
    text = data.get(input_key, "")
    if not text:
        return data

    text = re.sub(r'([!?。，,.])\1+', r'\1', text)
    data[input_key] = text
    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def empty_or_noise_filter(data: dict, input_key='chunk'):
    """
    过滤几乎没有有效字符的 chunk
    """
    text = data.get(input_key, "")
    # 返回 [] 为丢弃
    if not text:
        return []

    # 只剩标点或空白, 返回 [] 为丢弃
    if not re.search(r'[\w\u4e00-\u9fff]', text):
        return []

    return data

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def invalid_unicode_cleaner(data: dict, input_key='chunk'):
    """
    移除常见非法 Unicode 区段
    """
    text = data.get(input_key, "")
    if not text:
        return data

    # 非字符 & 私有区
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

def extract_qa_object(model_output):
    """提取第一个包含 instruction 和 output 字段的 JSON 对象"""
    json_pattern = r'\{[\s\S]*?\}'
    matches = re.findall(json_pattern, model_output)
    for match in matches:
        try:
            obj = json5.loads(match)
            if 'query' in obj and 'answer' in obj:
                return obj
        except json5.JSONDecodeError:
            continue
    return {}

class ChunkToQA(Text2qa):
    def __init__(self,
                 input_key='chunk',
                 query_key='query',
                 answer_key='answer',
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode="thread", **kwargs)

        self.input_key = input_key
        self.query_key = query_key
        self.answer_key = answer_key

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def forward(self, data: dict):
        assert self.input_key in data
        chunk = data.get(self.input_key, '')
        if not chunk:
            data[self.query_key] = ''
            data[self.answer_key] = ''
            return data

        prompt = f"""
根据下面文本生成一个 QA 对：
{chunk}

仅输出 JSON：
{{"{self.query_key}": "问题", "{self.answer_key}": "答案"}}
"""

        response = self.model(prompt)
        qa = self.extract_qa_object(response)

        data[self.query_key] = qa.get(self.query_key, '')
        data[self.answer_key] = qa.get(self.answer_key, '')
        return data

    def extract_qa_object(self, model_output):
        json_pattern = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern, model_output)
        for match in matches:
            try:
                obj = json5.loads(match)
                if self.query_key in obj and self.answer_key in obj:
                    return obj
            except json5.JSONDecodeError:
                continue
        return {}

class QAScorer(Text2qa):
    def __init__(self,
                 input_key='chunk',
                 output_key='score',
                 query_key='query',
                 answer_key='answer',
                 model=None,
                 **kwargs):
        super().__init__(_concurrency_mode="thread", **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

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

        prompt = f"""
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

仅输出 JSON：
{{"{self.output_key}": 0 or 1}}
"""

        response = self.model(prompt)
        res = self.extract_score_object(response)

        data[self.output_key] = res.get(self.output_key, 0)
        return data

    def extract_score_object(self, model_output):
        json_pattern = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern, model_output)
        for match in matches:
            try:
                obj = json5.loads(match)
                if self.output_key in obj:
                    return obj
            except json5.JSONDecodeError:
                continue
        return {}

@data_register(
    'data.Text2qa',
    rewrite_func='forward_batch_input',
    _concurrency_mode='process'
)
def minhash_dedup(data: list,
                  input_key='chunk',
                  num_perm=10,
                  sim_threshold=0.85):

    assert isinstance(data, list)

    keep = []
    drop = []

    lsh = MinHashLSH(
        threshold=sim_threshold,
        num_perm=num_perm
    )

    def build_minhash(text):
        mh = MinHash(num_perm=num_perm)
        for w in re.findall(r'\w+', text.lower()):
            mh.update(w.encode('utf-8'))
        return mh

    for idx, item in enumerate(data):
        text = item.get(input_key, '')

        if not text:
            drop.append(item)
            continue

        mh = build_minhash(text)

        dup = lsh.query(mh)
        if dup:
            drop.append(item)
            continue

        lsh.insert(f"id_{idx}", mh)
        keep.append(item)

    return keep
