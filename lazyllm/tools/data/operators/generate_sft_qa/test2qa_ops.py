import re
import uuid
import os
from datetime import datetime
from lazyllm import LOG
from lazyllm import OnlineChatModule
from ...base_data import data_register
import re
import json5

Text2qa = data_register.new_group('Text2qa')
# 使用静态变量缓存 tokenizer，避免重复加载
_TOKENIZER_CACHE = {}

def get_tokenizer(path):
    if path not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOKENIZER_CACHE[path]

text2qa = data_register.new_group('Text2qa')

@data_register('data.Text2qa', rewrite_func='forward')
def text_to_chunks(data, input_key='content', output_key='chunk', chunk_size=10, tokenize=True, tokenizer_path=None):
    assert isinstance(data, dict)
    text = data.get(input_key, '')
    if not text:
        return []

    # 1. 初始化 Tokenizer
    tk = None
    if tokenize:
        path = tokenizer_path or '/home/mnt/cuishaoting/run_sft/Qwen2.5-0.5B/tokenizer.json'
        try:
            tk = get_tokenizer(path)
        except Exception as e:
            LOG.error(f"Failed to load tokenizer from {path}: {e}")
            tokenize = False

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    chunks_text = []
    current_parts = []
    current_len = 0

    def get_len(t):
        return len(tk.encode(t, add_special_tokens=False)) if tokenize else len(t)

    for line in lines:
        l_len = get_len(line)

        if current_len + l_len <= chunk_size:
            current_parts.append(line)
            current_len += l_len
        else:
            if current_parts:
                chunks_text.append("\n".join(current_parts))
            current_parts = [line]
            current_len = l_len

    if current_parts:
        chunks_text.append("\n".join(current_parts))

    # ✅ 4. 构造输出
    results = []
    for chunk in chunks_text:
        new_item = data.copy()
        new_item[output_key] = chunk
        results.append(new_item)

    return results

def extract_json(response: str):
    if not response:
        return None

    if '</think>' in response:
        response = response.split('</think>')[-1]

    response = response.strip()
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]

    for pat in patterns:
        m = re.search(pat, response)
        if m:
            response = m.group(1).strip()
            break

    if '{' in response and '}' in response:
        response = response[response.find('{'): response.rfind('}') + 1]

    try:
        return json5.loads(response)
    except Exception as e:
        LOG.error(f"JSON parse failed: {e}\nRAW RESPONSE:\n{response}")
        return None

@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def print_chunks(data, input_key='chunk'):
    assert isinstance(data, dict)
    # print(data[input_key])
    return data



@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def chunk_to_qa(data, input_key='chunk'):
    assert isinstance(data, dict)
    
    chunk = data.get(input_key, '')

    prompt = f'''
    根据下面的字段生成一个qa对：
    {chunk}

    仅输出 JSON：
    {{"query": "问题", "answer": "答案"}}
    '''
    model = OnlineChatModule()
    response = model(prompt)
    qa_pairs = extract_json(response)
    try:
        data['query'] = qa_pairs.get('query', '')
        data['answer'] = qa_pairs.get('answer', '')
    except:
        data['query'] = ''
        data['answer'] = ''
    return data


@data_register('data.Text2qa', rewrite_func='forward', _concurrency_mode='process')
def qa_scorer(data, input_key='chunk'):
    assert isinstance(data, dict)
    
    chunk = data.get(input_key, '')
    query = data.get('query', '')
    answer = data.get('answer', '')

    if not chunk or not query or not answer:
        data['score'] = 0
        return data

    prompt = f'''
        请根据下面 chunk, 对生成的 QA 对打分：

        1. 原文：{chunk}
        2. 问题：{query}
        3. 答案：{answer}

        评分规则：
        - 如果问题和答案严格基于 chunk 内容，输出 1
        - 如果存在编造、无关、错误理解，输出 0

        仅输出 JSON：
        {{"score": 0 or 1}}
    '''
    model = OnlineChatModule()
    response = model(prompt)
    res = extract_json(response)
    try:
        score = res['score']
        score = int(score)
    except:
        score = 0

    data['score'] = score
    return data


@data_register('data.Text2qa', rewrite_func='forward_batch_input', _concurrency_mode='process')
def score_filter(data, threshold = 1):
    keep = []
    drop = []
    for i in data:
        if i['score'] >= threshold:
            keep.append(i)
        else:
            drop.append(i)
    return keep, drop
