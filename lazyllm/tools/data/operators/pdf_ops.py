from ..base_data import data_register
from lazyllm.tools.rag import MineruPDFReader
from lazyllm import LOG, TrainableModule
from lazyllm.components.formatter import JsonFormatter
from lazyllm.components.formatter import encode_query_with_filepaths
import os
import re

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'

Pdf2Qa = data_register.new_group('pdf2Qa')


class Pdf2Md(Pdf2Qa):
    def __init__(
        self,
        input_key='pdf_path',
        output_key='docs',
        reader_url=None,
        upload_mode=True,
        use_cache=False,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        if not reader_url:
            raise ValueError('You must pass in a reader_url.')
        self.input_key = input_key
        self.output_key = output_key
        self.use_cache = use_cache
        self.reader = MineruPDFReader(
            url=reader_url,
            upload_mode=upload_mode,
        )

    def forward(self, data):
        pdf_path = data.get(self.input_key)
        if not pdf_path:
            return None
        try:
            docs = self.reader(file=pdf_path, use_cache=self.use_cache)
            node = docs[0]
            content = getattr(node, 'content', None)
            return [{'content': c} for c in content]
        except Exception as e:
            LOG.warning(f'PDF read failed: {e}')
            return data


@data_register('data.Pdf2QA', rewrite_func='forward')
def multi_features_filter(data, input_key, threshold):
    items = data[input_key]
    values = []
    for x in items.values():
        try:
            values.append(float(x))
        except Exception:
            pass
    if not values:
        return []
    avg = sum(values) / len(values)
    if avg < threshold:
        return []
    return None


class PdfChunkToQA(Pdf2Qa):
    def __init__(
        self,
        input_key='chunk',
        query_key='query',
        answer_key='answer',
        model=None,
        user_prompt=None,
        mineru_api=None,
        image_key='image_path',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.user_prompt = user_prompt
        self.mineru_api = mineru_api or ''
        self.image_key = image_key
        output_structure = f'''
输出格式要求：
{{
    '{self.query_key}': '生成的问题',
    '{self.answer_key}': '答案'
}}
'''
        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.share()
        self.model.prompt(output_structure).formatter(JsonFormatter()).start()

    def _extract_images(self, text):
        pattern = r'!\[.*?\]\((images/[^)]+)\)'
        return re.findall(pattern, text or '')

    def forward(self, data: dict):
        assert self.input_key in data
        chunk = data.get(self.input_key, '')
        if not chunk:
            data[self.query_key] = ''
            data[self.answer_key] = ''
            return data
        image_rel_paths = self._extract_images(chunk)
        if image_rel_paths:
            base_dir = os.path.join('lazyllm', 'tools', 'data', 'operators', 'imgs')
            os.makedirs(base_dir, exist_ok=True)
            local_paths = []
            for rel_path in image_rel_paths:
                filename = os.path.basename(rel_path)
                local_path = os.path.join(base_dir, filename)
                local_paths.append(local_path)
                src_path = os.path.join(self.mineru_api, rel_path) if self.mineru_api else rel_path
                if src_path.startswith('http'):
                    import requests
                    r = requests.get(src_path)
                    with open(local_path, 'wb') as f:
                        f.write(r.content)
                else:
                    import shutil
                    shutil.copy(src_path, local_path)
            context = re.sub(r'!\[.*?\]\(images/[^)]+\)', '', chunk)
            query = context if context else 'Generate one QA pair based on the image.'
            # save the img from mineru server to local
            out = self.model(encode_query_with_filepaths(query, local_paths))
            data[self.query_key] = out.get(self.query_key, '')
            data[self.answer_key] = out.get(self.answer_key, '')
            data[self.image_key] = local_paths[0]
            return data
        user_prompt = self.user_prompt or '根据下面文本生成一个 QA 对：\n'
        inp = f'{user_prompt}\n{chunk}'
        qa = self.model(inp)
        data[self.query_key] = qa.get(self.query_key, '')
        data[self.answer_key] = qa.get(self.answer_key, '')
        return data


class PdfQAScorer(Pdf2Qa):
    def __init__(
        self,
        input_key='chunk',
        output_key='score',
        query_key='query',
        answer_key='answer',
        model=None,
        user_prompt=None,
        image_key='image_path',
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.user_prompt = user_prompt
        self.image_key = image_key
        output_structure = f'''
输出格式要求：
{{
    '{self.output_key}': 0
}}
'''
        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.share()
        self.model.prompt(output_structure).formatter(JsonFormatter()).start()

    def forward(self, data: dict):
        assert self.input_key in data
        assert self.query_key in data
        assert self.answer_key in data
        chunk = data.get(self.input_key, '')
        query = data.get(self.query_key, '')
        answer = data.get(self.answer_key, '')
        img_path = data.get(self.image_key, '')
        if not (chunk and query and answer and img_path):
            data[self.output_key] = 0
            return data
        qa_payload = f'问题{query}; 答案{answer}'
        user_prompt = self.user_prompt or f'''
请根据下面内容和图片(可以没有图片)对 QA 打分：

原文：
{chunk}

图片路径：
{img_path}

{qa_payload}

规则：
- 严格基于原文和图片 → 1
- 否则 → 0
'''
        res = self.model(encode_query_with_filepaths(user_prompt, [img_path]))
        data[self.output_key] = res.get(self.output_key, 0)
        return data
