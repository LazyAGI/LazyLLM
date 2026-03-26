from ..base_data import data_register
from lazyllm.tools.rag import MineruPDFReader
from lazyllm import LOG, TrainableModule
from lazyllm.components.formatter import JsonFormatter
from lazyllm.components.formatter import encode_query_with_filepaths
import os
import re
import io
from lazyllm.thirdparty import PIL
import requests

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'

Pdf2Qa = data_register.new_group('pdf2Qa')


class PdfProcessor(Pdf2Qa):
    def __init__(
        self,
        input_key='pdf_path',
        output_key='chunk',
        reader_url=None,
        use_cache=False,
        image_output_folder='./pdf_images',
        image_key='image_path',
        image_size=(336, 336),
        max_chunk_chars=1500,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        if not reader_url:
            raise ValueError('You must pass in a mineru url.')
        self.input_key = input_key
        self.output_key = output_key
        self.use_cache = use_cache
        self.reader = MineruPDFReader(
            url=reader_url,
            upload_mode=True,
        )

        self.base_url = reader_url.rstrip('/')
        self.pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        self.downloaded = set()
        self.image_key = image_key
        self.image_output_folder = image_output_folder
        self.image_size = image_size
        self.max_chunk_chars = max_chunk_chars

    def _merge_chunks(self, docs):
        merged = []
        buffer = ''

        for node in docs:
            text = node.text.strip()
            if not text:
                continue

            if len(buffer) + len(text) <= self.max_chunk_chars:
                buffer += '\n' + text if buffer else text
            else:
                if buffer:
                    merged.append(buffer)
                buffer = text

        if buffer:
            merged.append(buffer)
        return merged

    def _download_image(self, path):
        os.makedirs(self.image_output_folder, exist_ok=True)

        def letterbox_resize(img, size=336):
            w, h = img.size

            scale = size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), PIL.Image.BILINEAR)
            new_img = PIL.Image.new('RGB', (size, size), (0, 0, 0))
            paste_x = (size - new_w) // 2
            paste_y = (size - new_h) // 2

            new_img.paste(img, (paste_x, paste_y))

            return new_img

        filename = os.path.basename(path)
        save_path = os.path.join(self.image_output_folder, filename)

        if filename in self.downloaded:
            return save_path

        url = f'{self.base_url}/{path}'

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = PIL.Image.open(io.BytesIO(r.content)).convert('RGB')
                if isinstance(self.image_size, tuple):
                    img = img.resize(self.image_size)
                else:
                    img = letterbox_resize(img, self.image_size)
                img.save(save_path)

                self.downloaded.add(filename)
        except Exception as e:
            LOG.warning(f'Download failed: {url}, error: {e}')

        return save_path

    def _extract_images(self, text):
        return self.pattern.findall(text.strip())

    def forward(self, data, **kwargs):
        pdf_path = data.get(self.input_key)
        if not pdf_path:
            return []

        docs = self.reader._load_data(pdf_path, use_cache=self.use_cache)
        results = []
        merged_text = self._merge_chunks(docs)

        for text in merged_text:

            if 'images/' not in text:
                results.append(
                    {self.output_key: text, self.image_key: ''}
                )
                continue

            current_imgs = [
                p for p in self._extract_images(text)
                if p.startswith('images/')
            ]

            image_names = []

            for path in current_imgs:
                filename = self._download_image(path)
                image_names.append(filename)

            results.append({
                self.output_key: text,
                self.image_key: list(set(image_names)),
            })

        return results

@data_register('data.Pdf2QA', rewrite_func='forward')
def multi_features_filter(data, input_key, threshold):
    items = data.get(input_key, {})
    values = []
    for x in items.values():
        try:
            values.append(float(x))
        except Exception:
            LOG.warning(f'Could not convert value to float in multi_features_filter for item: {x}')
    if not values:
        return []
    avg = sum(values) / len(values)
    if avg < threshold:
        return []
    return None

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
        if not (query and answer):
            data[self.output_key] = 0
            return data
        if isinstance(img_path, str):
            img_path = [img_path] if img_path else []
        qa_payload = f'问题{query}; 答案{answer}'
        user_prompt = self.user_prompt or '''
请根据下面内容和图片(可以没有图片)对 QA 打分：

规则：
- 严格基于原文和图片 → 1
- 否则 → 0
'''
        user_prompt += f'''
原文：
{chunk}

QA对：
{qa_payload}'''
        res = self.model(encode_query_with_filepaths(user_prompt, img_path))
        data[self.output_key] = res.get(self.output_key, 0)
        return data

class ImageToVQA(Pdf2Qa):
    def __init__(self,
                 image_key='image_path',
                 query_key='query',
                 answer_key='answer',
                 model=None,
                 user_prompt=None,
                 context_key='context',
                 reference_key='reference',
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.image_key = image_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.user_prompt = user_prompt
        self.context_key = context_key
        self.reference_key = reference_key

        output_structure = f'''
输出格式要求：
{{
    '{self.query_key}': '问题',
    '{self.answer_key}': '答案'
}}
'''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model.share()

        self.model.prompt(dict(system=output_structure, drop_builtin_system=True))\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data: dict):
        assert self.image_key in data

        image_paths = data.get(self.image_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths] if image_paths else []

        default_query = '''
生成一个基于图像的中文问答对, 如果给出了问题或答案那么给答案输出完整的推理过程。如果输入的是纯文本，那么基于这个文本生成QA对。输出要具体，不能出现文中、作者、第几章这种模糊的问题。
'''
        query = self.user_prompt or default_query

        context = data.get(self.context_key, '')
        reference = data.get(self.reference_key, '')

        if context:
            query += f'\nContext：{context}'
        if reference:
            query += f'\nReference：{reference}'

        res = self.model(
            encode_query_with_filepaths(query, image_paths)
        )

        data[self.query_key] = res.get(self.query_key, '')
        data[self.answer_key] = res.get(self.answer_key, '')

        return data

@data_register('data.Pdf2QA', rewrite_func='forward')
def vqa_to_chat_format(
    data,
    image_key='image',
    query_key='query',
    answer_key='answer'
):
    image_path = data.get(image_key, '')
    query = data.get(query_key, '')
    answer = data.get(answer_key, '')

    if not (query and answer):
        return []

    if isinstance(image_path, str):
        image_path = [image_path] if image_path else []

    if image_path:
        chat_item = {
            'messages': [
                {
                    'role': 'user',
                    'content': f'<image>{query}'
                },
                {
                    'role': 'assistant',
                    'content': answer
                }
            ],
            'images': image_path
        }
    else:
        chat_item = {
            'messages': [
                {
                    'role': 'user',
                    'content': query
                },
                {
                    'role': 'assistant',
                    'content': answer
                }
            ]
        }

    return chat_item

@data_register('data.Pdf2QA', rewrite_func='forward')
def resize_image_inplace(
    data,
    image_key='image',
    size=(336, 336)
):
    image_paths = data.get(image_key, '')

    if not image_paths:
        return None

    if isinstance(image_paths, str):
        image_paths = [image_paths] if image_paths else []

    for img_path in image_paths:
        if not img_path or not os.path.exists(img_path):
            continue

        try:
            img = PIL.Image.open(img_path).convert('RGB')
            img = img.resize(size, PIL.Image.BICUBIC)
            img.save(img_path)

        except Exception as e:
            LOG.warning(f'Error processing {img_path}: {e}')

    return None
