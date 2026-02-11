import os
import re
import hashlib
from typing import Set, Optional
from ..base_data import data_register
from lazyllm import LOG
from lazyllm.thirdparty import PIL
from lazyllm.components.formatter import JsonFormatter
from lazyllm.components.formatter import encode_query_with_filepaths

PT = data_register.new_group('pt')
PT_MM = data_register.new_group('pt_mm')
PT_TXT = data_register.new_group('pt_text')


def _normalize_image_paths(image_path) -> list:
    if image_path is None or image_path == '':
        return []
    if isinstance(image_path, str):
        return [image_path]
    return list(image_path)


@data_register('data.pt_mm', rewrite_func='forward', _concurrency_mode='process')
def resolution_filter(data, image_key='image_path', min_width=256, min_height=256,
                      max_width=4096, max_height=4096, input_key=None):
    assert isinstance(data, dict)
    if input_key:
        image_key = input_key
    paths = _normalize_image_paths(data.get(image_key, ''))
    if not paths:
        return []
    valid_paths = []
    try:
        for image_path in paths:
            if not os.path.exists(image_path):
                LOG.warning(f'Image path not found or invalid: {image_path}')
                continue
            with PIL.Image.open(image_path) as img:
                width, height = img.size
                if width < min_width or height < min_height:
                    continue
                if width > max_width or height > max_height:
                    continue
                valid_paths.append(image_path)
        if not valid_paths:
            return []
        data[image_key] = valid_paths
        return data
    except Exception as e:
        LOG.warning(f'Failed to check image resolution: {e}')
        return []


@data_register('data.pt_mm', rewrite_func='forward', _concurrency_mode='process')
def resolution_resize(data, image_key='image_path', max_side=1024, input_key=None, inplace=True):
    assert isinstance(data, dict)
    if input_key:
        image_key = input_key
    paths = _normalize_image_paths(data.get(image_key, ''))
    if not paths:
        return []
    valid_paths = []
    try:
        for image_path in paths:
            if not os.path.exists(image_path):
                LOG.warning(f'Image path not found or invalid: {image_path}')
                continue
            with PIL.Image.open(image_path) as img:
                img.load()
                w, h = img.size
                if max(w, h) <= max_side:
                    valid_paths.append(image_path)
                    continue
                scale = max_side / max(w, h)
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                if new_w < 1 or new_h < 1:
                    continue
                resample = getattr(
                    getattr(PIL.Image, 'Resampling', None), 'LANCZOS', PIL.Image.LANCZOS
                )
                out = img.resize((new_w, new_h), resample)
                if inplace:
                    save_path = image_path
                else:
                    base, ext = os.path.splitext(image_path)
                    save_path = f'{base}_resized{ext}'
                out.save(save_path, quality=95)
                valid_paths.append(save_path)
        if not valid_paths:
            return []
        data[image_key] = valid_paths
        return data
    except Exception as e:
        LOG.warning(f'Failed to resize image resolution: {e}')
        return []


@data_register('data.pt_mm', rewrite_func='forward', _concurrency_mode='process')
def integrity_check(data, image_key='image_path', input_key=None):
    assert isinstance(data, dict)
    if input_key:
        image_key = input_key
    paths = _normalize_image_paths(data.get(image_key, ''))
    if not paths:
        return []
    valid_paths = []
    try:
        for image_path in paths:
            if not os.path.exists(image_path):
                LOG.warning(f'Image path not found: {image_path}')
                continue
            try:
                with PIL.Image.open(image_path) as img:
                    img.verify()
                if os.path.getsize(image_path) == 0:
                    continue
                valid_paths.append(image_path)
            except Exception as e:
                LOG.warning(f'Failed to check file integrity for {image_path}: {e}')
                continue
        if not valid_paths:
            return []
        data[image_key] = valid_paths
        return data
    except Exception as e:
        LOG.warning(f'Failed to check file integrity: {e}')
        return []


class TextRelevanceFilter(PT_MM):
    DEFAULT_PROMPT = (
        'You are an image-text relevance judge.\n'
        'Given ONE image and ONE piece of text, you must output STRICT JSON and nothing else.\n'
        'JSON schema:\n'
        '{\n'
        '  "relevance": 0.0,  // float in [0, 1]\n'
        '  "reason": ""      // short string\n'
        '}\n'
        'Rules:\n'
        '- relevance=1 means fully relevant; relevance=0 means irrelevant.\n'
        '- Do not output markdown, code fences, or any extra words outside JSON.\n'
    )

    def __init__(self, vlm, image_key='image_path', text_key='text', threshold=0.6,
                 prompt: Optional[str] = None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        if vlm is None:
            raise ValueError('TextRelevanceFilter requires vlm (vision-language model).')
        self.image_key = image_key
        self.text_key = text_key
        self.threshold = threshold
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._judge = vlm.share().prompt(self.prompt).formatter(JsonFormatter())

    def _calc_relevance(self, image_path, text):
        if not text or not image_path or not os.path.exists(image_path):
            return 0.0
        try:
            out = self._judge(encode_query_with_filepaths(text, [image_path]))
            v = out.get('relevance', 0.0) if isinstance(out, dict) else 0.0
            v = max(0.0, min(1.0, float(v))) if isinstance(v, (int, float)) else 0.0
            return v
        except Exception as e:
            LOG.warning(f'VLM relevance failed: {e}')
            return 0.0

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        paths = _normalize_image_paths(data.get(self.image_key, ''))
        text = data.get(self.text_key, '')
        if not paths or not text:
            return []
        try:
            scores = [self._calc_relevance(p, text) for p in paths]
            mean_relevance = sum(scores) / len(scores) if scores else 0.0
            if mean_relevance < self.threshold:
                return []
            valid_paths = [p for p, s in zip(paths, scores) if s >= self.threshold]
            if not valid_paths:
                return []
            data[self.image_key] = valid_paths
            data['image_text_relevance'] = mean_relevance
            return data
        except Exception as e:
            LOG.warning(f'Failed to calculate image-text relevance: {e}')
            return []


class ImageDedup(PT_MM):
    def __init__(self, image_key='image_path', hash_method='md5', **kwargs):
        super().__init__(**kwargs)
        self.image_key = image_key
        self.hash_method = hash_method

    def _calc_hash(self, image_path):
        try:
            if not os.path.exists(image_path):
                return None
            hash_obj = hashlib.new(self.hash_method)
            with open(image_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            LOG.warning(f'Failed to calculate hash for {image_path}: {e}')
            return None

    def forward_batch_input(self, data, **kwargs):
        assert isinstance(data, list)
        seen_hashes: Set[str] = set()
        deduplicated_data = []
        for item in data:
            assert isinstance(item, dict)
            paths = _normalize_image_paths(item.get(self.image_key, ''))
            if not paths:
                continue
            image_hash = self._calc_hash(paths[0])
            if image_hash is None:
                continue
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)
            deduplicated_data.append(item)
        return deduplicated_data


class VQAGenerator(PT_MM):
    DEFAULT_PROMPT = (
        'Generate Visual Question Answering (VQA) pairs from the given context and image(s). '
        'Output JSON only. Do not output any other irrelevant content.\n'
        '{\n'
        '  "qa_pairs": [\n'
        '    {"query": "", "answer": ""}\n'
        '  ]\n'
        '}\n'
        'Each item in qa_pairs has query (question) and answer. '
        'All questions should be answerable from the context and image.'
    )

    def __init__(self, vlm, image_key='image_path', context_key='context', num_qa=5,
                 prompt: Optional[str] = None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        if vlm is None:
            raise ValueError('VQAGenerator requires vlm (vision-language model).')
        self.image_key = image_key
        self.context_key = context_key
        self.num_qa = num_qa
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._generator = vlm.share().prompt(prompt or self.DEFAULT_PROMPT).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        paths = _normalize_image_paths(data.get(self.image_key, ''))
        context = data.get(self.context_key, '')
        if not paths or not context:
            return []
        try:
            query = f'Context: {context}\n\nGenerate {self.num_qa} QA pairs based on the context and image(s).'
            out = self._generator(encode_query_with_filepaths(query, paths))
            if not isinstance(out, dict):
                data['qa_pairs'] = []
                return data
            raw = out.get('qa_pairs', [])
            if not isinstance(raw, list):
                data['qa_pairs'] = []
                return data
            qa_pairs = []
            for item in raw:
                if isinstance(item, dict) and 'query' in item and 'answer' in item:
                    qa_pairs.append({'query': str(item['query']), 'answer': str(item['answer'])})
                elif isinstance(item, dict) and 'question' in item:
                    qa_pairs.append({
                        'query': str(item.get('question', item.get('query', ''))),
                        'answer': str(item.get('answer', item.get('ans', ''))),
                    })
            data['qa_pairs'] = qa_pairs
            return data
        except Exception as e:
            LOG.warning(f'VQA generation failed: {e}')
            return []


class Phi4QAGenerator(PT_TXT):
    DEFAULT_PROMPT = (
        'Convert the given context (text and/or images) into pretraining-format multi-turn Q&A dialogue data. '
        'Output JSON only. Do not output any other irrelevant content.\n'
        '{\n'
        '  "qa_pairs": [\n'
        '    {"query": "", "answer": ""}\n'
        '  ]\n'
        '}\n'
        'Each item has query (question) and answer. Generate natural, instructional Q&A suitable for LM pretraining.'
    )

    def __init__(self, vlm, image_key='image_path', context_key='context', num_qa=5,
                 prompt: Optional[str] = None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        if vlm is None:
            raise ValueError('Phi4QAGenerator requires vlm (vision-language model).')
        self.image_key = image_key
        self.context_key = context_key
        self.num_qa = num_qa
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._generator = vlm.share().prompt(self.prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        context = data.get(self.context_key, '')
        if not context:
            return []
        paths = _normalize_image_paths(data.get(self.image_key, ''))
        try:
            query = f'Context:\n{context}\n\nGenerate {self.num_qa} pretraining-format Q&A pairs (phi-4 style).'
            inputs = encode_query_with_filepaths(query, paths) if paths else query
            out = self._generator(inputs)
            if not isinstance(out, dict):
                data['qa_pairs'] = []
                return data
            raw = out.get('qa_pairs', [])
            if not isinstance(raw, list):
                data['qa_pairs'] = []
                return data
            qa_pairs = []
            for item in raw:
                if isinstance(item, dict) and 'query' in item and 'answer' in item:
                    qa_pairs.append({'query': str(item['query']), 'answer': str(item['answer'])})
                elif isinstance(item, dict) and 'question' in item:
                    qa_pairs.append({
                        'query': str(item.get('question', item.get('query', ''))),
                        'answer': str(item.get('answer', item.get('ans', ''))),
                    })
            data['qa_pairs'] = qa_pairs
            return data
        except Exception as e:
            LOG.warning(f'Phi4 Q&A generation failed: {e}')
            return []


class VQAScorer(PT_MM):
    DEFAULT_PROMPT = (
        'Rate the visual quality of this image and output JSON only. '
        'Do not output any other irrelevant content.\n'
        '{\n'
        '  "score": 0.0,\n'
        '  "clarity": 0.0,\n'
        '  "composition": 0.0,\n'
        '  "reason": ""\n'
        '}\n'
        'score: overall [0, 1]; clarity: sharpness [0, 1]; composition [0, 1]. All floats.'
    )

    def __init__(self, vlm, image_key='image_path',
                 prompt: Optional[str] = None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        if vlm is None:
            raise ValueError('VQAScorer requires vlm (vision-language model).')
        self.image_key = image_key
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._scorer = vlm.share().prompt(self.prompt).formatter(JsonFormatter())

    def _clamp_score(self, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    def _calc_quality(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return 0.0, {}
        try:
            query = 'Rate the visual quality of this image.'
            out = self._scorer(encode_query_with_filepaths(query, [image_path]))
            if not isinstance(out, dict):
                return 0.0, {}
            score = self._clamp_score(out.get('score', out.get('overall', 0.0)))
            return score, out
        except Exception as e:
            LOG.warning(f'VLM quality scoring failed: {e}')
            return 0.0, {}

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        paths = _normalize_image_paths(data.get(self.image_key, ''))
        if not paths:
            return []
        try:
            results = [self._calc_quality(p) for p in paths]
            score_dicts = [r[1] for r in results]
            n = len(score_dicts)
            data['quality_score'] = {
                'score': sum(
                    self._clamp_score(d.get('score', d.get('overall', 0)))
                    for d in score_dicts
                ) / n if n else 0.0,
                'clarity': sum(self._clamp_score(d.get('clarity', 0)) for d in score_dicts) / n if n else 0.0,
                'composition': sum(self._clamp_score(d.get('composition', 0)) for d in score_dicts) / n if n else 0.0,
                'reason': '; '.join(str(d.get('reason', '')) for d in score_dicts if d.get('reason')),
            }
            return data
        except Exception as e:
            LOG.warning(f'Failed to score image quality: {e}')
            return []


class ContextQualFilter(PT):
    DEFAULT_PROMPT = (
        'Evaluate whether the given context (text and/or images) is suitable for generating QA pairs. '
        'Output JSON only. Do not output any other irrelevant content.\n'
        '{\n'
        '  "score": 0,\n'
        '  "reason": ""\n'
        '}\n'
        'score: MUST be 0 or 1 only. 1=suitable, 0=not suitable. Good context has sufficient info for Q&A.'
    )

    def __init__(self, vlm, context_key='context', image_key='image_path',
                 prompt: Optional[str] = None,
                 _concurrency_mode='thread', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        if vlm is None:
            raise ValueError('ContextQualFilter requires vlm (vision-language model).')
        self.context_key = context_key
        self.image_key = image_key
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._evaluator = vlm.share().prompt(self.prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        context = data.get(self.context_key, '')
        if not context:
            return []
        paths = _normalize_image_paths(data.get(self.image_key, ''))
        try:
            query = f'Context:\n{context}\n\nIs this context suitable for generating QA pairs?'
            inputs = encode_query_with_filepaths(query, paths) if paths else query
            out = self._evaluator(inputs)
            if not isinstance(out, dict):
                return []
            score = out.get('score', out.get('suitable', 0))
            try:
                score = int(float(score))
            except (TypeError, ValueError):
                score = 0
            if score != 1:
                return []
            return data
        except Exception as e:
            LOG.warning(f'Context qualification evaluation failed: {e}')
            return []


class GraphRetriever(PT_MM):
    def __init__(self, context_key='context', img_key='image_path', images_folder: Optional[str] = None,
                 _concurrency_mode='process', **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.context_key = context_key
        self.img_key = img_key
        self.images_folder = images_folder

    def _extract_img_paths(self, img_data) -> list:
        valid_paths = []

        def _from_str(s):
            matches = re.findall(r'!\[.*?\]\((.*?)\)', str(s))
            candidates = matches if matches else [str(s)] if s else []
            for p in candidates:
                if not p or not p.strip():
                    continue
                raw = os.path.join(self.images_folder, os.path.basename(p)) if self.images_folder else p
                full = os.path.abspath(raw)
                if os.path.exists(full):
                    valid_paths.append(full)

        if isinstance(img_data, list):
            for item in img_data:
                if isinstance(item, list):
                    for sub in item:
                        _from_str(sub)
                else:
                    _from_str(item)
        else:
            _from_str(img_data)
        return list(dict.fromkeys(valid_paths))

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        context = data.get(self.context_key, '')
        if isinstance(context, list):
            context = '\n\n'.join(str(c) for c in context)
        if isinstance(context, str):
            # Escape braces so context can be safely used in .format() templates downstream
            context = context.replace('{', '{{').replace('}', '}}')
        if not context or not context.strip():
            return []

        valid_paths = self._extract_img_paths(data.get(self.img_key, []))
        data['context'] = context.strip()
        data[self.img_key] = valid_paths
        return data
