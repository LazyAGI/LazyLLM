import re
from functools import wraps
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from lazyllm import LOG


class ModelTypeClassifier:

    SUPPORTED_TYPES = {
        'stt', 'tts', 'sd', 'ocr', 'vlm', 'llm',
        'cross_modal_embed', 'embed', 'rerank'
    }
    _rules: List[Dict] = []

    @classmethod
    def _register(cls, priority: Optional[int] = None, name: Optional[str] = None):

        def decorator(func: Callable[[str], Optional[str]]) -> Callable[[str], Optional[str]]:
            rule = {
                'func': func,
                'name': name or func.__name__,
                'priority': priority if priority is not None else len(cls._rules)
            }

            if priority is not None:
                insert_index = 0
                for i, existing_rule in enumerate(cls._rules):
                    if existing_rule['priority'] <= priority:
                        insert_index = i + 1
                    else:
                        break
                cls._rules.insert(insert_index, rule)
            else:
                cls._rules.append(rule)

            @wraps(func)
            def wrapper(model_name: str) -> Optional[str]:
                return func(model_name)

            return wrapper
        return decorator

    def __call__(self, model_name: str) -> str:
        return self._infer(model_name)

    def _infer(self, model_name: str) -> str:

        if not model_name or not isinstance(model_name, str):
            raise ValueError('Parameter model_name must be a non-empty string.')

        for rule in self._rules:
            try:
                result = rule['func'](model_name)
                if result is not None:
                    if result not in self.SUPPORTED_TYPES:
                        LOG.warning(f'Rule {rule["name"]} returned unsupported model type: {result}.')
                        continue
                    LOG.info(f'Rule {rule["name"]} matched for model: {model_name}, classified as type: {result}.')
                    return result
            except Exception as e:
                LOG.warning(f'Rule {rule["name"]} execution error: {e}')
                continue

        LOG.warning(f'Cannot classify model type for: {model_name}. Defaulting to "llm" instead.')
        return 'llm'

    def _get_rules_info(self) -> List[Dict]:
        return [
            {
                'name': rule['name'],
                'priority': rule['priority'],
                'func_name': rule['func'].__name__
            }
            for rule in self._rules
        ]

    def __str__(self):
        rules_info = self._get_rules_info()
        return '\n'.join([f"{i+1}. {info['name']} (Priority: {info['priority']})"
                          for i, info in enumerate(rules_info)])

# Special Model Matching

special_models = {
    'llm': {'minicpm-2b-dpo-bf16'},
    'vlm': {
        'aria', 'r-4b', 'step3', 'deepseek-ocr', 'emu3-chat-hf', 'granite-speech-3.3-8b', 'idefics3-8b-llama3',
        'phi-4-multimodal-instruct', 'qwen2-audio-7b-instruct', 'mistral-small-3.1-24b-instruct-2503',
        'llava-next-video-7b-hf'},
    'stt': {},
    'tts': {'bark', 'kokoro-82m', 'csm-1b', 'chatterbox', 'fish-speech-1.5', 'dia-1.6b', 'vibevoice-1.5b'},
    'sd': {'step1x-edit-v1p2-preview', 'idm-vton', 'mochi-1-preview', 'instruct-pix2pix'},
    'ocr': {},
    'embed': {'embeddinggemma-300m', 'all-roberta-large-v1'},
    'rerank': {'gte-multilingual-reranker-base'},
    'cross_modal_embed': {'vlm2vec-full', 'siglip-base-patch16-224'},
}

@ModelTypeClassifier._register(name="Special Model Matching")
def special_model_rule(model_name: str) -> Optional[str]:
    model_name = model_name.split('/')[-1].lower()
    for model_type, model_set in special_models.items():
        if model_name in model_set:
            LOG.debug(f'Special Model matched: {model_name} for '
                      f'model type: {model_type} on model: {model_name}')
            return model_type
    return None

# Feature Keyword Matching

keywords = {
    'llm': [],
    'vlm': [
        'bee', 'vl', 'chameleon', 'fuyu', 'gemma-3', 'omni', 'tarsier', 'cogagent', 'pixtral',
        'qvq', 'molmo', 'ovis', 'paligemma', 'skywork', 'lightonocr', 'siglip', 'midashenglm',
        'llama-4', 'internvideo'],
    'stt': ['voxtral', 'voice'],
    'tts': ['music'],
    'sd': [
        'sd', 'wan', 'i2i', 'i2v', 't2v', 't2i', 'cog', 'video', 'animate', 'qwen-image', 'flux',
        'diffusion', 'controlnet', 'sdxl', 'photo', 'image', 'journey'],
    'ocr': [],
    'embed': ['gte'],
    'rerank': ['ms-marco', 'quora-roberta'],
    'cross_modal_embed': [],
}

@ModelTypeClassifier._register(name="Feature Keyword Matching")
def feature_keyword_rule(model_name: str) -> Optional[str]:
    model_name = model_name.split('/')[-1].lower()
    for model_type, keys in keywords.items():
        for key in keys:
            if key in model_name:
                LOG.debug(f'Feature keyword matched: {key} for '
                          f'model type: {model_type} on model: {model_name}')
                return model_type
    return None


# Regular Expression Matching

pattern_dict = OrderedDict([
    ('stt', [
        r'.*stt.*', r'.*asr.*',
        r'.*whisper.*', r'.*wav2vec.*'
    ]),
    ('tts', [
        r'.*tts.*', r'.*text.*speech.*',
        r'.*vits.*', r'.*tacotron.*'
    ]),
    ('sd', [
        r'.*stable.*diffusion.*', r'.*sd.*[0-9].*',
        r'.*dalle.*', r'.*midjourney.*'
    ]),
    ('ocr', [
        r'.*ocr.*', r'.*optical.*character.*',
        r'.*paddleocr.*', r'.*easyocr.*'
    ]),
    ('rerank', [
        r'.*rerank.*', r'.*colbert.*', r'.*bge.*rerank.*'
    ]),
    ('cross_modal_embed', [
        r'.*clip.*', r'.*align.*', r'.*bridge.*',
        r'.*multimodal.*', r'.*cross.*modal.*', r'.*e5.*v.*'
    ]),
    ('embed', [
        r'.*embed.*', r'.*bge.*', r'.*e5.*',
        r'.*sentence.*transformers.*'
    ]),
    ('vlm', [
        r'.*visual.*', r'.*vision.*',
        r'.*blip.*', r'.*llava.*', r'.*minigpt.*',
        r'.*glm.*v.*', r'.*minicpm.*[vo].*', r'.*intern-s1.*'
    ]),
    ('llm', [
        r'.*gpt.*', r'.*.llama.*',
        r'.*chat.*', r'.*instruct.*', r'.*text.*generation.*'
    ]),
])

@ModelTypeClassifier._register(name="Regular Expression Matching")
def regular_rule(model_name: str) -> Optional[str]:
    model_name_lower = model_name.lower()

    for model_type, patterns in pattern_dict.items():
        for pattern in patterns:
            if re.match(pattern, model_name_lower):
                LOG.debug(f'Regular rule matched: {pattern} for '
                          f'model type: {model_type} on model: {model_name}')
                return model_type
    return None

infer_model_type = ModelTypeClassifier()
