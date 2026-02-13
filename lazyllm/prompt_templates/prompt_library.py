import os
import json
import copy
import threading


from lazyllm import LOG, ChatPrompter
from lazyllm.common.registry import LazyLLMRegisterMetaClass


class LazyLLMPromptLibraryBase(metaclass=LazyLLMRegisterMetaClass):
    _prompts = {}
    _default_lang = 'zh'

    def __init__(self, lang=None):
        self.lang = lang
        if self.lang and self.lang not in self.supported_langs:
            LOG.warning(f'Language "{self.lang}" passed to init is not supported. Supported: {self.supported_langs}')

    @property
    def supported_langs(self):
        return list(self._prompts.keys())

    def get_prompt(self, key: str, lang=None):
        lang = lang or self.lang or self._default_lang
        if lang not in self._prompts:
            raise ValueError(f'Language "{lang}" not supported. Supported: {self.supported_langs}')
        prompt = self._prompts[lang].get(key)
        if prompt is None:
            raise ValueError(f'Prompt for key "{key}" not found in library (lang: {lang}).')
        return prompt

    def get_all_keys(self, lang=None) -> list:
        if lang is None:
            if self.lang:
                lang = self.lang
                LOG.info(f'get_all_keys: no lang passed, using instance language "{self.lang}"')
            else:
                lang = self._default_lang
                LOG.info(f'get_all_keys: no lang passed, using default language "{self._default_lang}"')

        if lang not in self._prompts:
            LOG.warning(f'Language "{lang}" not supported. Supported: {self.supported_langs}')
            return []
        return list(self._prompts[lang].keys())

class ActorPrompt(LazyLLMPromptLibraryBase):
    _prompts: dict = {}
    _prompts_paths = [
        ('awesome-chatgpt-prompts-zh.json', 'zh'),
        ('prompts.chat.json', 'en'),
    ]
    _loaded = False
    _load_lock = threading.Lock()
    _default_lang = 'zh'

    def __init__(self, lang=None):
        if not ActorPrompt._loaded:
            with ActorPrompt._load_lock:
                if not ActorPrompt._loaded:  # Double-checked locking
                    self._build_library()
                    ActorPrompt._loaded = True

        super().__init__(lang)

    def _load_prompts(self, path, lang):
        if lang not in self._prompts:
            self._prompts[lang] = {}

        if not os.path.exists(path):
            LOG.warning(f'ActorPrompt file not found: {path}')
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)

            count = 0
            for item in prompts:
                act = item.get('act')
                prompt = item.get('prompt')
                if act and prompt:
                    if act in self._prompts[lang]:
                        LOG.warning(
                            f'Duplicate act "{act}" found in {path} for '
                            f'lang {lang}. Overwriting existing prompt.')
                    self._prompts[lang][act] = prompt
                    count += 1
            LOG.debug(f'Loaded {count} prompts from {path} for lang {lang}')
        except json.JSONDecodeError:
            LOG.error(f'Failed to decode JSON from {path}')
        except Exception as e:
            LOG.error(f'Error loading prompts from {path}: {e}')

    def _build_library(self):
        base_path = os.path.dirname(__file__)
        for rel_path, lang in self._prompts_paths:
            abs_path = os.path.join(base_path, 'prompts_actor', rel_path)
            self._load_prompts(abs_path, lang)
            # Show summary of loaded prompts
            if lang in self._prompts:
                LOG.info(f'After loading {rel_path}, total prompts for lang "{lang}": {len(self._prompts[lang])}')

    def __call__(self, act: str, lang=None, return_raw=False) -> str:
        prompt = self.get_prompt(act, lang)  # Get the raw prompt string
        if prompt is None: return ''
        if return_raw:
            return prompt
        return ChatPrompter(prompt)

    def get_all_acts(self, lang=None) -> list:
        return self.get_all_keys(lang)

class DataPrompt(LazyLLMPromptLibraryBase):
    _prompts: dict = {}
    _default_lang = 'zh'

    def __init__(self, lang=None):
        super().__init__(lang)

    def __call__(self, key: str, lang=None, return_raw=False) -> str:
        prompt = self.get_prompt(key, lang)  # Get the raw prompt dict
        if prompt is None: return None
        if return_raw:
            return prompt  # Dict

        prompt = copy.deepcopy(prompt)
        tools = prompt.get('tools')
        history = prompt.get('history')
        extra_keys = prompt.get('extra_keys')
        system = prompt.get('system', '')
        user = prompt.get('user', '')

        return ChatPrompter({'system': system, 'user': user}, tools=tools, history=history, extra_keys=extra_keys)

    @classmethod
    def add_prompt(cls, act, system_prompt=None, user_prompt=None, tools=None, history=None, extra_keys=None, lang='zh'):
        assert system_prompt or user_prompt, 'At least one of system_prompt or user_prompt must be provided'
        if lang not in cls._prompts:
            cls._prompts[lang] = {}

        if act in cls._prompts[lang]:
            LOG.warning(f'Duplicate act "{act}" found in DataPrompt for lang {lang}. Overwriting.')

        cls._prompts[lang][act] = {
            'system': system_prompt if system_prompt is not None else '',
            'user': user_prompt if user_prompt is not None else '',
            'tools': tools,
            'history': history,
            'extra_keys': extra_keys
        }
