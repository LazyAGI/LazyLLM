import os
import json
import threading

from lazyllm import LOG

class PromptLibrary:
    _prompts: dict = {}
    _prompts_paths = [
        ('awesome-chatgpt-prompts-zh.json', 'zh'),
        ('prompts.chat.json', 'en'),
    ]
    _loaded = False
    _load_lock = threading.Lock()
    _default_lang = 'zh'

    def __init__(self, lang=None):
        if not PromptLibrary._loaded:
            with PromptLibrary._load_lock:
                if not PromptLibrary._loaded:  # Double-checked locking
                    self._build_library()
                    PromptLibrary._loaded = True

        self.lang = lang
        if self.lang and self.lang not in self.supported_langs:
            LOG.warning(f'Language "{self.lang}" passed to init is not supported. Supported: {self.supported_langs}')

    @property
    def supported_langs(self):
        return list(self._prompts.keys())

    def _load_prompts(self, path, lang):
        if lang not in self._prompts:
            self._prompts[lang] = {}

        if not os.path.exists(path):
            LOG.warning(f'PromptLibrary file not found: {path}')
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
            abs_path = os.path.join(base_path, 'prompts_lib', rel_path)
            self._load_prompts(abs_path, lang)
            # Show summary of loaded prompts
            if lang in self._prompts:
                LOG.info(f'After loading {rel_path}, total prompts for lang "{lang}": {len(self._prompts[lang])}')

    def __call__(self, act: str, lang=None) -> str:
        return self.get_prompt(act, lang)

    def get_prompt(self, act: str, lang=None) -> str:
        lang = lang or self.lang or self._default_lang
        if lang not in self._prompts:
            LOG.warning(f'Language "{lang}" not supported. Supported: {self.supported_langs}')
            return ''
        prompt = self._prompts[lang].get(act)
        if prompt is None:
            LOG.warning(f'Prompt for act "{act}" not found in library (lang: {lang}).')
            return ''
        return prompt

    def get_all_acts(self, lang=None) -> list:
        if lang is None:
            if self.lang:
                lang = self.lang
                LOG.info(f'get_all_acts: no lang passed, using instance language "{self.lang}"')
            else:
                lang = self._default_lang
                LOG.info(f'get_all_acts: no lang passed, using default language "{self._default_lang}"')

        if lang not in self._prompts:
            LOG.warning(f'Language "{lang}" not supported. Supported: {self.supported_langs}')
            return []
        return list(self._prompts[lang].keys())
