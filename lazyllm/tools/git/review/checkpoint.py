# Copyright (c) 2026 LazyAGI. All rights reserved.
import json
import os
import re
from typing import Any, Dict, Optional

import lazyllm


def _load_cache(cache_path: Optional[str], key: str) -> Optional[str]:
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(key) if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(cache_path: Optional[str], key: str, value: str) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        data: Dict[str, Any] = {}
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        data[key] = value
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _save_cache_multi(cache_path: Optional[str], entries: Dict[str, Any]) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        data: Dict[str, Any] = {}
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
        data.update(entries)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class _ReviewCheckpoint:
    _KEYS = ('arch_doc', 'review_spec', 'r2_shared_context', 'r1', 'r2', 'r3', 'final')

    def __init__(self, path: str) -> None:
        self._path = path
        self._data: Dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                lazyllm.LOG.info(f'Loaded checkpoint from {path}')
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def save(self, key: str, value: Any) -> None:
        self._data[key] = value
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            lazyllm.LOG.warning(f'Failed to write checkpoint: {e}')

    def clear(self) -> None:
        self._data = {}
        try:
            if os.path.isfile(self._path):
                os.remove(self._path)
        except OSError:
            pass

    @staticmethod
    def default_path(pr_number: int, repo: str) -> str:
        safe_repo = re.sub(r'[^a-zA-Z0-9_-]', '_', repo)
        base = os.path.join(os.path.expanduser(lazyllm.config['home']), 'review', safe_repo)
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f'pr{pr_number}.json')

    @staticmethod
    def review_cache_dir() -> str:
        home = os.path.expanduser(lazyllm.config['home'])
        d = os.path.join(home, 'review')
        os.makedirs(d, exist_ok=True)
        return d
