# Copyright (c) 2026 LazyAGI. All rights reserved.
import enum
import json
import os
import re
from typing import Any, Dict, Optional

import lazyllm


class ReviewStage(enum.Enum):
    CLONE = 'clone'
    ARCH = 'arch'
    SPEC = 'spec'
    PR_SUMMARY = 'pr_summary'
    R1 = 'r1'
    R2 = 'r2'
    R3 = 'r3'
    R4_DOC = 'r4_doc'
    R4 = 'r4'
    FINAL = 'final'

    @staticmethod
    def ordered() -> list:
        return [
            ReviewStage.CLONE, ReviewStage.ARCH, ReviewStage.SPEC,
            ReviewStage.PR_SUMMARY, ReviewStage.R1, ReviewStage.R2,
            ReviewStage.R3, ReviewStage.R4_DOC, ReviewStage.R4, ReviewStage.FINAL,
        ]

    def index(self) -> int:
        return ReviewStage.ordered().index(self)

    def __le__(self, other: 'ReviewStage') -> bool:
        return self.index() <= other.index()

    def __lt__(self, other: 'ReviewStage') -> bool:
        return self.index() < other.index()


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
    # internal key prefix for stage completion markers
    _STAGE_DONE_PREFIX = '_stage_done_'

    def __init__(self, path: str, resume_from: Optional[ReviewStage] = None) -> None:
        self._path = path
        self._data: Dict[str, Any] = {}
        self._resume_from = resume_from
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                lazyllm.LOG.info(f'Loaded checkpoint from {path}')
            except (json.JSONDecodeError, OSError):
                self._data = {}
        if resume_from is not None:
            self._invalidate_from(resume_from)

    def _invalidate_from(self, stage: ReviewStage) -> None:
        # clear all stage data at or after `stage`, keep earlier stages
        stage_to_keys: Dict[ReviewStage, list] = {
            ReviewStage.CLONE: ['clone_dir'],
            ReviewStage.ARCH: ['arch_doc'],
            ReviewStage.SPEC: ['review_spec'],
            ReviewStage.PR_SUMMARY: ['pr_summary'],
            ReviewStage.R1: [],   # r1 uses per-hunk keys (r1_hunk_*)
            ReviewStage.R2: ['r2_shared_context'],  # r2 uses per-file keys (r2_file_*)
            ReviewStage.R3: ['r3'],
            ReviewStage.R4: ['r4'],
            ReviewStage.FINAL: ['final'],
        }
        for s in ReviewStage.ordered():
            if s.index() >= stage.index():
                for key in stage_to_keys.get(s, []):
                    self._data.pop(key, None)
                # clear per-hunk / per-file keys for r1 and r2
                if s == ReviewStage.R1:
                    for k in list(self._data.keys()):
                        if k.startswith('r1_hunk_'):
                            del self._data[k]
                elif s == ReviewStage.R2:
                    for k in list(self._data.keys()):
                        if k.startswith('r2_file_'):
                            del self._data[k]
                # clear stage-done marker
                self._data.pop(self._STAGE_DONE_PREFIX + s.value, None)
        self._flush()

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def save(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def mark_stage_done(self, stage: ReviewStage) -> None:
        self._data[self._STAGE_DONE_PREFIX + stage.value] = True
        self._flush()

    def is_stage_done(self, stage: ReviewStage) -> bool:
        return bool(self._data.get(self._STAGE_DONE_PREFIX + stage.value))

    def should_use_cache(self, stage: ReviewStage) -> bool:
        if self._resume_from is None:
            return True
        return stage < self._resume_from

    def _flush(self) -> None:
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
    def _review_root() -> str:
        return os.path.join(os.path.expanduser(lazyllm.config['home']), 'review')

    @staticmethod
    def global_cache_dir() -> str:
        d = os.path.join(_ReviewCheckpoint._review_root(), 'cache')
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def repo_cache_dir(repo: str) -> str:
        safe_repo = re.sub(r'[^a-zA-Z0-9_-]', '_', repo)
        d = os.path.join(_ReviewCheckpoint._review_root(), 'cache', safe_repo)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def pr_dir(pr_number: int, repo: str) -> str:
        d = os.path.join(_ReviewCheckpoint.repo_cache_dir(repo), str(pr_number))
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def default_path(pr_number: int, repo: str) -> str:
        return os.path.join(_ReviewCheckpoint.pr_dir(pr_number, repo), 'checkpoint.json')

    # kept for backward compatibility
    @staticmethod
    def review_cache_dir() -> str:
        return _ReviewCheckpoint.global_cache_dir()
