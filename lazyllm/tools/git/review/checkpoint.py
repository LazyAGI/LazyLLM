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
    RSCENE = 'rscene'
    RCHAIN = 'rchain'
    R1 = 'r1'
    R2A = 'r2a'
    R2 = 'r2'
    RMOD = 'rmod'
    RCOV = 'rcov'
    R3 = 'r3'
    FINAL = 'final'
    UPLOAD = 'upload'

    @staticmethod
    def ordered() -> list:
        return _REVIEW_STAGE_ORDER

    def index(self) -> int:
        return _REVIEW_STAGE_ORDER.index(self)

    def __le__(self, other: 'ReviewStage') -> bool:
        return self.index() <= other.index()

    def __lt__(self, other: 'ReviewStage') -> bool:
        return self.index() < other.index()


_REVIEW_STAGE_ORDER = [
    ReviewStage.CLONE, ReviewStage.ARCH, ReviewStage.SPEC,
    ReviewStage.PR_SUMMARY, ReviewStage.RSCENE, ReviewStage.RCHAIN,
    ReviewStage.R1, ReviewStage.R2A, ReviewStage.R2, ReviewStage.RMOD,
    ReviewStage.R3, ReviewStage.RCOV, ReviewStage.FINAL, ReviewStage.UPLOAD,
]


def _load_cache(cache_path: Optional[str], key: str) -> Optional[str]:
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(key) if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_path: Optional[str], updates: Dict[str, Any]) -> None:
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
        data.update(updates)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _save_cache(cache_path: Optional[str], key: str, value: str) -> None:
    _write_cache(cache_path, {key: value})


def _save_cache_multi(cache_path: Optional[str], entries: Dict[str, Any]) -> None:
    _write_cache(cache_path, entries)


class _ReviewCheckpoint:
    _STAGE_DONE_PREFIX = '_stage_done_'
    _INVALIDATED_FROM_KEY = '_invalidated_from'
    _KEY_TO_STAGE: Dict[str, 'ReviewStage'] = {
        'clone_dir': ReviewStage.CLONE,
        'head_sha': ReviewStage.CLONE,
        'diff_text': ReviewStage.CLONE,
        'arch_doc': ReviewStage.ARCH,
        'review_spec': ReviewStage.SPEC,
        'pr_summary': ReviewStage.PR_SUMMARY,
        'r1': ReviewStage.R1,
        'pr_design_doc': ReviewStage.R2A,
        'r2': ReviewStage.R2,
        'rmod': ReviewStage.RMOD,
        'rcov_issues': ReviewStage.RCOV,
        'r3': ReviewStage.R3,
        'r3_shared_context': ReviewStage.R3,
        'final': ReviewStage.FINAL,
        'final_comments': ReviewStage.UPLOAD,
        'upload_done_batches': ReviewStage.UPLOAD,
    }
    _PIPELINE_VERSION_KEY = '_pipeline_version'
    # Bump _PIPELINE_VERSION to invalidate ALL cached stages across ALL PRs
    _PIPELINE_VERSION = 3
    _REVIEW_ROUND_VERSION_KEY = '_review_round_version'
    # Bump _REVIEW_ROUND_VERSION to invalidate only the FINAL (R4) stage cache
    _REVIEW_ROUND_VERSION = 4

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
        old_ver = self._data.get(self._PIPELINE_VERSION_KEY, 0)
        if old_ver < self._PIPELINE_VERSION:
            lazyllm.LOG.warning(
                f'Checkpoint pipeline version {old_ver} < {self._PIPELINE_VERSION}, invalidating all stages'
            )
            self._data = {}
        self._data[self._PIPELINE_VERSION_KEY] = self._PIPELINE_VERSION
        self._flush()
        if resume_from is not None:
            self._mark_invalidated_from(resume_from)

    def _mark_invalidated_from(self, stage: ReviewStage) -> None:
        # record the invalidation boundary without deleting any data
        # data at or after this stage will be ignored by get() / should_use_cache()
        self._data[self._INVALIDATED_FROM_KEY] = stage.value
        self._flush()

    def _invalidated_stage(self) -> Optional[ReviewStage]:
        val = self._data.get(self._INVALIDATED_FROM_KEY)
        if val is None:
            return None
        try:
            return ReviewStage(val)
        except ValueError:
            return None

    def _stage_for_key(self, key: str) -> Optional[ReviewStage]:
        if key in _ReviewCheckpoint._KEY_TO_STAGE:
            return _ReviewCheckpoint._KEY_TO_STAGE[key]
        if key.startswith('r1_hunk_'):
            return ReviewStage.R1
        if key.startswith('r1_window_'):
            return ReviewStage.R1
        if key.startswith('r3_file_') or key.startswith('r3_disc_') or key.startswith('r3_group_'):
            return ReviewStage.R3
        if key.startswith(self._STAGE_DONE_PREFIX):
            stage_val = key[len(self._STAGE_DONE_PREFIX):]
            try:
                return ReviewStage(stage_val)
            except ValueError:
                return None
        return None

    def get(self, key: str) -> Any:
        val = self._data.get(key)
        if val is None:
            return None
        # clone_dir staleness check: treat as missing if directory no longer exists
        if key == 'clone_dir' and val and not os.path.isdir(val):
            lazyllm.LOG.warning(f'Cached clone_dir {val} no longer exists, treating as missing')
            return None
        # if this key belongs to an invalidated stage, treat as missing
        inv = self._invalidated_stage()
        if inv is not None:
            key_stage = self._stage_for_key(key)
            if key_stage is not None and key_stage.index() >= inv.index():
                return None
        return val

    def save(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def mark_stage_done(self, stage: ReviewStage) -> None:
        self._data[self._STAGE_DONE_PREFIX + stage.value] = True
        # clear the invalidation marker only after the full pipeline completes (FINAL),
        # so that all downstream stages remain invalidated until the run finishes
        inv = self._invalidated_stage()
        if inv is not None and stage == ReviewStage.UPLOAD:
            self._data.pop(self._INVALIDATED_FROM_KEY, None)
            self._resume_from = None
        self._flush()

    def is_stage_done(self, stage: ReviewStage) -> bool:
        return bool(self.get(self._STAGE_DONE_PREFIX + stage.value))

    def should_use_cache(self, stage: ReviewStage) -> bool:
        # UPLOAD is a side-effectful stage (GitHub API call): never skip it based on
        # the absence of an invalidation marker alone — require an explicit done flag.
        if stage == ReviewStage.UPLOAD:
            return self.is_stage_done(ReviewStage.UPLOAD)
        if self._resume_from is None:
            inv = self._invalidated_stage()
            if inv is None:
                return True
            return stage < inv
        return stage < self._resume_from

    def _flush(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            lazyllm.LOG.warning(f'Failed to write checkpoint: {e}')

    # stages whose data survives a head_sha rotation (repo-level, not PR-diff-level)
    _STABLE_STAGES = frozenset({ReviewStage.CLONE, ReviewStage.ARCH, ReviewStage.SPEC})
    # keys that belong to a stable stage but must still be purged on rotation
    _PURGE_ON_ROTATE = frozenset({'diff_text', 'head_sha'})

    def rotate_on_head_sha_change(self, new_sha: str) -> bool:
        # If head_sha changed, backup checkpoint and purge all review-round data.
        # Returns True if rotation happened, False otherwise.
        # Keeps only repo-level keys (clone, arch, spec) and pipeline metadata.
        old_sha = self._data.get('head_sha')
        if not old_sha or old_sha == new_sha:
            return False
        # backup old checkpoint
        if os.path.isfile(self._path):
            backup = self._path.replace('.json', f'.{old_sha[:8]}.json')
            try:
                import shutil
                shutil.copy2(self._path, backup)
                lazyllm.LOG.info(f'Checkpoint backed up to {backup}')
            except OSError as e:
                lazyllm.LOG.warning(f'Failed to backup checkpoint: {e}')
        # determine which keys to keep
        keep: Dict[str, Any] = {}
        for key, val in self._data.items():
            # always keep pipeline metadata
            if key in (self._PIPELINE_VERSION_KEY, self._INVALIDATED_FROM_KEY):
                keep[key] = val
                continue
            # keep stage-done flags for stable stages only (except CLONE — diff needs re-fetch)
            if key.startswith(self._STAGE_DONE_PREFIX):
                stage_val = key[len(self._STAGE_DONE_PREFIX):]
                try:
                    s = ReviewStage(stage_val)
                    if s in self._STABLE_STAGES and s != ReviewStage.CLONE:
                        keep[key] = val
                except ValueError:
                    pass
                continue
            # keep data keys belonging to stable stages (except purge-on-rotate keys)
            key_stage = self._stage_for_key(key)
            if key_stage is not None and key_stage in self._STABLE_STAGES and key not in self._PURGE_ON_ROTATE:
                keep[key] = val
        keep['head_sha'] = new_sha
        # remove invalidation marker — we've physically purged, no need for soft invalidation
        keep.pop(self._INVALIDATED_FROM_KEY, None)
        self._data = keep
        self._flush()
        lazyllm.LOG.warning(
            f'head_sha changed ({old_sha[:8]} → {new_sha[:8]}): '
            f'checkpoint rotated, kept {len(keep)} stable keys, purged review-round data'
        )
        return True

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
    def pr_dir(pr_number: Any, repo: str) -> str:
        d = os.path.join(_ReviewCheckpoint.repo_cache_dir(repo), str(pr_number))
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def default_path(pr_number: Any, repo: str) -> str:
        return os.path.join(_ReviewCheckpoint.pr_dir(pr_number, repo), 'checkpoint.json')

    # deprecated: use global_cache_dir() directly; will be removed in a future version
    @staticmethod
    def review_cache_dir() -> str:
        return _ReviewCheckpoint.global_cache_dir()
