# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import subprocess
import tempfile
from typing import Any, Optional, Tuple

import lazyllm

_GIT_ENV = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': 'echo'}


def _is_complete_clone(clone_dir: str) -> bool:
    try:
        result = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_pull_if_outdated(clone_dir: str, branch: str) -> bool:
    try:
        fetch = subprocess.run(
            ['git', '-C', clone_dir, 'fetch', '--depth', '1', 'origin', branch],
            capture_output=True, text=True, timeout=60, env=_GIT_ENV,
        )
        if fetch.returncode != 0:
            lazyllm.LOG.warning(f'git fetch failed: {fetch.stderr.strip()}')
            return False
        local = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        remote = subprocess.run(
            ['git', '-C', clone_dir, 'rev-parse', 'FETCH_HEAD'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if local == remote:
            lazyllm.LOG.info(f'Clone is up-to-date at {local[:8]}')
            return False
        subprocess.run(
            ['git', '-C', clone_dir, 'reset', '--hard', 'FETCH_HEAD'],
            capture_output=True, text=True, timeout=30, check=True,
        )
        lazyllm.LOG.info(f'Clone updated: {local[:8]} → {remote[:8]}')
        return True
    except Exception as e:
        lazyllm.LOG.warning(f'Failed to pull latest changes: {e}')
        return False


def _pin_clone_to_sha(clone_dir: str, pin_sha: str) -> None:
    cur = subprocess.run(
        ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if cur.startswith(pin_sha[:8]) or pin_sha.startswith(cur[:8]):
        return
    try:
        subprocess.run(
            ['git', '-C', clone_dir, 'fetch', 'origin', pin_sha],
            capture_output=True, text=True, timeout=120, env=_GIT_ENV,
        )
        subprocess.run(
            ['git', '-C', clone_dir, 'checkout', pin_sha],
            capture_output=True, text=True, timeout=30, check=True,
        )
        lazyllm.LOG.info(f'Clone pinned to SHA {pin_sha[:8]}')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        lazyllm.LOG.warning(f'Failed to pin clone to {pin_sha[:8]}: {e}')


def _collect_code_summary(clone_dir: str) -> str:
    from .file_context import _SKIP_DIRS, _SKIP_EXTS
    tree_lines = [
        os.path.join(os.path.relpath(root, clone_dir), fname) if os.path.relpath(root, clone_dir) != '.' else fname
        for root, dirs, files in os.walk(clone_dir)
        for _ in [dirs.__setitem__(slice(None), [d for d in dirs if d not in _SKIP_DIRS])]
        for fname in sorted(files) if not any(fname.endswith(ext) for ext in _SKIP_EXTS)
    ]
    return '\n'.join(tree_lines)


def _fetch_repo_code(repo_url: str, branch: str, work_dir: Optional[str] = None,
                     pin_sha: Optional[str] = None) -> Tuple[str, str]:
    import shutil
    clone_dir = work_dir or tempfile.mkdtemp(prefix='lazyllm_review_')
    if os.path.isdir(clone_dir):
        if _is_complete_clone(clone_dir):
            if pin_sha:
                cur = subprocess.run(
                    ['git', '-C', clone_dir, 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                if cur.startswith(pin_sha[:8]) or pin_sha.startswith(cur[:8]):
                    lazyllm.LOG.info(f'Clone already at pinned SHA {pin_sha[:8]}')
                    return clone_dir, _collect_code_summary(clone_dir)
            else:
                lazyllm.LOG.info(f'Reusing existing clone at {clone_dir}, checking for updates...')
                _try_pull_if_outdated(clone_dir, branch)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
    try:
        if not _is_complete_clone(clone_dir):
            depth_args = ['--depth', '1'] if not pin_sha else []
            subprocess.run(
                ['git', 'clone', '--single-branch', '--branch', branch, *depth_args,
                 '--', repo_url, clone_dir],
                capture_output=True, text=True, timeout=300, check=True, env=_GIT_ENV,
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'git clone failed: {e.stderr or e.stdout}') from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError('git clone timed out') from e

    if pin_sha and _is_complete_clone(clone_dir):
        _pin_clone_to_sha(clone_dir, pin_sha)

    return clone_dir, _collect_code_summary(clone_dir)


def _resolve_clone_target(pr: Any, base_repo: str) -> Tuple[str, str]:
    raw = getattr(pr, 'raw', None) or {}

    head = raw.get('head') or {}
    head_repo = head.get('repo') or {}
    head_clone_url = (head_repo.get('clone_url')
                      or head_repo.get('http_url_to_repo')
                      or head_repo.get('web_url')
                      or '')
    head_branch = head.get('ref') or head.get('branch') or ''

    base = raw.get('base') or {}
    base_repo_info = base.get('repo') or {}
    base_full_name = base_repo_info.get('full_name') or base_repo
    head_full_name = head_repo.get('full_name') or ''

    def _default_url(r: str) -> str:
        if r.startswith('http'):
            return r if r.endswith('.git') else r + '.git'
        return f'https://github.com/{r}.git'

    if head_clone_url and head_full_name and head_full_name != base_full_name:
        if not head_clone_url.endswith('.git'):
            head_clone_url += '.git'
        branch = head_branch or 'main'
        return head_clone_url, branch

    if head_branch:
        return _default_url(base_repo), head_branch

    base_branch = base.get('ref') or getattr(pr, 'target_branch', '') or 'main'
    return _default_url(base_repo), base_branch
