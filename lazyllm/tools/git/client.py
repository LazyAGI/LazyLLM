# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
import subprocess
from typing import Dict, Optional, Tuple

import lazyllm
from lazyllm import config

from .base import LazyLLMGitBase

# Host -> backend name for inferring backend from repo URL.
_HOST_TO_BACKEND: Dict[str, str] = {
    'github.com': 'github',
    'gitlab.com': 'gitlab',
    'gitee.com': 'gitee',
    'gitcode.com': 'gitcode',
}

# Env vars that indicate which backend to use when backend is not configured.
_BACKEND_ENV_VARS: Dict[str, tuple] = {
    'github': ('GITHUB_TOKEN', 'GH_TOKEN'),
    'gitlab': ('GITLAB_TOKEN',),
    'gitee': ('GITEE_TOKEN',),
    'gitcode': ('GITCODE_TOKEN',),
}


def _resolve_token_for_backend(backend: str, token: Optional[str] = None) -> str:
    '''Resolve token for the given backend from argument, env, or gh CLI (github only).'''
    if token and token.strip():
        return token.strip()
    backend = backend.lower()
    env_keys = _BACKEND_ENV_VARS.get(backend, ())
    for key in env_keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    if backend == 'github':
        try:
            out = subprocess.run(
                ['gh', 'auth', 'token'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout and out.stdout.strip():
                return out.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
    raise ValueError(
        f'No token for backend {backend!r}. Set token=... or env {env_keys!r}; '
        'for GitHub you can also use `gh auth login`.'
    )


def _detect_backend_from_repo(repo: str) -> Optional[str]:
    '''
    Infer backend from repo URL if it looks like a full URL (https://host/..., git@host:...).
    Returns None for plain owner/repo or unrecognized host.
    '''
    if not repo or not isinstance(repo, str):
        return None
    s = repo.strip().strip('/')
    if s.startswith('git@'):
        m = re.match(r'git@([^:]+):(.+)', s)
        if m:
            host = m.group(1).lower()
            return _HOST_TO_BACKEND.get(host) or (
                'github' if 'github' in host else
                'gitlab' if 'gitlab' in host else
                'gitee' if 'gitee' in host else
                'gitcode' if 'gitcode' in host else None
            )
        return None
    if s.startswith('https://') or s.startswith('http://'):
        parts = s.replace('https://', '').replace('http://', '').split('/', 2)
        if len(parts) >= 1:
            host = parts[0].lower().split(':')[0]
            return _HOST_TO_BACKEND.get(host)
    return None


def _normalize_repo_to_path(repo: str) -> str:
    '''
    Normalize repo to owner/repo form for backend APIs. Strips .git; from full URL
    (https://host/owner/repo or git@host:owner/repo) extracts the path part.
    '''
    if not repo or not isinstance(repo, str):
        return ''
    s = repo.strip().strip('/')
    if s.endswith('.git'):
        s = s[:-4].strip().strip('/')
    if s.startswith('git@'):
        m = re.match(r'git@[^:]+:(.+)', s)
        if m:
            return m.group(1).strip('/')
        return s
    if s.startswith('https://') or s.startswith('http://'):
        parts = s.split('/', 3)
        if len(parts) >= 4:
            return parts[3].strip('/')
        if len(parts) == 3:
            return parts[2].strip('/') if parts[2] else s
        return s
    return s


def _detect_backend_from_env() -> Tuple[Optional[str], Optional[str]]:
    '''If any backend-specific env var is set, return (backend, token).'''
    for backend, keys in _BACKEND_ENV_VARS.items():
        for key in keys:
            value = os.environ.get(key)
            if value and (value := value.strip()):
                return backend, value
    return None, None


def _detect_backend_gh_cli() -> Tuple[Optional[str], Optional[str]]:
    '''If gh is installed and authenticated, return ("github", token).'''
    try:
        out = subprocess.run(
            ['gh', 'auth', 'token'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            return 'github', out.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None, None


config.add('git_backend', str, None, 'GIT_BACKEND',
           description='Default git backend: github, gitlab, gitee, gitcode. None for auto-detect.')

class Git:
    def __new__(cls, backend: Optional[str] = None, token: Optional[str] = None, repo: Optional[str] = None,
                user: Optional[str] = None, api_base: Optional[str] = None, return_trace: bool = False,
                ) -> LazyLLMGitBase:
        # 1. User passed backend -> use it
        # 2. If user passed repo, try to infer backend from URL
        if not backend and repo:
            backend = _detect_backend_from_repo(repo)
        # 3. Read config['git_backend']
        if not backend:
            backend = config['git_backend']
        # 4. Not determined and no token -> env or gh CLI
        if not backend and not token:
            backend, token = _detect_backend_from_env()
            if not backend:
                backend, token = _detect_backend_gh_cli()
        # 5. Default to github
        if not backend:
            backend = 'github'
        # Resolve token (from arg, env, or gh) when backend is known
        token = _resolve_token_for_backend(backend, token)
        # Normalize repo to owner/repo for backend APIs (full URL -> path only)
        repo_path = _normalize_repo_to_path(repo) if repo else ''
        return getattr(lazyllm.git, backend)(
            token=token, repo=repo_path, user=user, api_base=api_base, return_trace=return_trace
        )
