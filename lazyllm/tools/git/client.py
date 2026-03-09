# Copyright (c) 2026 LazyAGI. All rights reserved.
'''
Unified Git client: selects backend by config, source, or auto-detect (env, gh CLI, default github).
Backend selection: if user set config['git_backend'] (see lazyllm.configs), use it; else if no backend
configured, check GITHUB_TOKEN etc. env vars and choose by that; else check gh CLI auth; else default github.
'''
import os
import subprocess
from typing import Any, Dict, Optional, Type

import lazyllm
from lazyllm.common.registry import LazyLLMRegisterMetaClass

from .base import LazyLLMGitBase

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


def _detect_backend_from_env() -> Optional[str]:
    '''If any backend-specific env var is set, return that backend name.'''
    for backend, keys in _BACKEND_ENV_VARS.items():
        for key in keys:
            if os.environ.get(key) and os.environ.get(key).strip():
                return backend
    return None


def _detect_backend_gh_cli() -> Optional[str]:
    '''If gh is installed and authenticated, return "github".'''
    try:
        out = subprocess.run(
            ['gh', 'auth', 'token'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            return 'github'
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _resolve_backend(backend: Optional[str], source: Optional[str]) -> str:
    '''
    Final backend: source overrides; else backend param; else config git_backend (lazyllm.configs);
    else env (GITHUB_TOKEN etc.); else gh CLI; else github.
    '''
    if source is not None and str(source).strip():
        return str(source).strip().lower()
    if backend is not None and str(backend).strip():
        return str(backend).strip().lower()
    try:
        cfg = getattr(lazyllm, 'config', None)
        if cfg is not None:
            v = cfg['git_backend']
            if v is not None and str(v).strip():
                return str(v).strip().lower()
    except (KeyError, AttributeError, TypeError):
        pass
    detected = _detect_backend_from_env()
    if detected is not None:
        return detected
    detected = _detect_backend_gh_cli()
    if detected is not None:
        return detected
    return 'github'


class Git:
    '''
    Unified Git client. Use backend/source to choose platform; when not set, backend is
    auto-detected from config, env (GITHUB_TOKEN, GITLAB_TOKEN, etc.), gh CLI, then default github.
    '''

    def __new__(
        cls,
        backend: Optional[str] = None,
        source: Optional[str] = None,
        token: Optional[str] = None,
        repo: Optional[str] = None,
        api_base: Optional[str] = None,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> LazyLLMGitBase:
        '''
        Return a Git backend instance (GitHub, GitLab, Gitee, or GitCode).

        If source is provided, it is used as the backend and no auto-detect runs.
        Otherwise backend is taken from argument, then config["git_backend"], then
        env vars (GITHUB_TOKEN, GITLAB_TOKEN, GITEE_TOKEN, GITCODE_TOKEN), then
        gh CLI if authenticated, then "github".

        Args:
            backend: Backend name (github, gitlab, gitee, gitcode). Overridden by source.
            source: If set, use as backend and skip config/env/gh detection.
            token: Access token; resolved from env or gh when None.
            repo: Repository identifier (e.g. owner/repo).
            api_base: Optional API base URL for the backend.
            return_trace: Whether to return call trace.
            **kwargs: Passed to the backend constructor.

        Returns:
            An instance of the chosen backend (GitHub, GitLab, Gitee, or GitCode).
        '''
        resolved = _resolve_backend(backend, source)
        token = _resolve_token_for_backend(resolved, token)
        try:
            registry = LazyLLMRegisterMetaClass.all_clses['git']
            BackendClass: Type[LazyLLMGitBase] = registry[resolved]
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f'Unknown git backend {resolved!r}. Valid: github, gitlab, gitee, gitcode.'
            ) from e
        return BackendClass(
            token=token,
            repo=repo or '',
            api_base=api_base,
            return_trace=return_trace,
            **kwargs,
        )
