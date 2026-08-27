# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
import re
from typing import Any, Dict, Optional

import lazyllm
from lazyllm import LOG

_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


lazyllm.globals.config.add('dynamic_fs_auth', dict, None, 'DYNAMIC_FS_AUTH',
                           description='Per-source dynamic FS auth: {source: token}.')
lazyllm.globals.config.add('dynamic_tool_auth', dict, None, 'DYNAMIC_TOOL_AUTH',
                           description='Per-tool dynamic auth: {tool_name: token}. '
                           'Used by search engines and other API-key-based tools.')


# Maps a canonical tool name to the globals.config key it writes into.
# Add new tools here — no code changes needed elsewhere.
#
# 'dynamic_fs_auth'   – FS-layer tools (LazyLLMFSBase subclasses)
# 'dynamic_tool_auth' – API-key-based tools (SearchBase subclasses, etc.)
TOOL_AUTH_REGISTRY: Dict[str, str] = {
    # ── FS tools (LazyLLMFSBase) ──────────────────────────────────────────
    'feishu': 'dynamic_fs_auth',
    'lark': 'dynamic_fs_auth',   # alias for feishu
    'notion': 'dynamic_fs_auth',
    'confluence': 'dynamic_fs_auth',
    'googledrive': 'dynamic_fs_auth',
    'onedrive': 'dynamic_fs_auth',
    'yuque': 'dynamic_fs_auth',
    'ones': 'dynamic_fs_auth',
    's3': 'dynamic_fs_auth',
    # ── Search / API-key tools (SearchBase) ──────────────────────────────
    'bing': 'dynamic_tool_auth',
    'google': 'dynamic_tool_auth',
    'tencent': 'dynamic_tool_auth',
    'bocha': 'dynamic_tool_auth',
    'serpapi': 'dynamic_tool_auth',
    'tavily': 'dynamic_tool_auth',
    'semantic_scholar': 'dynamic_tool_auth',
    'google_books': 'dynamic_tool_auth',
    'stackoverflow': 'dynamic_tool_auth',
    'sciverse': 'dynamic_tool_auth',
}

# Default config key for tools not listed in TOOL_AUTH_REGISTRY.
_DEFAULT_CONFIG_KEY = 'dynamic_tool_auth'


def inject_tool_config(tool_config: Optional[Dict[str, Any]]) -> None:
    '''Inject per-request tool credentials into lazyllm globals.

    tool_config maps tool names to credential tokens or token lists::

        {
            "feishu": "u-xxx",    # OAuth2 access token (caller is responsible for freshness)
            "bing":   "sk-xxx",
            "google": ["AIza...", "AIza..."],
        }

    The destination config key for each tool is determined by
    :data:`TOOL_AUTH_REGISTRY`.  Unknown tools fall back to
    ``dynamic_tool_auth``.

    After this call, globals.config is updated, e.g.::

        globals.config['dynamic_fs_auth']   = {..., 'feishu': 'u-xxx'}
        globals.config['dynamic_tool_auth'] = {..., 'bing': 'sk-xxx', 'google': 'AIza...'}
    '''
    if not tool_config:
        return

    # Collect updates grouped by config key.
    updates: Dict[str, Dict[str, Any]] = {}
    injected: list = []

    for tool_name, token in tool_config.items():
        if isinstance(token, str):
            value = token.strip()
        elif isinstance(token, (list, tuple)) and all(isinstance(k, str) for k in token):
            keys = [k.strip() for k in token if k.strip()]
            value = keys if len(keys) > 1 else (keys[0] if keys else '')
        else:
            LOG.warning(f'[inject_tool_config] skipping {tool_name!r}: expected str or list[str] token, '
                        f'got {type(token).__name__}')
            continue
        if not value:
            LOG.warning(f'[inject_tool_config] skipping {tool_name!r}: token is empty')
            continue

        canonical = tool_name.lower().strip()
        config_key = TOOL_AUTH_REGISTRY.get(canonical, _DEFAULT_CONFIG_KEY)

        updates.setdefault(config_key, {})[canonical] = value
        injected.append(canonical)

    for config_key, new_entries in updates.items():
        existing = lazyllm.globals.config[config_key] or {}
        lazyllm.globals.config[config_key] = {**existing, **new_entries}

    LOG.info(f'[inject_tool_config] injected tools: {sorted(injected)}')


def get_dynamic_env_vars() -> Dict[str, str]:
    raw = lazyllm.globals.get('dynamic_env_vars', {}) or {}
    return {
        str(name): str(value)
        for name, value in raw.items()
        if name is not None and value is not None
    }


def effective_env_value(name: str) -> str:
    key = str(name or '').strip()
    if not key:
        return ''
    dynamic = get_dynamic_env_vars()
    if key in dynamic:
        return str(dynamic[key])
    return str(os.getenv(key) or '')


def _validate_inject_env_name(name: Any) -> Optional[str]:
    key = str(name or '').strip()
    if not key:
        return None
    if '\0' in key:
        LOG.warning('[inject_env_vars] skipping env name containing NUL')
        return None
    if not _ENV_NAME_RE.fullmatch(key):
        LOG.warning(f'[inject_env_vars] skipping invalid env name: {key!r}')
        return None
    return key


def inject_env_vars(env_vars: Optional[Dict[str, Any]]) -> None:
    '''Inject environment variables for skill script execution.

    Values are stored in lazyllm globals for the active session and consumed by
    SkillManager.run_script when it starts the script subprocess. This does not
    mutate the parent process ``os.environ``.

    Semantics:
    - A non-empty value overwrites the same name for this session.
    - An empty string removes a previously injected name (clear).
    - ``None`` values are ignored.
    '''
    if not env_vars:
        return
    existing = dict(get_dynamic_env_vars())
    assigned: list = []
    cleared: list = []
    for name, value in env_vars.items():
        key = _validate_inject_env_name(name)
        if not key or value is None:
            continue
        text = str(value)
        if '\0' in text:
            LOG.warning(f'[inject_env_vars] skipping {key!r}: value contains NUL')
            continue
        if not text.strip():
            if key in existing:
                existing.pop(key, None)
                cleared.append(key)
            continue
        existing[key] = text
        assigned.append(key)
    if not assigned and not cleared:
        return
    lazyllm.globals['dynamic_env_vars'] = existing
    LOG.info(
        f'[inject_env_vars] injected env vars: {sorted(assigned)}; '
        f'cleared env vars: {sorted(cleared)}'
    )
