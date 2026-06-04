# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, Optional

import lazyllm
from lazyllm import LOG


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
