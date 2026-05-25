from typing import Any, Dict, Optional

# Maps the 'type' field in a role config dict to the _dynamic_module_slot used
# by _DynamicSourceRouterMixin subclasses.
_TYPE_TO_SLOT: Dict[str, str] = {
    'llm': 'chat',
    'chat': 'chat',
    'vlm': 'chat',
    'embed': 'embed',
    'rerank': 'embed',
    'cross_modal_embed': 'embed',
}

# Role name prefixes that imply the 'embed' slot even when 'type' is absent.
_EMBED_ROLE_PREFIXES = ('embed_', 'reranker')


def _infer_slot(role: str, role_cfg: Dict[str, Any]) -> str:
    '''Infer the _dynamic_module_slot for a role from its config or name.'''
    type_val = (role_cfg.get('type') or '').lower().strip()
    if type_val:
        return _TYPE_TO_SLOT.get(type_val, 'chat')
    for prefix in _EMBED_ROLE_PREFIXES:
        if role.startswith(prefix) or role == 'reranker':
            return 'embed'
    return 'chat'


def inject_model_config(model_config: Optional[Dict[str, Any]]) -> None:
    '''Inject per-request model configuration into lazyllm globals.

    model_config keys are role names (e.g. 'llm', 'embed_main', 'reranker').
    Each value is a config dict::

        {
            "llm":        {"source": "openai",      "model": "gpt-4o",      "api_key": "sk-..."},
            "embed_main": {"source": "siliconflow", "model": "BAAI/bge-m3", "api_key": "..."},
            "reranker":   {"source": "siliconflow", "model": "BAAI/bge-reranker-v2-m3", "api_key": "..."},
        }

    The slot ('chat' or 'embed') is inferred from the 'type' field in each role
    config, or from the role name prefix when 'type' is absent.

    After this call, globals.config has the following structure::

        globals.config['dynamic_model_configs'] = ConfigsDict({
            'llm':          {'chat':  {'source': 'openai',      'model': 'gpt-4o', ...}},
            'embed_main':   {'embed': {'source': 'siliconflow', 'model': 'bge-m3', ...}},
            'reranker':     {'embed': {'source': 'siliconflow', 'model': 'bge-reranker', ...}},
        })
        globals.config['openai_api_key'] = ConfigsDict({'llm': 'sk-...'})
        globals.config['siliconflow_api_key'] = ConfigsDict({
            'embed_main': 'sk-...',
            'reranker':   'sk-...',
        })
    '''
    if not model_config:
        return

    import lazyllm
    from lazyllm import LOG
    from lazyllm.module.llms.onlinemodule.dynamic_router import ConfigsDict

    cfg = ConfigsDict()
    api_key_configs: Dict[str, Any] = {}
    injected_roles = []

    for role, role_cfg in model_config.items():
        if not isinstance(role_cfg, dict):
            LOG.warning(f'[inject_model_config] skipping role {role!r}: expected dict, got {type(role_cfg).__name__}')
            continue
        source = (role_cfg.get('source') or '').strip()
        model = (role_cfg.get('model') or '').strip()
        base_url = (role_cfg.get('base_url') or '').strip()
        if not source and not model and not base_url:
            LOG.warning(f'[inject_model_config] skipping role {role!r}: no usable fields')
            continue

        slot = _infer_slot(role, role_cfg)
        bucket: Dict[str, Any] = {}
        if source:
            bucket['source'] = source
        if model:
            bucket['model'] = model
        if base_url:
            bucket['url'] = base_url
        skip_auth = role_cfg.get('skip_auth')
        if skip_auth is not None:
            if isinstance(skip_auth, str):
                skip_auth = skip_auth.strip().lower() not in ('false', '0', 'no', '')
            bucket['skip_auth'] = bool(skip_auth)

        cfg.setdefault(role, {})[slot] = bucket
        injected_roles.append(role)

        api_key = (role_cfg.get('api_key') or '').strip()
        if api_key and source:
            config_key = f'{source}_api_key'
            api_key_configs.setdefault(config_key, ConfigsDict())[role] = api_key

    for config_key, api_key_cfg in api_key_configs.items():
        lazyllm.globals.config[config_key] = api_key_cfg
    lazyllm.globals.config['dynamic_model_configs'] = cfg
    LOG.info(f'[inject_model_config] injected roles: {sorted(injected_roles)}')
