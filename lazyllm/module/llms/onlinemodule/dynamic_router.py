import threading
from typing import Any, Dict
from lazyllm import globals

LAZY_DYNAMIC_API_KEY_TOKENS = frozenset(('auto', 'dynamic'))

globals.config.add('dynamic_model_configs', dict, None, 'DYNAMIC_MODEL_CONFIGS',
                   description='Per-module dynamic params: chat, embed, multimodal; each with source, model, url, '
                   'skip_auth. Use lazyllm.globals.config.ConfigsDict '
                   'for {module_id: {chat:..., embed:..., multimodal:...}}.')


class DynamicSourceRouterMixin(object):
    '''Dynamic source router mixin.'''
    _dynamic_bool_key: str = ''
    _dynamic_module_slot: str = ''
    _dynamic_source_error: str = 'No source is configured for dynamic source.'

    @classmethod
    def _get_dynamic_bucket(cls) -> Dict[str, Any]:
        if not globals.config[cls._dynamic_bool_key]:
            return {}
        raw = globals.config['dynamic_model_configs']
        if raw is None or not isinstance(raw, dict):
            return {}
        inner = raw.get(cls._dynamic_module_slot)
        if not isinstance(inner, dict):
            return {}
        return inner

    @classmethod
    def _dynamic_enabled(cls) -> bool:
        return cls._get_dynamic_bucket().get('source') is not None

    @classmethod
    def _should_use_dynamic(cls, source: str, dynamic_auth: bool, skip_auth: bool = False) -> bool:
        if dynamic_auth:
            assert source == 'dynamic', 'source should be dynamic for dynamic auth.'
            assert not skip_auth, 'skip_auth should be False for dynamic source.'
        return source == 'dynamic' or bool(not source and cls._dynamic_enabled())

    def _init_dynamic_auth(self, api_key: str, dynamic_auth: bool):
        assert api_key is None or api_key in LAZY_DYNAMIC_API_KEY_TOKENS, \
            'api_key should be given in forward or globals.config.'
        self._api_key = 'dynamic' if (dynamic_auth or api_key == 'auto') else api_key
        self._suppliers: Dict[tuple, Any] = {}
        self._lock = threading.Lock()

    def _build_supplier(self, source: str, skip_auth: bool):
        raise NotImplementedError

    def _get_supplier(self):
        bucket = self.__class__._get_dynamic_bucket()
        source = bucket.get('source')
        if source is None:
            raise KeyError(self._dynamic_source_error)
        sa = bucket.get('skip_auth')
        skip_auth = bool(getattr(self, '_skip_auth', False)) if sa is None else bool(sa)
        supplier_key = (source, skip_auth)
        if supplier_key not in self._suppliers:
            with self._lock:
                if supplier_key not in self._suppliers:
                    self._suppliers[supplier_key] = self._build_supplier(source, skip_auth)
        return self._suppliers[supplier_key]

    def _merge_dynamic_forward_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        bucket = self.__class__._get_dynamic_bucket()
        if not bucket: return kwargs
        out = dict(kwargs)
        if 'url' not in out and 'base_url' not in out:
            if (u := bucket.get('url')) is not None: out['url'] = u
        if ('model' not in out and 'model_name' not in out
                and 'embed_model_name' not in out):
            if (m := bucket.get('model')) is not None: out['model'] = m
        return out

    def forward(self, *args, **kwargs):
        merged = self._merge_dynamic_forward_kwargs(kwargs)
        return self._get_supplier().forward(*args, **merged)
