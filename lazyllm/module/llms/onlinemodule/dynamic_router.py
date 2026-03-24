import threading
from typing import Dict, Any
from lazyllm import globals

LAZY_DYNAMIC_API_KEY_TOKENS = frozenset(('auto', 'dynamic'))


class DynamicSourceRouterMixin(object):
    '''Dynamic source router mixin.'''
    _dynamic_source_config: str = ''
    _dynamic_source_error: str = 'No source is configured for dynamic source.'

    @classmethod
    def _should_use_dynamic(cls, source: str, dynamic_auth: bool, skip_auth: bool = False) -> bool:
        if dynamic_auth:
            assert source == 'dynamic', 'source should be dynamic for dynamic auth.'
            assert not skip_auth, 'skip_auth should be False for dynamic source.'
        return source == 'dynamic' or bool(globals.config[cls._dynamic_source_config] and not source)

    def _init_dynamic_auth(self, api_key: str, dynamic_auth: bool):
        assert api_key is None or api_key in LAZY_DYNAMIC_API_KEY_TOKENS, \
            'api_key should be given in forward or globals.config.'
        self._api_key = 'dynamic' if (dynamic_auth or api_key == 'auto') else api_key
        self._suppliers: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _build_supplier(self, source: str):
        raise NotImplementedError

    def _get_supplier(self):
        source = globals.config[self._dynamic_source_config]
        if source is None:
            raise KeyError(self._dynamic_source_error)
        if source not in self._suppliers:
            with self._lock:
                if source not in self._suppliers:
                    self._suppliers[source] = self._build_supplier(source)
        return self._suppliers[source]

    def forward(self, *args, **kwargs):
        return self._get_supplier().forward(*args, **kwargs)
