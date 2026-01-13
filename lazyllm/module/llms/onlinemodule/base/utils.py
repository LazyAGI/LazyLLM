from ....module import ModuleBase
from lazyllm import config, LazyLLMRegisterMetaClass
from typing import Optional, Union, List
import random
import re


config.add('cache_online_module', bool, False, 'CACHE_ONLINE_MODULE',
           description='Whether to cache the online module result. Use for unit test.')

def _normalize_key(s: str) -> str:
    return re.sub(r'[_\-\s]', '', s).lower()

def _parse_supplier(cls) -> Optional[str]:
    for base in cls.__mro__:
        name = base.__name__
        if name.startswith('LazyLLM') and name.endswith('Base'):
            return _normalize_key(name[len('LazyLLM'):-len('Base')])
    return None

def _parse_type(cls, supplier_key: str) -> Optional[str]:
    name = cls.__name__
    if name.endswith('Module'):
        name = name[:-len('Module')]
    if name.endswith('MultiModal'):
        name = name[:-len('MultiModal')]
    normalized = _normalize_key(name)
    if not normalized.startswith(supplier_key):
        return None
    type_key = normalized[len(supplier_key):]
    return type_key or None

def build_online_group(cls) -> Optional[Union[str, tuple]]:
    if cls.__name__.endswith('Base'):
        return None
    supplier = _parse_supplier(cls)
    if not supplier:
        return None
    type_key = _parse_type(cls, supplier)
    if not type_key:
        return ''
    return (f'online.{type_key}', supplier)


class LazyLLMOnlineBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    @classmethod
    def __lazyllm_group__(cls):
        return build_online_group(cls)

    def __init__(self, api_key: Optional[Union[str, List[str]]],
                 skip_auth: Optional[bool] = False, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        if not skip_auth and not api_key: raise ValueError('api_key is required')
        self.__api_keys = '' if skip_auth else api_key
        self.__headers = [self._get_header(key) for key in (api_key if isinstance(api_key, list) else [api_key])]
        if config['cache_online_module']:
            self.use_cache()

    @property
    def _api_key(self):
        return random.choice(self.__api_keys) if isinstance(self.__api_keys, list) else self.__api_keys

    @staticmethod
    def _get_header(api_key: str) -> dict:
        return {'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + api_key} if api_key else {})}

    def _get_empty_header(self, api_key: Optional[str] = None) -> dict:
        api_key = api_key or self._api_key
        return {'Authorization': f'Bearer {api_key}'} if api_key else None

    @property
    def _header(self):
        return random.choice(self.__headers)


OnlineModuleBase = LazyLLMOnlineBase  # alias for legacy imports
