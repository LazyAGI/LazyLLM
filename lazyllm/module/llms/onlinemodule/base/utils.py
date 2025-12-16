from ....module import ModuleBase
from lazyllm import config
from typing import Optional, Union, List
import random


config.add('cache_online_module', bool, False, 'CACHE_ONLINE_MODULE',
           description='Whether to cache the online module result. Use for unit test.')


class OnlineModuleBase(ModuleBase):
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

    @property
    def _header(self):
        return random.choice(self.__headers)
