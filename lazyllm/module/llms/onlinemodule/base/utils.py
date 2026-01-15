from ....module import ModuleBase
from lazyllm import config, LazyLLMRegisterMetaClass
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Optional, Union, List
import random


config.add('cache_online_module', bool, False, 'CACHE_ONLINE_MODULE',
           description='Whether to cache the online module result. Use for unit test.')
allowed = {'chat', LLMType.EMBED, LLMType.MULTIMODAL_EMBED, LLMType.RERANK,
           LLMType.STT, LLMType.TTS, LLMType.TEXT2IMAGE, LLMType.IMAGE_EDITING}


def select_source_with_default_key(available_models, explicit_source: Optional[str] = None):
    if explicit_source:
        assert explicit_source in available_models, f'Unsupported source: {explicit_source}'
        return explicit_source, None
    default_source = config['default_source'] if 'default_source' in config.get_all_configs() else None
    default_key = config['default_key'] if 'default_key' in config.get_all_configs() else None
    if default_source and default_key and default_source in available_models:
        return default_source, default_key
    for candidate in available_models.keys():
        if config[f'{candidate}_api_key']:
            return candidate, None
    raise KeyError(f'No api_key is configured for any of the models {available_models.keys()}.')


class LazyLLMOnlineBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):

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

    @staticmethod
    def _check_and_add_config(key, description):
        if key.lower() not in config.get_all_configs():
            config.add(key, str, '', f'{key.upper()}', description=description)

    @staticmethod
    def __lazyllm_after_registry_hook__(group_name: str, name: str, isleaf: bool):

        config_key_dict = [
            ('_api_key', 'The API key for '),
            ('_model_name', 'The default model name for '),
            ('_text2image_model_name', 'The default text2image model name for '),
            ('_tts_model_name', 'The default tts model name for '),
            ('_stt_model_name', 'The default stt model name for '),
        ]

        if group_name == '':
            assert name == 'online'
        elif not isleaf:
            assert group_name == 'online', 'The group can only be "online" here.'
            assert name.lower() in allowed, 'group name error'
        else:
            subgroup = group_name.split('.')[-1]
            assert name.lower().endswith(subgroup), 'Wrong subclass name schema.'
            supplier = name[:-len(subgroup)].lower()

            for key, description in config_key_dict:
                LazyLLMOnlineBase._check_and_add_config(key=supplier + key,
                                                        description=description + supplier)

            if supplier == 'sensenova':
                LazyLLMOnlineBase._check_and_add_config(key='sensenova_secret_key',
                                                        description='The secret key for SenseNova.')
