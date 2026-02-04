import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Any
from .base import OnlineMultiModalBase
from .base.utils import select_source_with_default_key


class _OnlineMultiModalMeta(type):
    '''Metaclass for OnlineMultiModalModule to support isinstance checks'''
    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineMultiModalBase):
            return True
        return super().__instancecheck__(__instance)

class OnlineMultiModalModule(metaclass=_OnlineMultiModalMeta):
    '''
    Factory class for creating online multimodal models.

    Supports various multimodal functions including:
    - Speech-to-Text (STT)
    - Text-to-Speech (TTS)
    - Text-to-Image generation

    Example:
        # Create an online STT
        stt = OnlineMultiModalModule(source='qwen', function='stt')

        # Create an online TTS
        tts = OnlineMultiModalModule(source='qwen', function='tts')

        # Create an online text-to-image
        img_gen = OnlineMultiModalModule(source='qwen', function='text2image')
    '''
    TYPE_GROUP_MAP = {
        'stt': LLMType.STT,
        'tts': LLMType.TTS,
        'text2image': LLMType.TEXT2IMAGE,
        'image_editing': LLMType.TEXT2IMAGE,
    }

    @staticmethod
    def _validate_parameters(source: str, model: str, type: str, base_url: str, **kwargs) -> tuple:
        assert type in OnlineMultiModalModule.TYPE_GROUP_MAP, f'Invalid type: {type}'
        if model in lazyllm.online[type] and source is None:
            source, model = model, source
        if source is None and kwargs.get('api_key'):
            raise ValueError('No source is given but an api_key is provided.')
        register_type = OnlineMultiModalModule.TYPE_GROUP_MAP.get(type).lower()
        source, default_key = select_source_with_default_key(lazyllm.online[register_type],
                                                             explicit_source=source,
                                                             type=type)
        if default_key and not kwargs.get('api_key'):
            kwargs['api_key'] = default_key

        if kwargs.get('skip_auth', False):
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        default_module_cls = getattr(lazyllm.online[register_type], source)
        if type == 'image_editing':
            default_model_name = getattr(default_module_cls, 'IMAGE_EDITING_MODEL_NAME', None)
        else:
            default_model_name = getattr(default_module_cls, 'MODEL_NAME', None)
        if model is None and default_model_name:
            model = default_model_name
            lazyllm.LOG.info(f'For type {type}, source {source}. Automatically selected default model: {model}')

        if base_url is not None:
            kwargs['base_url'] = base_url
        return source, model, kwargs

    def __new__(self,
                model: str = None,
                source: str = None,
                type: str = None,
                base_url: str = None,
                return_trace: bool = False,
                **kwargs):
        if type is None:
            type = kwargs.pop('function', None)
        type = LLMType._normalize(type)
        source, model, kwargs_normalized = OnlineMultiModalModule._validate_parameters(
            source=source, model=model, type=type, base_url=base_url, **kwargs
        )
        params = {'return_trace': return_trace,
                  'type': type}
        if model is not None:
            params['model'] = model
        params.update(kwargs_normalized)
        register_type = OnlineMultiModalModule.TYPE_GROUP_MAP.get(type).lower()
        return getattr(lazyllm.online[register_type], source)(**params)
