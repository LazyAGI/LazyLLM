import lazyllm
from typing import Any
from .base import OnlineMultiModalBase
from .base.utils import select_source_with_default_key
from .supplier.qwen import QwenSTTModule, QwenTTSModule, QwenTextToImageModule
from .supplier.doubao import DoubaoTextToImageModule
from .supplier.glm import GLMSTTModule, GLMTextToImageModule
from .supplier.siliconflow import SiliconFlowTextToImageModule, SiliconFlowTTSModule
from .supplier.minimax import MinimaxTextToImageModule, MinimaxTTSModule
from .supplier.aiping import AipingTextToImageModule


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
        'stt': 'stt',
        'tts': 'tts',
        'text2image': 'texttoimage',
        'image_editing': 'texttoimage',
    }

    @staticmethod
    def _get_group(type: str):
        group_name = OnlineMultiModalModule.TYPE_GROUP_MAP.get(type)
        if not group_name:
            raise AssertionError(f'Invalid type: {type}')
        group = getattr(lazyllm.online, group_name)
        return {k: v for k, v in group.items() if k != 'base'}

    @staticmethod
    def _validate_parameters(source: str, model: str, type: str, base_url: str, **kwargs) -> tuple:
        available_model = OnlineMultiModalModule._get_group(type)
        if model in available_model and source is None:
            source, model = model, source
        if source is None and kwargs.get('api_key'):
            raise ValueError('No source is given but an api_key is provided.')
        source, default_key = select_source_with_default_key(available_model, explicit_source=source)
        if default_key and not kwargs.get('api_key'):
            kwargs['api_key'] = default_key

        if kwargs.get('skip_auth', False):
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        if type == 'image_editing':
            default_module_cls = available_model[source]
            default_editing_model = getattr(default_module_cls, 'IMAGE_EDITING_MODEL_NAME', None)
            if model is None and default_editing_model:
                model = default_editing_model
                lazyllm.LOG.info(f'Image editing enabled for {source}. Automatically selected default model: {model}')

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
        source, model, kwargs_normalized = OnlineMultiModalModule._validate_parameters(
            source=source, model=model, type=type, base_url=base_url, **kwargs
        )
        params = {'return_trace': return_trace,
                  'type': type}
        if model is not None:
            params['model'] = model
        params.update(kwargs_normalized)
        available_model = OnlineMultiModalModule._get_group(type)
        return available_model[source](**params)
