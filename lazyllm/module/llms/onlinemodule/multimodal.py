import lazyllm
from typing import Any
from .base import OnlineMultiModalBase
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
    STT_MODELS = {
        'qwen': QwenSTTModule,
        'glm': GLMSTTModule
    }
    TTS_MODELS = {
        'qwen': QwenTTSModule,
        'siliconflow': SiliconFlowTTSModule,
        'minimax': MinimaxTTSModule
    }
    TEXT2IMAGE_MODELS = {
        'qwen': QwenTextToImageModule,
        'doubao': DoubaoTextToImageModule,
        'glm': GLMTextToImageModule,
        'siliconflow': SiliconFlowTextToImageModule,
        'minimax': MinimaxTextToImageModule,
        'aiping': AipingTextToImageModule
    }

    TYPE_MODEL_MAP = {
        'stt': STT_MODELS,
        'tts': TTS_MODELS,
        'text2image': TEXT2IMAGE_MODELS,
        'image_editing': TEXT2IMAGE_MODELS,
    }

    @staticmethod
    def _validate_parameters(source: str, model: str, type: str, base_url: str, **kwargs) -> tuple:
        assert type in OnlineMultiModalModule.TYPE_MODEL_MAP, f'Invalid type: {type}'
        available_model = OnlineMultiModalModule.TYPE_MODEL_MAP[type]
        if model in available_model and source is None:
            source, model = model, source

        if kwargs.get('skip_auth', False):
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        if source is None:
            if kwargs.get('api_key'):
                raise ValueError('No source is given but an api_key is provided.')
            for src in available_model:
                if lazyllm.config[f'{src}_api_key']:
                    source = src
                    break
            if source is None:
                raise KeyError(f'No api_key is configured for any of the models {available_model}.')

        assert source in available_model, f'Unsupported source: {source}'

        if type == 'image_editing':
            default_module_cls = OnlineMultiModalModule.TYPE_MODEL_MAP[type][source]
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
        available_model = OnlineMultiModalModule.TYPE_MODEL_MAP[type]
        return available_model[source](**params)
