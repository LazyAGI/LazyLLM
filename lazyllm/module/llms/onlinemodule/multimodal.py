import lazyllm
from typing import Any, Dict
from .base import OnlineMultiModalBase
from .supplier.qwen import QwenSTTModule, QwenTTSModule, QwenTextToImageModule
from .supplier.doubao import DoubaoTextToImageModule
from .supplier.glm import GLMSTTModule, GLMTextToImageModule
from .supplier.siliconflow import SiliconFlowTextToImageModule, SiliconFlowTTSModule, SiliconFlowTextToImageEditModule
from .supplier.minimax import MinimaxTextToImageModule, MinimaxTTSModule


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
        'minimax': MinimaxTextToImageModule
    }
    

    @staticmethod
    def _encapsulate_parameters(base_url: str,
                                model: str,
                                return_trace: bool,
                                image_edit: bool,
                                **kwargs) -> Dict[str, Any]:
        '''Encapsulate parameters for module initialization'''
        params = {'return_trace': return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        if image_edit is not None:
            params['image_edit'] = image_edit
        params.update(kwargs)
        return params

    def __new__(self,
                model: str = None,
                source: str = None,
                base_url: str = None,
                return_trace: bool = False,
                function: str = 'stt',
                image_edit: bool = False, # 新增
                **kwargs):
        '''
        Create a new OnlineMultiModalModule instance.

        Args:
            model: Model name to use
            source: Model provider (e.g., 'qwen', 'openai', 'glm')
            base_url: Base URL for the model API
            return_trace: Whether to return trace information
            function: Function type ('stt', 'tts', 'text2image')
            **kwargs: Additional parameters for the specific module

        Returns:
            Instance of the appropriate module class

        Raises:
            ValueError: If function is not supported
            KeyError: If no API key is configured
        '''
        # Define function to model mapping
        FUNCTION_MODEL_MAP = {
            'stt': OnlineMultiModalModule.STT_MODELS,
            'tts': OnlineMultiModalModule.TTS_MODELS,
            'text2image': OnlineMultiModalModule.TEXT2IMAGE_MODELS
        }

        if function not in FUNCTION_MODEL_MAP:
            raise ValueError(f'Invalid function: {function}')

        available_model = FUNCTION_MODEL_MAP[function]

        if model in available_model and source is None:
            source, model = model, source

        # 新增
        params = OnlineMultiModalModule._encapsulate_parameters(base_url, model, return_trace, image_edit, **kwargs)

        if kwargs.get('skip_auth', False):
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        if source is None:
            if 'api_key' in kwargs and kwargs['api_key']:
                raise ValueError('No source is given but an api_key is provided.')
            for source in available_model:
                if lazyllm.config[f'{source}_api_key']:
                    break
            else:
                raise KeyError(f'No api_key is configured for any of the models {available_model}.')

        assert source in available_model, f'Unsupported source: {source}'
        return available_model[source](**params)
