import lazyllm
from typing import Any, Dict
from .base import OnlineMultiModalBase
from .supplier.qwen import QwenSTTModule, QwenTTSModule, QwenTextToImageModule
from .supplier.doubao import DoubaoTextToImageModule
from .supplier.glm import GLMSTTModule, GLMTextToImageModule
from .supplier.siliconflow import SiliconFlowTextToImageModule, SiliconFlowTTSModule
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

    FUNCTION_MODEL_MAP = {
        'stt': STT_MODELS,
        'tts': TTS_MODELS,
        'text2image': TEXT2IMAGE_MODELS
    }
    
    @staticmethod
    def _validate_parameters(source: str, model: str, function: str, base_url: str, **kwargs) -> tuple:
        assert function in OnlineMultiModalModule.FUNCTION_MODEL_MAP, f'Invalid function: {function}'
        available_model = OnlineMultiModalModule.FUNCTION_MODEL_MAP[function]
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
                if lazyllm.config.get(f'{src}_api_key'):
                    source = src
                    break
            if source is None:
                raise KeyError(f'No api_key is configured for any of the models {available_model}.')

        assert source in available_model, f'Unsupported source: {source}'

        if function == 'text2image':
            image_editing = kwargs.get('image_editing', False)
            model_class = available_model[source]
            edit_list = getattr(model_class, 'IMAGE_EDITING_MODEL', None)
            if edit_list:
                if not image_editing and model in edit_list:
                    image_editing = True
                    lazyllm.LOG.info(f'Model {model} supports image editing. Automatically enabled image_editing.')
                if image_editing and model is None:
                    model = edit_list[0]
                    lazyllm.LOG.info(f'Image editing enabled for {source}. Automatically selected model: {model}')
                elif image_editing and model not in edit_list:
                    lazyllm.LOG.warning(
                        f'Model {model} from {source} does not support image editing. '
                        f'Please use models from: {edit_list}'
                    )
            else: 
                if image_editing:
                    lazyllm.LOG.warning(
                        f'Image editing requested for {source}, but no editing models available for this provider.'
                    )
            kwargs['image_editing'] = image_editing

        
        if base_url is not None:
            kwargs['base_url'] = base_url

        return source, model, kwargs
        
    
    def __new__(self,
                model: str = None,
                source: str = None,
                base_url: str = None,
                return_trace: bool = False,
                function: str = 'stt',
                **kwargs):

        source, model, kwargs_normalized = OnlineMultiModalModule._validate_parameters(
            source=source, model=model, function=function, base_url=base_url,**kwargs
        )
        params = {'return_trace': return_trace}
        if model is not None:
            params['model'] = model
        params.update(kwargs_normalized)
        available_model = OnlineMultiModalModule.FUNCTION_MODEL_MAP[function]
        return available_model[source](**params)
