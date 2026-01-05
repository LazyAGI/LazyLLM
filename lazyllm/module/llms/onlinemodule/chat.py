import lazyllm
from typing import Any, Dict, Optional
from .map_model_type import get_model_type
from .base import OnlineChatModuleBase
from .supplier.openai import OpenAIModule
from .supplier.glm import GLMModule
from .supplier.kimi import KimiModule
from .supplier.sensenova import SenseNovaModule
from .supplier.qwen import QwenModule
from .supplier.doubao import DoubaoModule
from .supplier.deepseek import DeepSeekModule
from .supplier.siliconflow import SiliconFlowModule
from .supplier.minimax import MinimaxModule
from .supplier.ppio import PPIOModule

class _ChatModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineChatModule(metaclass=_ChatModuleMeta):
    MODELS = {'openai': OpenAIModule,
              'sensenova': SenseNovaModule,
              'glm': GLMModule,
              'kimi': KimiModule,
              'qwen': QwenModule,
              'doubao': DoubaoModule,
              'deepseek': DeepSeekModule,
              'siliconflow': SiliconFlowModule,
              'minimax': MinimaxModule,
              'ppio': PPIOModule}

    @staticmethod
    def _encapsulate_parameters(base_url: str, model: str, stream: bool, return_trace: bool, **kwargs) -> Dict[str, Any]:
        params = {'stream': stream, 'return_trace': return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)
        return params

    def __new__(self, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None, **kwargs):
        if model in OnlineChatModule.MODELS.keys() and source is None: source, model = model, source
        if type is None and model:
            type = get_model_type(model)
        if type in ['embed', 'rerank', 'cross_modal_embed']:
            raise AssertionError(f'\'{model}\' should use OnlineEmbeddingModule')
        elif type in ['stt', 'tts', 'sd']:
            raise AssertionError(f'\'{model}\' should use OnlineMultiModalModule')
        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace,
                                                          skip_auth=skip_auth, type=type.upper() if type else None,
                                                          **kwargs)

        if skip_auth:
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        if source is None:
            if 'api_key' in kwargs and kwargs['api_key']:
                raise ValueError('No source is given but an api_key is provided.')
            for source in OnlineChatModule.MODELS.keys():
                if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f'No api_key is configured for any of the models {OnlineChatModule.MODELS.keys()}.')

        assert source in OnlineChatModule.MODELS.keys(), f'Unsupported source: {source}'
        return OnlineChatModule.MODELS[source](**params)
