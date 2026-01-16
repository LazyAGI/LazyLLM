import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Any, Dict, Optional
from .map_model_type import get_model_type
from .base import OnlineChatModuleBase
from .base.utils import select_source_with_default_key


class _ChatModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineChatModule(metaclass=_ChatModuleMeta):

    @staticmethod
    def _encapsulate_parameters(base_url: str, model: str, stream: bool, return_trace: bool, **kwargs) -> Dict[str, Any]:
        params = {'stream': stream, 'return_trace': return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)
        return params

    def __new__(self, model: str = None, source: str = None, base_url: str = None, api_key: str = None,
                stream: bool = True, return_trace: bool = False, skip_auth: bool = False,
                type: Optional[str] = None, **kwargs):
        if model in lazyllm.online.chat and source is None: source, model = model, source
        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        source, default_key = select_source_with_default_key(lazyllm.online.chat,
                                                             explicit_source=source,
                                                             type=LLMType.CHAT)
        if default_key and not api_key:
            api_key = default_key

        if type is None and model:
            type = get_model_type(model)
        if type in ['embed', 'rerank', 'cross_modal_embed']:
            raise AssertionError(f'\'{model}\' should use OnlineEmbeddingModule')
        elif type in ['stt', 'tts', 'sd']:
            raise AssertionError(f'\'{model}\' should use OnlineMultiModalModule')
        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace, api_key=api_key,
                                                          skip_auth=skip_auth, type=type.upper() if type else None,
                                                          **kwargs)
        if skip_auth:
            source = source or 'openai'
            if not base_url:
                raise KeyError('base_url must be set for local serving.')

        return getattr(lazyllm.online.chat, source)(**params)
