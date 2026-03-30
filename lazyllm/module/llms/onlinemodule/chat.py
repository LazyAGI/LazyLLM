import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Any, ContextManager, Dict, List, Optional, Union
from lazyllm.common.bind import _MetaBind

from ...servermodule import LLMBase, StaticParams
from lazyllm.module import ModuleBase
from .map_model_type import get_model_type
from .base import OnlineChatModuleBase
from .base.utils import select_source_with_default_key
from .dynamic_router import DynamicSourceRouterMixin, dynamic_model_config_context


def dynamic_chat_config(
    modules: Optional[Union[Any, List[Any]]] = None,
    *,
    source: Optional[str] = None,
    model: Optional[str] = None,
    url: Optional[str] = None,
    skip_auth: Optional[bool] = None,
) -> ContextManager[None]:
    return dynamic_model_config_context('chat', modules, source=source, model=model, url=url, skip_auth=skip_auth)


class _ChatModuleMeta(_MetaBind):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineChatModule(ModuleBase, LLMBase, DynamicSourceRouterMixin, metaclass=_ChatModuleMeta):
    _dynamic_module_slot = 'chat'
    _dynamic_source_error = 'No source is configured for dynamic LLM source.'

    @staticmethod
    def _encapsulate_parameters(base_url: str, model: str, stream: bool, return_trace: bool, **kwargs) -> Dict[str, Any]:
        params = {'stream': stream, 'return_trace': return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)
        return params

    def __new__(cls, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None,
                api_key: str = None, static_params: Optional[StaticParams] = None, id: Optional[str] = None,
                name: Optional[str] = None, group_id: Optional[str] = None, dynamic_auth: bool = False, **kwargs):
        if model in lazyllm.online.chat and source is None: source, model = model, source
        if cls._should_use_dynamic(source, dynamic_auth, skip_auth):
            return super().__new__(cls)

        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        source, default_key = select_source_with_default_key(lazyllm.online.chat, source, LLMType.CHAT)
        api_key = api_key if api_key is not None else default_key
        if skip_auth and not base_url:
            raise KeyError('base_url must be set for local serving.')

        if type is None and model:
            type = get_model_type(model)
        if type in ['embed', 'rerank', 'cross_modal_embed']:
            raise AssertionError(f'\'{model}\' should use OnlineEmbeddingModule')
        elif type in ['stt', 'tts', 'sd']:
            raise AssertionError(f'\'{model}\' should use OnlineMultiModalModule')
        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace, api_key=api_key,
                                                          skip_auth=skip_auth, type=type.upper() if type else None,
                                                          **kwargs)
        return getattr(lazyllm.online.chat, source)(**params)

    def __init__(self, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                 return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None,
                 api_key: str = None, static_params: Optional[StaticParams] = None, id: Optional[str] = None,
                 name: Optional[str] = None, group_id: Optional[str] = None, dynamic_auth: bool = False, **kwargs):
        assert model is None, 'model should be given in forward method or global config.'
        assert base_url is None, 'base_url should be given in forward method or global config.'
        if type in ['embed', 'rerank', 'cross_modal_embed']:
            raise AssertionError(f'\'{type}\' should use OnlineEmbeddingModule')
        elif type in ['stt', 'tts', 'sd']:
            raise AssertionError(f'\'{type}\' should use OnlineMultiModalModule')

        normalized_type = type.upper() if type else None
        ModuleBase.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, type=normalized_type, static_params=static_params)
        self._kwargs = kwargs
        self._skip_auth = skip_auth
        self._type = normalized_type  # overwrite type to avoid convert None to 'llm'
        self._init_dynamic_auth(api_key, dynamic_auth)

    def _build_supplier(self, source: str, skip_auth: bool):
        params = {
            'stream': self._stream, 'type': self._type, 'static_params': self._static_params,
            'skip_auth': skip_auth, 'api_key': self._api_key,
            'return_trace': self._return_trace,
            **self._kwargs,
        }
        return getattr(lazyllm.online.chat, source)(**params)
