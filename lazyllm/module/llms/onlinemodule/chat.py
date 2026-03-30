import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Any, ContextManager, List, Optional, Union
from lazyllm.common.bind import _MetaBind

from ...servermodule import LLMBase, StaticParams
from .base import OnlineChatModuleBase
from .base.utils import select_source_with_default_key
from .dynamic_router import _DynamicSourceRouterMixin, dynamic_model_config_context


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


class OnlineChatModule(_DynamicSourceRouterMixin, LLMBase, metaclass=_ChatModuleMeta):
    _dynamic_module_slot = 'chat'
    _dynamic_source_error = 'No source is configured for dynamic LLM source.'

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

        type = cls._resolve_type_name(type, model, options=[LLMType.LLM, LLMType.CHAT, LLMType.VLM])
        return getattr(lazyllm.online.chat, source)(
            base_url=base_url, model=model, stream=stream, return_trace=return_trace,
            api_key=api_key, skip_auth=skip_auth, type=type, **kwargs)

    def __init__(self, model: str = None, source: str = None, base_url: str = None, stream: bool = True,
                 return_trace: bool = False, skip_auth: bool = False, type: Optional[str] = None,
                 api_key: str = None, static_params: Optional[StaticParams] = None, id: Optional[str] = None,
                 name: Optional[str] = None, group_id: Optional[str] = None, dynamic_auth: bool = False, **kwargs):
        normalized_type = self._resolve_type_name(type, model, options=[LLMType.LLM, LLMType.CHAT, LLMType.VLM])
        _DynamicSourceRouterMixin.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        LLMBase.__init__(self, stream=stream, type=normalized_type, static_params=static_params)
        self._kwargs = kwargs
        self._base_url = base_url
        self._model_name = model
        self._skip_auth = skip_auth
        self._init_dynamic_auth(api_key, dynamic_auth)

    def _build_supplier(self, source: str, skip_auth: bool):
        params = {
            'base_url': self._base_url, 'model': self._model_name, 'stream': self._stream, 'type': self._type,
            'static_params': self._static_params, 'skip_auth': skip_auth, 'api_key': self._api_key,
            'return_trace': self._return_trace, **self._kwargs}
        return getattr(lazyllm.online.chat, source)(**params)
