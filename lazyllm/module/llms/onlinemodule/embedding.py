from typing import Any, ContextManager, List, Optional, Union

import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from lazyllm.common.bind import _MetaBind
from .base import OnlineEmbeddingModuleBase
from .base.utils import select_source_with_default_key
from .supplier.doubao import DoubaoEmbed, DoubaoMultimodalEmbed
from .map_model_type import get_model_type
from .dynamic_router import _DynamicSourceRouterMixin, dynamic_model_config_context


def dynamic_embed_config(
    modules: Optional[Union[Any, List[Any]]] = None,
    *,
    source: Optional[str] = None,
    model: Optional[str] = None,
    url: Optional[str] = None,
    skip_auth: Optional[bool] = None,
) -> ContextManager[None]:
    return dynamic_model_config_context('embed', modules, source=source, model=model, url=url, skip_auth=skip_auth)


class __EmbedModuleMeta(_MetaBind):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(_DynamicSourceRouterMixin, metaclass=__EmbedModuleMeta):
    _dynamic_module_slot = 'embed'
    _dynamic_source_error = 'No source is configured for dynamic embedding source.'

    @staticmethod
    def _resolve_type_name(type_name: Optional[str], embed_model_name: Optional[str]) -> str:
        if type_name is not None:
            return type_name
        resolved = get_model_type(embed_model_name) if embed_model_name else 'embed'
        return resolved if resolved in ('embed', 'rerank') else 'embed'

    @staticmethod
    def _create_supplier(source: str, type_name: str, embed_model_name: str, params: dict):
        if type_name == 'embed':
            if source == 'doubao' and embed_model_name and embed_model_name.startswith('doubao-embedding-vision'):
                return DoubaoMultimodalEmbed(**params)
            if source == 'doubao':
                return DoubaoEmbed(**params)
            return getattr(lazyllm.online.embed, source)(**params)
        if type_name == 'rerank':
            return getattr(lazyllm.online.rerank, source)(**params)
        raise ValueError('Unknown type of online embedding module.')

    def __new__(cls, source: str = None, embed_url: str = None, embed_model_name: str = None,
                return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                group_id: Optional[str] = None, type: Optional[str] = None, batch_size: int = 32, **kwargs):
        if cls._should_use_dynamic(source, dynamic_auth, skip_auth):
            return super().__new__(cls)
        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        type_name = OnlineEmbeddingModule._resolve_type_name(kwargs.pop('type', None), embed_model_name)
        if type_name == 'embed':
            source, default_key = select_source_with_default_key(lazyllm.online.embed, source, LLMType.EMBED)
        elif type_name == 'rerank':
            source, default_key = select_source_with_default_key(lazyllm.online.rerank, source, LLMType.RERANK)
        else:
            raise ValueError('Unknown type of online embedding module.')
        api_key = api_key if api_key is not None else default_key
        if skip_auth and not embed_url:
            raise KeyError('embed_url must be set for local serving.')
        params = {'embed_url': embed_url, 'embed_model_name': embed_model_name, 'return_trace': return_trace,
                  'batch_size': batch_size, 'api_key': api_key, 'skip_auth': skip_auth, **kwargs}
        return OnlineEmbeddingModule._create_supplier(source, type_name, embed_model_name, params)

    def __init__(self, source: str = None, embed_url: str = None, embed_model_name: str = None,
                 return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                 skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                 group_id: Optional[str] = None, type: Optional[str] = None, batch_size: int = 32, **kwargs):
        _DynamicSourceRouterMixin.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        self._embed_url = embed_url
        self._embed_model_name = embed_model_name
        self._type = type
        self._skip_auth = skip_auth
        self._kwargs = kwargs
        self._batch_size = batch_size
        self._init_dynamic_auth(api_key, dynamic_auth)

    def _build_supplier(self, source: str, skip_auth: bool):
        params = {'embed_url': self._embed_url, 'embed_model_name': self._embed_model_name,
                  'return_trace': self._return_trace, 'batch_size': self._batch_size,
                  'type': self._type, 'api_key': self._api_key, 'skip_auth': skip_auth, **self._kwargs}
        return OnlineEmbeddingModule._create_supplier(source, self._type, self._embed_model_name, params)
