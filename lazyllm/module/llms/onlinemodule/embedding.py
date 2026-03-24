from typing import Any, Dict, Optional

import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from lazyllm import globals
from .base import OnlineEmbeddingModuleBase
from .base.utils import select_source_with_default_key
from .supplier.doubao import DoubaoEmbed, DoubaoMultimodalEmbed
from .map_model_type import get_model_type
from .dynamic_router import DynamicSourceRouterMixin


class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


globals.config.add('dynamic_embedding_source', str, None, 'DYNAMIC_EMBEDDING_SOURCE',
                   description='The embedding source to use defined in session scope.')


class OnlineEmbeddingModule(DynamicSourceRouterMixin, metaclass=__EmbedModuleMeta):
    _dynamic_source_config = 'dynamic_embedding_source'
    _dynamic_source_error = 'No source is configured for dynamic embedding source.'

    @staticmethod
    def _encapsulate_parameters(embed_url: str, embed_model_name: str, **kwargs) -> Dict[str, Any]:
        params = {}
        if embed_url is not None:
            params['embed_url'] = embed_url
        if embed_model_name is not None:
            params['embed_model_name'] = embed_model_name
        params.update(kwargs)
        return params

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
                api_key: str = None, dynamic_auth: bool = False, skip_auth: bool = False, **kwargs):
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
        params = OnlineEmbeddingModule._encapsulate_parameters(
            embed_url, embed_model_name, api_key=api_key, skip_auth=skip_auth, **kwargs
        )
        return OnlineEmbeddingModule._create_supplier(source, type_name, embed_model_name, params)

    def __init__(self, source: str = None, embed_url: str = None, embed_model_name: str = None,
                 api_key: str = None, dynamic_auth: bool = False, skip_auth: bool = False, **kwargs):
        self._embed_url = embed_url
        self._embed_model_name = embed_model_name
        self._type_name = OnlineEmbeddingModule._resolve_type_name(kwargs.pop('type', None), embed_model_name)
        self._skip_auth = skip_auth
        self._kwargs = kwargs
        self._init_dynamic_auth(api_key, dynamic_auth)

    def _build_supplier(self, source: str):
        params = OnlineEmbeddingModule._encapsulate_parameters(
            self._embed_url, self._embed_model_name, api_key=self._api_key,
            skip_auth=self._skip_auth, **self._kwargs
        )
        return OnlineEmbeddingModule._create_supplier(source, self._type_name, self._embed_model_name, params)
