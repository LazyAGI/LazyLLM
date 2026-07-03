from typing import Any, ContextManager, List, Optional, Union

import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from lazyllm.common.bind import _MetaBind
from .base import OnlineEmbeddingModuleBase
from .base.utils import select_source_with_default_key, resolve_online_params
from .supplier.doubao import DoubaoEmbed, DoubaoMultimodalEmbed
from .supplier.qwen import QwenMultimodalEmbed
from .supplier.siliconflow import SiliconFlowMultimodalEmbed
from .map_model_type import get_model_type
from .dynamic_router import _DynamicSourceRouterMixin, dynamic_model_config_context


def _is_qwen_multimodal_embed_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    model_name = model_name.lower()
    return (model_name.endswith('vl-embedding')
            or 'embedding-vision' in model_name
            or model_name.startswith('multimodal-embedding')
        )


def _is_siliconflow_multimodal_embed_model(model_name: Optional[str]) -> bool:
    return bool(model_name and model_name.lower() == 'qwen/qwen3-vl-embedding-8b')


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
            if type_name == LLMType.CROSS_MODAL_EMBED:
                return 'cross_modal_embed'
            return type_name
        resolved = get_model_type(embed_model_name) if embed_model_name else 'embed'
        return resolved if resolved in ('embed', 'rerank', 'cross_modal_embed') else 'embed'

    @staticmethod
    def _create_supplier(source: str, type_name: str, embed_model_name: str, params: dict):
        if type_name == 'cross_modal_embed':
            if source == 'doubao':
                return DoubaoMultimodalEmbed(**params)
            if source == 'qwen':
                return QwenMultimodalEmbed(**params)
            if source == 'siliconflow':
                return SiliconFlowMultimodalEmbed(**params)
            # OpenAI-compatible self-hosted cross-modal embedding (e.g. siglip via source=openai).
            if source in lazyllm.online.embed:
                return getattr(lazyllm.online.embed, source)(**params)
            raise ValueError(f'Source {source!r} does not support CROSS_MODAL_EMBED.')
        if type_name == 'embed':
            if source == 'doubao' and embed_model_name and embed_model_name.startswith('doubao-embedding-vision'):
                return DoubaoMultimodalEmbed(**params)
            if source == 'qwen' and _is_qwen_multimodal_embed_model(embed_model_name):
                return QwenMultimodalEmbed(**params)
            if source == 'siliconflow' and _is_siliconflow_multimodal_embed_model(embed_model_name):
                return SiliconFlowMultimodalEmbed(**params)
            if source == 'doubao':
                return DoubaoEmbed(**params)
            return getattr(lazyllm.online.embed, source)(**params)
        if type_name == 'rerank':
            return getattr(lazyllm.online.rerank, source)(**params)
        raise ValueError('Unknown type of online embedding module.')

    @staticmethod
    def _is_embed_source(name: str) -> bool:
        return name in lazyllm.online.embed or name in lazyllm.online.rerank

    def __new__(cls, model: str = None, source: str = None, url: str = None,
                return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                group_id: Optional[str] = None, type: Optional[str] = None, batch_size: int = 32,
                num_worker: int = 4, **kwargs):
        model, source, url, kwargs = resolve_online_params(
            model, source, url, kwargs,
            model_aliases=('embed_model_name', 'model_name'), url_aliases=('embed_url', 'base_url'),
            source_registry=OnlineEmbeddingModule._is_embed_source)
        if cls._should_use_dynamic(source, dynamic_auth, skip_auth):
            return super().__new__(cls)
        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        type_name = OnlineEmbeddingModule._resolve_type_name(type, model)
        if type_name in ('embed', 'cross_modal_embed'):
            source, default_key = select_source_with_default_key(lazyllm.online.embed, source, LLMType.EMBED)
        elif type_name == 'rerank':
            source, default_key = select_source_with_default_key(lazyllm.online.rerank, source, LLMType.RERANK)
        else:
            raise ValueError('Unknown type of online embedding module.')
        api_key = api_key if api_key is not None else default_key
        if skip_auth and not url:
            raise ValueError('url must be set for local serving.')
        params = {'embed_url': url, 'embed_model_name': model, 'return_trace': return_trace,
                  'batch_size': batch_size, 'num_worker': num_worker,
                  'api_key': api_key, 'skip_auth': skip_auth, **kwargs}
        return OnlineEmbeddingModule._create_supplier(source, type_name, model, params)

    def __init__(self, model: str = None, source: str = None, url: str = None,
                 return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                 skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                 group_id: Optional[str] = None, type: Optional[str] = None, batch_size: int = 32,
                 num_worker: int = 4, **kwargs):
        model, source, url, kwargs = resolve_online_params(
            model, source, url, kwargs,
            model_aliases=('embed_model_name', 'model_name'), url_aliases=('embed_url', 'base_url'),
            source_registry=OnlineEmbeddingModule._is_embed_source)
        _DynamicSourceRouterMixin.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        self._embed_url = url
        self._embed_model_name = model
        if source == 'dynamic' and type is None:
            raise ValueError('type must be explicitly provided when source is dynamic.')
        self._type = OnlineEmbeddingModule._resolve_type_name(type, model)
        self._skip_auth = skip_auth
        self._kwargs = kwargs
        self._kwargs.setdefault('num_worker', num_worker)
        self._batch_size = batch_size
        self._init_dynamic_auth(api_key, dynamic_auth)

    # Expose batch_size on the dynamic router (which subclasses ModuleBase, not
    # OnlineEmbeddingModuleBase). Without it, parallel_do_embedding's _check_batch
    # cannot detect batch capability and falls back to one concurrent request per
    # node, flooding the provider and triggering rate limits.
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    def _build_supplier(self, source: str, skip_auth: bool):
        params = {'embed_url': self._embed_url, 'embed_model_name': self._embed_model_name,
                  'return_trace': self._return_trace, 'batch_size': self._batch_size,
                  'api_key': self._api_key, 'skip_auth': skip_auth, **self._kwargs}
        return OnlineEmbeddingModule._create_supplier(source, self._type, self._embed_model_name, params)
