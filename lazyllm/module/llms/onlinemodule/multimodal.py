import lazyllm
from lazyllm.components.utils.downloader.model_downloader import LLMType
from typing import Any, ContextManager, List, Optional, Union
from lazyllm.common.bind import _MetaBind
from lazyllm.module import ModuleBase
from .base import OnlineMultiModalBase
from .base.utils import select_source_with_default_key
from .map_model_type import get_model_type
from .dynamic_router import _DynamicSourceRouterMixin, dynamic_model_config_context


def dynamic_multimodal_config(
    modules: Optional[Union[Any, List[Any]]] = None,
    *,
    source: Optional[str] = None,
    model: Optional[str] = None,
    url: Optional[str] = None,
    skip_auth: Optional[bool] = None,
) -> ContextManager[None]:
    return dynamic_model_config_context('multimodal', modules, source=source, model=model, url=url,
                                        skip_auth=skip_auth)


class _OnlineMultiModalMeta(_MetaBind):
    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineMultiModalBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineMultiModalModule(ModuleBase, _DynamicSourceRouterMixin, metaclass=_OnlineMultiModalMeta):
    _dynamic_module_slot = 'multimodal'
    _dynamic_source_error = 'No source is configured for dynamic multimodal source.'
    TYPE_GROUP_MAP = {
        'stt': LLMType.STT,
        'tts': LLMType.TTS,
        'text2image': LLMType.TEXT2IMAGE,
        'image_editing': LLMType.TEXT2IMAGE,
    }

    @staticmethod
    def _resolve_type_name(type_name: Optional[str], model: Optional[str]) -> str:
        if type_name is not None:
            return LLMType._normalize(type_name)
        resolved = get_model_type(model) if model else None
        if resolved == 'sd':
            return 'text2image'
        assert resolved in OnlineMultiModalModule.TYPE_GROUP_MAP, 'type must be provided for OnlineMultiModalModule.'
        return resolved

    @staticmethod
    def _validate_parameters(source: Optional[str], model: Optional[str], type: str, base_url: Optional[str],
                             skip_auth: bool = False, **kwargs) -> tuple:
        assert type in OnlineMultiModalModule.TYPE_GROUP_MAP, f'Invalid type: {type}'
        if model in lazyllm.online[type] and source is None:
            source, model = model, source
        register_type = OnlineMultiModalModule.TYPE_GROUP_MAP.get(type).lower()
        source, default_key = select_source_with_default_key(lazyllm.online[register_type], source, type)
        if default_key and not kwargs.get('api_key'):
            kwargs['api_key'] = default_key
        if skip_auth and not base_url:
            raise KeyError('base_url must be set for local serving.')
        default_module_cls = getattr(lazyllm.online[register_type], source)
        default_model_name = getattr(default_module_cls, 'IMAGE_EDITING_MODEL_NAME' if type == 'image_editing'
                                     else 'MODEL_NAME', None)
        if model is None and default_model_name:
            model = default_model_name
            lazyllm.LOG.info(f'For type {type}, source {source}. Automatically selected default model: {model}')
        if base_url is not None:
            kwargs['base_url'] = base_url
        return source, model, kwargs

    def __new__(cls, model: str = None, source: str = None, type: str = None, base_url: str = None,
                return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                group_id: Optional[str] = None, **kwargs):
        if cls._should_use_dynamic(source, dynamic_auth, skip_auth):
            return super().__new__(cls)
        if source is None and api_key is not None:
            raise ValueError('No source is given but an api_key is provided.')
        if api_key is not None:
            kwargs['api_key'] = api_key
        type = OnlineMultiModalModule._resolve_type_name(
            type if type is not None else kwargs.pop('function', None), model)
        source, model, kwargs_normalized = OnlineMultiModalModule._validate_parameters(
            source=source, model=model, type=type, base_url=base_url, skip_auth=skip_auth, **kwargs)
        params = {'return_trace': return_trace, 'type': type}
        if model is not None:
            params['model'] = model
        params.update(kwargs_normalized)
        register_type = OnlineMultiModalModule.TYPE_GROUP_MAP.get(type).lower()
        return getattr(lazyllm.online[register_type], source)(**params)

    def __init__(self, model: str = None, source: str = None, type: str = None, base_url: str = None,
                 return_trace: bool = False, api_key: str = None, dynamic_auth: bool = False,
                 skip_auth: bool = False, id: Optional[str] = None, name: Optional[str] = None,
                 group_id: Optional[str] = None, **kwargs):
        ModuleBase.__init__(self, id=id, name=name, group_id=group_id, return_trace=return_trace)
        self._model_name = model
        self._base_url = base_url
        self._skip_auth = skip_auth
        self._type = self._resolve_type_name(type, model)
        self._kwargs = kwargs
        self._init_dynamic_auth(api_key, dynamic_auth)

    def _build_supplier(self, source: str, skip_auth: bool):
        params = {'base_url': self._base_url, 'model': self._model_name, 'return_trace': self._return_trace,
                  'type': self._type, 'api_key': self._api_key, 'skip_auth': skip_auth, **self._kwargs}
        register_type = OnlineMultiModalModule.TYPE_GROUP_MAP.get(self._type).lower()
        return getattr(lazyllm.online[register_type], source)(**params)

    def forward(self, *args, **kwargs):
        return _DynamicSourceRouterMixin.forward(self, *args, **kwargs)
