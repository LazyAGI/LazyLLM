from typing import Any, Dict, Optional
from .onlinemodule import (
    OnlineChatModule, OnlineEmbeddingModule, OnlineMultiModalModule,
    OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
)
from .onlinemodule.map_model_type import get_model_type


class _OnlineModuleMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        return isinstance(__instance, (OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase))


class OnlineModule(metaclass=_OnlineModuleMeta):
    '''Unified entry that routes to chat, embedding or multimodal online modules.'''

    _CHAT_TYPES = {'llm', 'vlm'}
    _EMBED_TYPES = {'embed', 'cross_modal_embed', 'rerank'}
    _MULTI_TYPE_TO_FUNCTION = {'stt': 'stt', 'tts': 'tts', 'sd': 'text2image'}
    _FUNCTION_TO_TYPE = {'stt': 'stt', 'tts': 'tts', 'text2image': 'sd', 'sd': 'sd'}

    @classmethod
    def _resolve_type(cls, model: Optional[str], type_hint: Optional[str], function_hint: Optional[str]) -> str:
        if type_hint:
            return type_hint.lower()
        if function_hint:
            func = function_hint.lower()
            if func in cls._FUNCTION_TO_TYPE:
                return cls._FUNCTION_TO_TYPE[func]
        if model:
            return (get_model_type(model) or 'llm').lower()
        return 'llm'

    @classmethod
    def _resolve_function(cls, resolved_type: str, function_hint: Optional[str]) -> Optional[str]:
        if function_hint:
            func = function_hint.lower()
            if func == 'sd':
                func = 'text2image'
            return func
        return cls._MULTI_TYPE_TO_FUNCTION.get(resolved_type)

    def __new__(self, model: Optional[str] = None, source: Optional[str] = None, *,
                type: Optional[str] = None, function: Optional[str] = None, 
                url: Optional[str] = None, **kwargs):
        params: Dict[str, Any] = dict(kwargs)
        resolved_type = self._resolve_type(model, type or params.get('type'), function or params.get('function'))

        if resolved_type in self._EMBED_TYPES:
            embed_kwargs = params.copy()
            embed_kwargs.pop('function', None)
            embed_kwargs.setdefault('type', 'rerank' if resolved_type == 'rerank' else 'embed')
            return OnlineEmbeddingModule(source=source,
                                         embed_url=url,
                                         embed_model_name=model,
                                         **embed_kwargs)

        if resolved_type in self._MULTI_TYPE_TO_FUNCTION:
            multi_kwargs = params.copy()
            multi_kwargs.pop('type', None)
            function_name = self._resolve_function(resolved_type, function or multi_kwargs.pop('function', None))
            if function_name not in {'stt', 'tts', 'text2image'}:
                raise ValueError('Invalid function for OnlineMultiModalModule.')
            return OnlineMultiModalModule(model=model, source=source, base_url=url,
                                          function=function_name, **multi_kwargs)

        chat_kwargs = params.copy()
        chat_kwargs.pop('function', None)
        chat_kwargs.setdefault('type', resolved_type)
        return OnlineChatModule(model=model, source=source, base_url=url, **chat_kwargs)
