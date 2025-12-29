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
    _MULTI_TYPES = {'stt', 'tts', 'text2image'}
    _MULTI_TYPE_ALIASES = {'sd': 'text2image', 'text2image': 'text2image', 'stt': 'stt', 'tts': 'tts'}

    @classmethod
    def _normalize_type(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        lowered = value.lower()
        if lowered in cls._MULTI_TYPE_ALIASES:
            return cls._MULTI_TYPE_ALIASES[lowered]
        return lowered

    @classmethod
    def _resolve_type(cls, model: Optional[str], type_hint: Optional[str]) -> str:
        normalized = cls._normalize_type(type_hint)
        if normalized:
            return normalized
        if model:
            inferred = cls._normalize_type(get_model_type(model))
            if inferred:
                return inferred
        return 'llm'

    def __new__(self, model: Optional[str] = None, source: Optional[str] = None, *,
                type: Optional[str] = None, url: Optional[str] = None, **kwargs):
        params: Dict[str, Any] = dict(kwargs)
        legacy_function = params.pop('function', None)
        resolved_type = self._resolve_type(model, type or params.get('type') or legacy_function)

        if resolved_type in self._EMBED_TYPES:
            embed_kwargs = params.copy()
            embed_kwargs.pop('function', None)
            embed_kwargs.setdefault('type', 'rerank' if resolved_type == 'rerank' else 'embed')
            return OnlineEmbeddingModule(source=source,
                                         embed_url=url,
                                         embed_model_name=model,
                                         **embed_kwargs)

        if resolved_type in self._MULTI_TYPES:
            multi_kwargs = params.copy()
            multi_kwargs.pop('type', None)
            return OnlineMultiModalModule(model=model, source=source, base_url=url,
                                          function=resolved_type, **multi_kwargs)

        chat_kwargs = params.copy()
        chat_kwargs.pop('function', None)
        chat_kwargs.setdefault('type', resolved_type)
        return OnlineChatModule(model=model, source=source, base_url=url, **chat_kwargs)
