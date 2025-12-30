from typing import Any, Dict, Optional
from .onlinemodule import (
    OnlineChatModule, OnlineEmbeddingModule, OnlineMultiModalModule,
    OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
)
from lazyllm.components.utils.downloader.model_downloader import LLMType
from .onlinemodule.map_model_type import get_model_type


class _OnlineModuleMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        return isinstance(__instance, (OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase))


class OnlineModule(metaclass=_OnlineModuleMeta):
    '''Unified entry that routes to chat, embedding or multimodal online modules.'''

    _EMBED_TYPES = (LLMType.EMBED, LLMType.CROSS_MODAL_EMBED, LLMType.RERANK)
    _MULTI_TYPE_TO_FUNCTION = {
        LLMType.STT: 'stt',
        LLMType.TTS: 'tts',
        LLMType.SD: 'text2image',
        LLMType.TEXT2IMAGE: 'text2image',
    }

    @staticmethod
    def _transfer_type_to_LLMType(value: Optional[Any]) -> Optional[LLMType]:
        if not value:
            return None
        if isinstance(value, LLMType):
            return value
        try:
            return LLMType(value)
        except (ValueError, TypeError):
            raise Exception(f'The type {value} is not supported in OnlineModule.')

    def __new__(self, model: Optional[str] = None, source: Optional[str] = None, *,
                type: Optional[str] = None, url: Optional[str] = None, **kwargs):
        params: Dict[str, Any] = dict(kwargs)
        legacy_function = params.pop('function', None)
        type_hint = type or legacy_function
        if type_hint:
            resolved_type = self._transfer_type_to_LLMType(type_hint)
        else:
            inferred_type = (get_model_type(model) or 'llm') if model else 'llm'
            resolved_type = self._transfer_type_to_LLMType(inferred_type)
        resolved_type = resolved_type or LLMType.LLM

        if resolved_type in self._EMBED_TYPES:
            embed_kwargs = params.copy()
            embed_kwargs.pop('function', None)
            embed_kwargs.setdefault('type', 'rerank' if resolved_type == LLMType.RERANK else 'embed')
            return OnlineEmbeddingModule(source=source,
                                         embed_url=url,
                                         embed_model_name=model,
                                         **embed_kwargs)

        if resolved_type in self._MULTI_TYPE_TO_FUNCTION:
            multi_kwargs = params.copy()
            multi_kwargs.pop('type', None)
            return OnlineMultiModalModule(model=model, source=source, base_url=url,
                                          function=self._MULTI_TYPE_TO_FUNCTION[resolved_type], **multi_kwargs)

        chat_kwargs = params.copy()
        chat_kwargs.pop('function', None)
        chat_kwargs.setdefault('type', resolved_type.value.lower())
        return OnlineChatModule(model=model, source=source, base_url=url, **chat_kwargs)
