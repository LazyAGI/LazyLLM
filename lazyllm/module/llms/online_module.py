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
        LLMType.IMAGE_EDITING: 'text2image',   
    }

    def __new__(self, model: Optional[str] = None, source: Optional[str] = None, *,
                type: Optional[str] = None, url: Optional[str] = None, **kwargs):
        params: Dict[str, Any] = dict(kwargs)
        resolved_type = type or params.pop('function', None)
        if not resolved_type:
            resolved_type = (get_model_type(model) or 'llm') if model else 'llm'
        resolved_type = resolved_type.lower()

        if resolved_type in self._EMBED_TYPES:
            embed_kwargs = params.copy()
            embed_kwargs.pop('function', None)
            embed_kwargs.setdefault('type', 'rerank' if resolved_type == LLMType.RERANK else 'embed')
            return OnlineEmbeddingModule(source=source,
                                         embed_url=url,
                                         embed_model_name=model,
                                         **embed_kwargs)
        
        if resolved_type in list(self._MULTI_TYPE_TO_FUNCTION.keys()):
            multi_kwargs = params.copy()
            multi_kwargs.pop('function', None)
            multi_kwargs.setdefault('type', resolved_type)
            return OnlineMultiModalModule(model=model, source=source, base_url=url,
                                          function=self._MULTI_TYPE_TO_FUNCTION[LLMType(resolved_type)],
                                          **multi_kwargs)

        chat_kwargs = params.copy()
        chat_kwargs.pop('function', None)
        chat_kwargs.setdefault('type', resolved_type)
        return OnlineChatModule(model=model, source=source, base_url=url, **chat_kwargs)
