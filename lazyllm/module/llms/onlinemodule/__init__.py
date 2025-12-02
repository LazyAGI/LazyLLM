from .chat import OnlineChatModule
from .embedding import OnlineEmbeddingModule
from .multimodal import OnlineMultiModalModule
from .base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase


class OnlineModule():
    pass


__all__ = [
    'OnlineChatModule',
    'OnlineEmbeddingModule',
    'OnlineMultiModalModule',
    'OnlineChatModuleBase',
    'OnlineEmbeddingModuleBase',
    'OnlineMultiModalBase',
]
