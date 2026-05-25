from .automodel import AutoModel
from .onlinemodule import (
    OnlineEmbeddingModule, OnlineEmbeddingModuleBase,
    OnlineChatModule, OnlineChatModuleBase,
    OnlineMultiModalModule, OnlineMultiModalBase
)
from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from .model_config_inject import inject_model_config
from lazyllm import namespace

namespace.register_module(['AutoModel', 'OnlineModule', 'OnlineChatModule',
                           'OnlineEmbeddingModule', 'OnlineMultiModalModule'])


__all__ = [
    'AutoModel',
    'OnlineModule',
    'OnlineEmbeddingModule',
    'OnlineEmbeddingModuleBase',
    'OnlineChatModule',
    'OnlineChatModuleBase',
    'TrainableModule',
    'OnlineMultiModalModule',
    'OnlineMultiModalBase',
    'inject_model_config',
]
