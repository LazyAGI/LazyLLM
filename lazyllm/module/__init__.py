from .module import (ModuleBase, TrainableModule, ActionModule,
                     ServerModule, UrlModule, register)
from .trialmodule import TrialModule
from .onlineChatModule import OnlineChatModule, OnlineChatModuleBase
from .onlineEmbedding import OnlineEmbeddingModule, OnlineEmbeddingModuleBase

import lazyllm
# openai api key
lazyllm.config.add("openai_api_key", str, "", "OPENAI_API_KEY")
# kimi api key
lazyllm.config.add("moonshotai_api_key", str, "", "MOONSHOTAI_API_KEY")
# glm api key
lazyllm.config.add("glm_api_key", str, "", "GLM_API_KEY")
# qwen api key
lazyllm.config.add("qwen_api_key", str, "", "DASHSCOPE_API_KEY")
# sensenova ak sk
lazyllm.config.add("sensenova_ak", str, "", "SENSENOVA_ACCESS_KEY_ID")
lazyllm.config.add("sensenova_sk", str, "", "SENSENOVA_ACCESS_KEY_SECRET")
# doubao api key
lazyllm.config.add("doubao_api_key", str, "", "DOUBAO_API_KEY")

__all__ = [
    'register',
    'ModuleBase',
    'UrlModule',
    'TrainableModule',
    'ActionModule',
    'ServerModule',
    'TrialModule',
    "OnlineChatModule",
    "OnlineChatModuleBase",
    "OnlineEmbeddingModule",
    "OnlineEmbeddingModuleBase"
]
