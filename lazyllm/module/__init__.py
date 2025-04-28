from .module import (ModuleBase, TrainableModule, ActionModule,
                     ServerModule, UrlModule, register)
from .trialmodule import TrialModule
from .onlineChatModule import OnlineChatModule, OnlineChatModuleBase
from .onlineEmbedding import OnlineEmbeddingModule, OnlineEmbeddingModuleBase
from .automodel import AutoModel

import lazyllm
# openai api key
lazyllm.config.add("openai_api_key", str, "", "OPENAI_API_KEY")
# kimi api key
lazyllm.config.add("kimi_api_key", str, "", "KIMI_API_KEY")
# glm api key
lazyllm.config.add("glm_api_key", str, "", "GLM_API_KEY")
# glm model name
lazyllm.config.add("glm_model_name", str, "", "GLM_MODEL_NAME")
# qwen api key
lazyllm.config.add("qwen_api_key", str, "", "QWEN_API_KEY")
# qwen model name
lazyllm.config.add("qwen_model_name", str, "", "QWEN_MODEL_NAME")
# sensenova ak sk
lazyllm.config.add("sensenova_api_key", str, "", "SENSENOVA_API_KEY")
lazyllm.config.add("sensenova_secret_key", str, "", "SENSENOVA_SECRET_KEY")
# doubao api key
lazyllm.config.add("doubao_api_key", str, "", "DOUBAO_API_KEY")
# doubao model name
lazyllm.config.add("doubao_model_name", str, "", "DOUBAO_MODEL_NAME")
# deepseek api key
lazyllm.config.add("deepseek_api_key", str, "", "DEEPSEEK_API_KEY")
# https proxy
lazyllm.config.add("http_proxy", str, "", "HTTP_PROXY")
lazyllm.config.add("https_proxy", str, "", "HTTPS_PROXY")

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
    "OnlineEmbeddingModuleBase",
    "AutoModel",
]
