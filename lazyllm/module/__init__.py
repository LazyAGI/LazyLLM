from .module import (ModuleBase, TrainableModule, ActionModule,
                     ServerModule, UrlModule, register)
from .trialmodule import TrialModule
from .onlineChatModule import OnlineChatModule, OnlineChatModuleBase
from .onlineEmbedding import OnlineEmbeddingModule, OnlineEmbeddingModuleBase
from .automodel import AutoModel
from .onlineMultiModule import OnlineMultiModule, OnlineMultiModuleBase

import lazyllm
# openai api key
lazyllm.config.add("openai_api_key", str, "", "OPENAI_API_KEY")
# openai tts model name
lazyllm.config.add("openai_tts_model_name", str, "", "OPENAI_TTS_MODEL_NAME")
# openai text2image model name
lazyllm.config.add("openai_text2image_model_name", str, "", "OPENAI_TEXT2IMAGE_MODEL_NAME")
# openai stt model name
lazyllm.config.add("openai_stt_model_name", str, "", "OPENAI_STT_MODEL_NAME")
# kimi api key
lazyllm.config.add("kimi_api_key", str, "", "KIMI_API_KEY")
# glm api key
lazyllm.config.add("glm_api_key", str, "", "GLM_API_KEY")
# glm model name
lazyllm.config.add("glm_model_name", str, "", "GLM_MODEL_NAME")
# glm stt model name
lazyllm.config.add("glm_stt_model_name", str, "", "GLM_STT_MODEL_NAME")
# qwen api key
lazyllm.config.add("qwen_api_key", str, "", "QWEN_API_KEY")
# qwen model name
lazyllm.config.add("qwen_model_name", str, "", "QWEN_MODEL_NAME")
# qwen stt model name
lazyllm.config.add("qwen_stt_model_name", str, "", "QWEN_STT_MODEL_NAME")
# qwen tts model name
lazyllm.config.add("qwen_tts_model_name", str, "", "QWEN_TTS_MODEL_NAME")
# qwen text2image model name
lazyllm.config.add("qwen_text2image_model_name", str, "", "QWEN_TEXT2IMAGE_MODEL_NAME")
# sensenova ak sk
lazyllm.config.add("sensenova_api_key", str, "", "SENSENOVA_API_KEY")
lazyllm.config.add("sensenova_secret_key", str, "", "SENSENOVA_SECRET_KEY")
# doubao api key
lazyllm.config.add("doubao_api_key", str, "", "DOUBAO_API_KEY")
# doubao model name
lazyllm.config.add("doubao_model_name", str, "", "DOUBAO_MODEL_NAME")
# doubao text2image model name
lazyllm.config.add("doubao_text2image_model_name", str, "", "DOUBAO_TEXT2IMAGE_MODEL_NAME")
# deepseek api key
lazyllm.config.add("deepseek_api_key", str, "", "DEEPSEEK_API_KEY")

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
    "OnlineMultiModule",
    "OnlineMultiModuleBase",
    "AutoModel",
]
