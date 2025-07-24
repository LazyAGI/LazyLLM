from .automodel import AutoModel
from .onlinemodule import OnlineEmbeddingModule, OnlineEmbeddingModuleBase
from .onlinemodule import OnlineChatModule, OnlineChatModuleBase
from .trainablemodule import TrainableModule
from .onlinemodule import OnlineMultiModalModule, OnlineMultiModalBase
from lazyllm import config


__all__ = [
    "AutoModel",
    "OnlineEmbeddingModule",
    "OnlineEmbeddingModuleBase",
    "OnlineChatModule",
    "OnlineChatModuleBase",
    "TrainableModule",
    "OnlineMultiModalModule",
    "OnlineMultiModalBase",
]


for key in OnlineChatModule.MODELS.keys():
    config.add(f"{key}_api_key", str, "", f"{key.upper()}_API_KEY")
    config.add(f"{key}_model_name", str, "", f"{key.upper()}_MODEL_NAME")
    config.add(f"{key}_text2image_model_name", str, "", f"{key.upper()}_TEXT2IMAGE_MODEL_NAME")
    config.add(f"{key}_tts_model_name", str, "", f"{key.upper()}_TTS_MODEL_NAME")
    config.add(f"{key}_stt_model_name", str, "", f"{key.upper()}_STT_MODEL_NAME")

config.add("sensenova_secret_key", str, "", "SENSENOVA_SECRET_KEY")
