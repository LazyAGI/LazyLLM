from .automodel import AutoModel
from .onlinemodule import (
    OnlineEmbeddingModule, OnlineEmbeddingModuleBase,
    OnlineChatModule, OnlineChatModuleBase,
    OnlineMultiModalModule, OnlineMultiModalBase
)
from .online_module import OnlineModule
from .trainablemodule import TrainableModule
from lazyllm import config, namespace

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
]


for key in OnlineChatModule.MODELS.keys():
    config.add(f'{key}_api_key', str, '', f'{key.upper()}_API_KEY', description=f'The API key for {key}.')
    config.add(f'{key}_model_name', str, '', f'{key.upper()}_MODEL_NAME',
               description=f'The default model name for {key}.')
    config.add(f'{key}_text2image_model_name', str, '', f'{key.upper()}_TEXT2IMAGE_MODEL_NAME',
               description=f'The default text2image model name for {key}.')
    config.add(f'{key}_tts_model_name', str, '', f'{key.upper()}_TTS_MODEL_NAME',
               description=f'The default tts model name for {key}.')
    config.add(f'{key}_stt_model_name', str, '', f'{key.upper()}_STT_MODEL_NAME',
               description=f'The default stt model name for {key}.')

config.add('sensenova_secret_key', str, '', 'SENSENOVA_SECRET_KEY', description='The secret key for SenseNova.')
