import lazyllm
from urllib.parse import urljoin
from .onlineChatModuleBase import OnlineChatModuleBase

class DoubaoModule(OnlineChatModuleBase):
    MODEL_NAME = "doubao-1-5-pro-32k-250115"

    def __init__(self,
                 model: str = None,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3/",
                 api_key: str = None,
                 stream: bool = True,
                 return_trace: bool = False):
        super().__init__(model_series="DOUBAO",
                         api_key=api_key or lazyllm.config['doubao_api_key'],
                         base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoModule.MODEL_NAME,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace)

    def _get_system_prompt(self):
        return ("You are Doubao, an AI assistant. Your task is to provide appropriate responses "
                "and support to users' questions and requests.")

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')
