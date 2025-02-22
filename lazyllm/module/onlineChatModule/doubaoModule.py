import lazyllm
from urllib.parse import urljoin
from .onlineChatModuleBase import OnlineChatModuleBase

class DoubaoModule(OnlineChatModuleBase):

    def __init__(self,
                 model: str = "mistral-7b-instruct-v0.2",
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3/",
                 api_key: str = None,
                 stream: bool = True,
                 return_trace: bool = False):
        super().__init__(model_series="DOUBAO",
                         api_key=api_key or lazyllm.config['doubao_api_key'],
                         base_url=base_url,
                         model_name=model,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace)
        if not model:
            raise ValueError("Doubao model must be specified.")

    def _get_system_prompt(self):
        return ("You are Doubao, an AI assistant. Your task is to provide appropriate responses "
                "and support to users' questions and requests.")

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')
