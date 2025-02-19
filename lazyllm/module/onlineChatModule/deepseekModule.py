from urllib.parse import urljoin
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase


class DeepSeekModule(OnlineChatModuleBase):
    def __init__(self,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 api_key: str = None,
                 stream: bool = True,
                 return_trace: bool = False,
                 **kwargs):
        super().__init__(model_series="DEEPSEEK",
                         api_key=api_key or lazyllm.config['deepseek_api_key'],
                         base_url=base_url,
                         model_name=model,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace,
                         **kwargs)

    def _get_system_prompt(self):
        return "You are an intelligent assistant developed by China's DeepSeek. You are a helpful assistanti."

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, "chat/completions")
