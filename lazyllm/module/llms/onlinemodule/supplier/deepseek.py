from urllib.parse import urljoin
import lazyllm
from ..base import OnlineChatModuleBase


class DeepSeekModule(OnlineChatModuleBase):
    """DeepSeek large language model interface module.

Args:
    base_url (str): API base URL, defaults to "https://api.deepseek.com"
    model (str): Model name, defaults to "deepseek-chat"
    api_key (str): API key, if None, gets from configuration
    stream (bool): Whether to enable streaming output, defaults to True
    return_trace (bool): Whether to return trace information, defaults to False
    **kwargs: Other parameters passed to base class
"""
    def __init__(self, base_url: str = 'https://api.deepseek.com', model: str = 'deepseek-chat',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='DEEPSEEK', api_key=api_key or lazyllm.config['deepseek_api_key'],
                         base_url=base_url, model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by China\'s DeepSeek. You are a helpful assistanti.'

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')
