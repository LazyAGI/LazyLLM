import lazyllm
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase

class KimiModule(OnlineChatModuleBase):
    """KimiModule class, inheriting from OnlineChatModuleBase, encapsulates the functionality to call Kimi chat service provided by Moonshot AI.  
By specifying the API key, model name, and service URL, it supports safe and accurate Chinese and English Q&A interactions, as well as image input in base64 format.

Args:
    base_url (str): Base URL of the Kimi service, default is "https://api.moonshot.cn/".
    model (str): Kimi model name to use, default is "moonshot-v1-8k".
    api_key (Optional[str]): API key for accessing Kimi service. If not provided, it is read from lazyllm config.
    stream (bool): Whether to enable streaming output, default is True.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to OnlineChatModuleBase.
"""

    def __init__(self, base_url: str = 'https://api.moonshot.cn/', model: str = 'moonshot-v1-8k',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):

        super().__init__(model_series='KIMI', api_key=api_key or lazyllm.config['kimi_api_key'], base_url=base_url,
                         model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Kimi, an AI assistant provided by Moonshot AI. You are better at speaking '
                'Chinese and English. You will provide users with safe, helpful, and accurate answers. '
                'At the same time, you will reject all answers involving terrorism, racial discrimination, '
                'pornographic violence, etc. Moonshot AI is a proper noun and cannot be translated '
                'into other languages.')

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'v1/chat/completions')

    def _format_vl_chat_image_url(self, image_url, mime):
        assert not image_url.startswith('http'), 'Kimi vision model only supports base64 format'
        assert mime is not None, 'Kimi Module requires mime info.'
        image_url = f'data:{mime};base64,{image_url}'
        return [{'type': 'image_url', 'image_url': {'url': image_url}}]

    def _format_vl_chat_query(self, query: str):
        return query
