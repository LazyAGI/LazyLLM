from urllib.parse import urljoin
import requests
import lazyllm
from ..base import OnlineChatModuleBase


REGISTRY_KEY = 'deepseek'


class DeepSeekModule(OnlineChatModuleBase):
    __lazyllm_registry_key__ = REGISTRY_KEY

    def __init__(self, base_url: str = 'https://api.deepseek.com', model: str = 'deepseek-chat',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='DEEPSEEK', api_key=api_key or lazyllm.config['deepseek_api_key'],
                         base_url=base_url, model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by China\'s DeepSeek. You are a helpful assistanti.'

    def _validate_api_key(self):
        try:
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
