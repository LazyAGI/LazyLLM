from urllib.parse import urljoin
import requests
from lazyllm import LOG
from typing import Optional
from ..base import OnlineChatModuleBase


class DeepSeekChat(OnlineChatModuleBase):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.deepseek.com'
        model = model or 'deepseek-chat'
        if model in ('deepseek-chat', 'deepseek-reasoner'):
            LOG.warning(
                f'Model "{model}" is deprecated and will be removed after 2026/07/24. '
                'Please use "deepseek-v4-flash" or "deepseek-v4-pro" instead.')
        super().__init__(api_key=api_key or self._default_api_key(),
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
