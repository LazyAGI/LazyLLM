from typing import Tuple
import requests
import lazyllm
from ..base import OnlineChatModuleBase

# PPIO (Paiou Cloud) online model module.
# PPIO provides OpenAI-compatible API interface, supporting both streaming and non-streaming responses.
class PPIOModule(OnlineChatModuleBase):
    TRAINABLE_MODEL_LIST = []
    NO_PROXY = False

    # Initialize PPIO module.
    # Args:
    #     base_url: API base URL, defaults to 'https://api.ppinfra.com/openai'
    #     model: Model name, defaults to 'deepseek/deepseek-v3.2'
    #     api_key: API key, if not provided, will be read from config
    #     stream: Whether to use streaming output, defaults to True
    #     return_trace: Whether to return execution trace, defaults to False
    #     skip_auth: Whether to skip authentication, defaults to False
    #     **kw: Other parameters
    def __init__(self, base_url: str = 'https://api.ppinfra.com/openai',
                 model: str = 'deepseek/deepseek-v3.2',
                 api_key: str = None, stream: bool = True,
                 return_trace: bool = False, skip_auth: bool = False, **kw):
        OnlineChatModuleBase.__init__(
            self,
            model_series='PPIO',
            api_key=api_key or lazyllm.config['ppio_api_key'],
            base_url=base_url,
            model_name=model,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kw
        )

    # Return PPIO system prompt.
    def _get_system_prompt(self):
        return 'You are a helpful AI assistant.'

    # Validate API key by sending a minimal chat request.
    def _validate_api_key(self):
        try:
            data = {'model': self._model_name, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1}
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    # Chat API URL - PPIO endpoint is /openai/chat/completions.
    @property
    def _chat_url(self):
        base = self._base_url.rstrip('/')
        if not base.endswith('/openai'):
            base = base + '/openai'
        return base + '/chat/completions'

    # PPIO does not support deployment, return model name and running status.
    def _create_deployment(self) -> Tuple[str, str]:
        return (self._model_name, 'RUNNING')

    # PPIO does not support deployment query, return running status.
    def _query_deployment(self, deployment_id) -> str:
        return 'RUNNING'

    def __repr__(self):
        return lazyllm.make_repr('Module', 'PPIO', name=self._model_name, url=self._base_url,
                                 stream=bool(self._stream), return_trace=self._return_trace)
