from typing import Optional

from ..base import OnlineChatModuleBase


class OrcaRouterChat(OnlineChatModuleBase):
    PROVIDER_NAME = 'orcarouter'

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = False, **kw):
        base_url = base_url or 'https://api.orcarouter.ai/v1/'
        model = model or 'orcarouter/auto'
        super().__init__(api_key=api_key or self._default_api_key(),
                         base_url=base_url, model_name=model, stream=stream,
                         return_trace=return_trace, skip_auth=skip_auth, **kw)

    def _get_system_prompt(self):
        return 'You are a helpful assistant.'
