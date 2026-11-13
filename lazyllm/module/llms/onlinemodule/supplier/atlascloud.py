from ..base import OnlineChatModuleBase


class AtlasCloudChat(OnlineChatModuleBase):
    PROVIDER_NAME = 'atlascloud'

    def __init__(self, base_url: str = None, model: str = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.atlascloud.ai/v1/'
        model = model or 'openai/gpt-4.1-mini'
        super().__init__(api_key=api_key or self._default_api_key(),
                         base_url=base_url, model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are a helpful assistant accessed through Atlas Cloud.'
