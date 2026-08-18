from lazyllm import LOG
from typing import Optional
from ..base import ModelFailureCode, ModelFinish, OnlineChatModuleBase
from ..base.provider_error_mapping import register_provider_error_mapping


register_provider_error_mapping(
    'deepseek',
    extends='openai_compatible',
    http_map={
        402: ModelFailureCode.QUOTA_EXHAUSTED,
        422: ModelFailureCode.INVALID_REQUEST,
        429: ModelFailureCode.RATE_LIMITED,
        503: ModelFailureCode.PROVIDER_OVERLOADED,
    },
)


class DeepSeekChat(OnlineChatModuleBase):
    _PROVIDER_SOURCE = 'deepseek'
    _FINISH_REASON_MAP = {
        **OnlineChatModuleBase._FINISH_REASON_MAP,
        'insufficient_system_resource': ModelFinish.INSUFFICIENT_SYSTEM_RESOURCE,
    }

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
