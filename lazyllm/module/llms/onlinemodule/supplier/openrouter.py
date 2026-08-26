from typing import Optional

from .openai import OpenAIChat


class OpenRouterChat(OpenAIChat):
    """OpenRouter's OpenAI-compatible chat completions API."""

    PROVIDER_NAME = 'openrouter'
    TRAINABLE_MODEL_LIST = []

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            base_url=base_url or 'https://openrouter.ai/api/v1/',
            model=model or 'openrouter/auto',
            api_key=api_key,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _get_system_prompt(self):
        return 'You are a helpful assistant accessed through OpenRouter.'
