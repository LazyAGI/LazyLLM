from typing import Optional

from .openai import OpenAIChat, OpenAIEmbed


class RuncrateChat(OpenAIChat):
    PROVIDER_NAME = 'runcrate'
    TRAINABLE_MODEL_LIST = []

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            base_url=base_url or 'https://api.runcrate.ai/v1/',
            model=model or 'deepseek-ai/DeepSeek-V4.1-Flash',
            api_key=api_key,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _get_system_prompt(self):
        return 'You are a helpful assistant accessed through Runcrate.'


class RuncrateEmbed(OpenAIEmbed):
    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, **kwargs):
        super().__init__(
            embed_url=embed_url or 'https://api.runcrate.ai/v1/',
            embed_model_name=embed_model_name or 'Qwen/Qwen3-Embedding-0.6B',
            api_key=api_key,
            **kwargs,
        )
