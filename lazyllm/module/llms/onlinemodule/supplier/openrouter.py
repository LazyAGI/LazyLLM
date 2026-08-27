import base64
from typing import List, Optional
from urllib.parse import urljoin

import requests

from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file

from ..base import LazyLLMOnlineText2ImageModuleBase
from .openai import OpenAIChat, OpenAIEmbed


class OpenRouterChat(OpenAIChat):
    PROVIDER_NAME = 'openrouter'
    TRAINABLE_MODEL_LIST = []

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            base_url=base_url or 'https://openrouter.ai/api/v1/',
            model=model or 'openrouter/free',
            api_key=api_key,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _get_system_prompt(self):
        return 'You are a helpful assistant accessed through OpenRouter.'


class OpenRouterEmbed(OpenAIEmbed):
    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, **kwargs):
        super().__init__(
            embed_url=embed_url or 'https://openrouter.ai/api/v1/',
            embed_model_name=embed_model_name or 'liquid/lfm-2.5-embedding-350m:free',
            api_key=api_key,
            **kwargs,
        )


class OpenRouterText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'bytedance-seed/seedream-4.5'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            api_key=api_key or self._default_api_key(),
            model=model or self.MODEL_NAME,
            url=base_url or 'https://openrouter.ai/api/v1/',
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _forward(self, input: str = None, files: List[str] = None, url: str = None,
                 model: str = None, n: int = 1, **kwargs):
        if files:
            raise ValueError('OpenRouter text-to-image does not accept reference images in this mode.')
        payload = {'model': model or self._model_name, 'prompt': input, 'n': n, **kwargs}
        response = requests.post(
            urljoin(url or self._base_url, 'images'),
            headers=self._header,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

        image_bytes = []
        for item in response.json().get('data', []):
            if item.get('b64_json'):
                encoded = item['b64_json'].split(',', 1)[-1]
                image_bytes.append(base64.b64decode(encoded, validate=True))
            elif item.get('url'):
                image_bytes.append(self._load_images(item['url'])[0][1])
        if not image_bytes:
            raise RuntimeError('OpenRouter image API returned no images.')
        return encode_query_with_filepaths(None, bytes_to_file(image_bytes))
