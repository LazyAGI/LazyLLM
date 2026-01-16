import requests
import lazyllm
from typing import Tuple, List, Dict, Union
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineRerankModuleBase, LazyLLMOnlineText2ImageModuleBase
)
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase

TIMEOUT = 300

class AipingChat(OnlineChatModuleBase, FileHandlerBase):
    VLM_MODEL_PREFIX = [
        'Qwen2.5-VL-',
        'Qwen3-VL-',
        'GLM-4.5V',
        'GLM-4.6V'
    ]

    def __init__(self, base_url: str = 'https://aiping.cn/api/v1/', model: str = 'DeepSeek-R1',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='AIPING',
                                      api_key=api_key or lazyllm.config['aiping_api_key'],
                                      base_url=base_url, model_name=model, stream=stream,
                                      return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by AIPing. You are a helpful assistant.'

    def _validate_api_key(self):
        try:
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=TIMEOUT)
            return response.status_code == 200
        except Exception:
            return False


class AipingEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self, embed_url: str = 'https://aiping.cn/api/v1/embeddings',
                 embed_model_name: str = 'text-embedding-v1', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__('AIPING', embed_url, api_key or lazyllm.config['aiping_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class AipingRerank(LazyLLMOnlineRerankModuleBase):
    def __init__(self, embed_url: str = 'https://aiping.cn/api/v1/rerank',
                 embed_model_name: str = 'Qwen3-Reranker-0.6B', api_key: str = None, **kw):
        super().__init__('AIPING', embed_url, api_key or lazyllm.config['aiping_api_key'],
                         embed_model_name, **kw)

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            'model': self._embed_model_name,
            'query': query,
            'documents': documents,
            'top_n': top_n
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        results = response.get('results', [])
        if not results:
            return []
        return [(result['index'], result['relevance_score']) for result in results]


class AipingText2Image(LazyLLMOnlineText2ImageModuleBase):
    def __init__(self, api_key: str = None, model_name: str = 'Qwen-Image',
                 base_url: str = 'https://aiping.cn/api/v1/',
                 return_trace: bool = False, **kwargs):
        LazyLLMOnlineText2ImageModuleBase.__init__(self, model_series='AIPING',
                                                   model_name=model_name,
                                                   api_key=api_key or lazyllm.config['aiping_api_key'],
                                                   return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'
        self._base_url = base_url

    def _make_request(self, endpoint, payload, timeout=TIMEOUT):
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self._base_url}{endpoint}'

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            lazyllm.LOG.error(f'Request failed: {e}')
            raise

    def _forward(self, input: str = None, negative_prompt: str = None, n: int = None,
                 size: str = None, seed: int = None, **kwargs):
        if not input:
            raise ValueError('Prompt is required')

        input_params = {
            'prompt': input,
            'negative_prompt': negative_prompt or '模糊，低质量'
        }

        extra_body = {}

        if n is not None:
            extra_body['n'] = n

        if size is not None:
            extra_body['size'] = size

        if seed is not None:
            extra_body['seed'] = seed

        payload = {
            'model': self._model_name,
            'input': input_params
        }

        if extra_body:
            payload['extra_body'] = extra_body

        try:
            result = self._make_request(self._endpoint, payload)

            images = result.get('data')
            if not images or not isinstance(images, list) or not images:
                raise ValueError(f'Unexpected response format: {result}')

            image_urls = [img.get('url') for img in images if img.get('url')]
            if not image_urls:
                raise ValueError(f'No image URLs found in response: {result}')

            return encode_query_with_filepaths(None, bytes_to_file([requests.get(url).content for url in image_urls]))

        except Exception as e:
            lazyllm.LOG.error(f'Failed to generate image: {e}')
            raise Exception(f'Failed to generate image: {str(e)}')
