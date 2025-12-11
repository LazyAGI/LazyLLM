import requests
import lazyllm
import os
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase

# 全局超时设置（秒）
TIMEOUT = 300

# AIPing API 基础URL
# 优先级：环境变量 > 默认值
AIPING_BASE_URL = (
    os.environ.get("AIPING_BASE_URL")
    or "https://aiping.cn/api/v1/"
)


class AipingModule(OnlineChatModuleBase, FileHandlerBase):
    """
    AIPing AI大语言模型实现
    支持多种模型类型，包括大模型和视觉问答模型
    """
    # 视觉语言模型前缀，支持图像理解的多模态模型
    VLM_MODEL_PREFIX = [
        'Qwen2.5-VL-',
        'Qwen3-VL-',
        'GLM-4.5V',
        'GLM-4.6V'
    ]

    def __init__(self, base_url: str = None, model: str = 'DeepSeek-R1',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        if base_url is None:
            base_url = AIPING_BASE_URL
        OnlineChatModuleBase.__init__(self, model_series='AIPING',
                                      api_key=api_key,
                                      base_url=base_url, model_name=model, stream=stream,
                                      return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by AIPing. You are a helpful assistant.'

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

    def _validate_api_key(self):
        """验证API Key"""
        try:
            # 参考 doubao.py 的实现
            chat_url = urljoin(self._base_url, 'chat/completions')
            headers = {
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1  # Only generate 1 token for validation
            }
            response = requests.post(chat_url, headers=headers, json=data, timeout=TIMEOUT)
            return response.status_code == 200
        except Exception:
            return False
        
    @property
    def get_models(self):
        url = urljoin(self._base_url, 'models')
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None


class AipingEmbedding(OnlineEmbeddingModuleBase):
    """
    AIPing文本向量化模型
    """
    def __init__(self, embed_url: str = None,
                 embed_model_name: str = 'text-embedding-v1', api_key: str = None,
                 batch_size: int = 16, **kw):
        if embed_url is None:
            embed_url = urljoin(AIPING_BASE_URL, 'embeddings')
        super().__init__('AIPING', embed_url, api_key,
                         embed_model_name, batch_size=batch_size, **kw)


class AipingReranking(OnlineEmbeddingModuleBase):
    """
    AIPing重排序模型
    """
    def __init__(self, embed_url: str = None,
                 embed_model_name: str = 'Qwen3-Reranker-0.6B', api_key: str = None, **kw):
        if embed_url is None:
            embed_url = urljoin(AIPING_BASE_URL, 'rerank')
        super().__init__('AIPING', embed_url, api_key,
                         embed_model_name, **kw)

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        """封装数据"""
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
        """解析响应"""
        results = response.get('results', [])
        if not results:
            return []
        return [(result['index'], result['relevance_score']) for result in results]


class AipingTextToImageModule(OnlineMultiModalBase):
    """
    AIPing文生图模型
    支持多种图像生成模型
    """
    SUPPORTED_MODELS = [
        'Qwen-Image',
        'HunyuanImage-3.0',
        '即梦文生图 3.0',
        '即梦文生图 3.1',
        'Doubao-Seedream-4.0',
        'Kolors',
        'Qwen-Image-Plus',
        'Wan2.5-T2I-Preview'
    ]

    def __init__(self, api_key: str = None, model_name: str = 'Qwen-Image',
                 base_url: str = None,
                 return_trace: bool = False, **kwargs):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model: {model_name}. Supported models: {self.SUPPORTED_MODELS}')

        OnlineMultiModalBase.__init__(self, model_series='AIPING',
                                      model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        if base_url is None:
            base_url = AIPING_BASE_URL
        self._endpoint = 'images/generations'
        self._base_url = base_url
        self._api_key = api_key

    def _make_request(self, endpoint, payload, timeout=TIMEOUT):
        """发起HTTP请求"""
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
        """文生图推理"""
        if not input:
            raise ValueError('Prompt is required')

        # 构建输入参数
        input_params = {
            'prompt': input,
            'negative_prompt': negative_prompt or '模糊，低质量'
        }

        # 构建额外的body参数
        extra_body = {}

        # 如果n不为None，添加到extra_body
        if n is not None:
            extra_body['n'] = n

        # 如果size不为None，添加到extra_body
        if size is not None:
            extra_body['size'] = size

        # 如果seed不为None，添加到extra_body
        if seed is not None:
            extra_body['seed'] = seed

        payload = {
            'model': self._model_name,
            'input': input_params
        }

        # 如果extra_body不为空，添加到payload
        if extra_body:
            payload['extra_body'] = extra_body

        try:
            result = self._make_request(self._endpoint, payload)

            # AIPing新格式：{"data": [{"url": "https://..."}, ...]}
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
