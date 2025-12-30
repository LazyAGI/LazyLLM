import lazyllm
from typing import Dict, List, Union, Optional
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime

class DoubaoModule(OnlineChatModuleBase):
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='DOUBAO', api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoModule.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal request'''
        try:
            # Doubao (Volcano Engine) validates API key using a minimal chat request
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1  # Only generate 1 token for validation
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = 'doubao-embedding-vision-241215',
                 api_key: str = None):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{'text': input}]
        elif isinstance(input, list):
            # Validate input format, at most 1 text segment + 1 image
            if len(input) == 0:
                raise ValueError('Input list cannot be empty')
            if len(input) > 2:
                raise ValueError('Input list must contain at most 2 items (1 text and/or 1 image)')
        else:
            raise ValueError('Input must be either a string or a list of dictionaries')

        json_data = {
            'input': input,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[float]:
        # Doubao multimodal embedding returns a single fused embedding
        return response['data']['embedding']


class DoubaoMultiModal(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        api_key = api_key or lazyllm.config['doubao_api_key']
        OnlineMultiModalBase.__init__(self, model_series='DOUBAO', model_name=model_name, api_key=api_key,
                                      return_trace=return_trace, base_url=base_url, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(base_url=base_url, api_key=api_key)

class DoubaoTextToImageModule(DoubaoMultiModal):
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model_name=model_name
                                  or DoubaoTextToImageModule.MODEL_NAME
                                  or lazyllm.config['doubao_text2image_model_name'],
                                  return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, size: str = '1024x1024', seed: int = -1, guidance_scale: float = 2.5,
                 watermark: bool = True, **kwargs):
        model_name = kwargs.pop('_forward_model', self._model_name)
        base_url = kwargs.pop('_forward_url', None)
        client = self._client
        if base_url and base_url != getattr(self, '_base_url', None):
            client = volcenginesdkarkruntime.Ark(base_url=base_url, api_key=self._api_key)
        imagesResponse = client.images.generate(
            model=model_name,
            prompt=input,
            size=size,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
            **kwargs
        )
        return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                for result in imagesResponse.data]))
