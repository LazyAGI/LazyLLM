import lazyllm
from typing import Dict, List, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime

class DoubaoModule(OnlineChatModuleBase):
    """Doubao online chat module, inheriting from OnlineChatModuleBase.  
Encapsulates the Doubao API (ByteDance) for multi-turn Q&A interactions. Defaults to model `doubao-1-5-pro-32k-250115`, supporting streaming and optional trace return.

Args:
    model (str): The model name to use. Defaults to `doubao-1-5-pro-32k-250115`.
    base_url (str): Base URL of the API, default is "https://ark.cn-beijing.volces.com/api/v3/".
    api_key (Optional[str]): Doubao API key. If not provided, it is read from `lazyllm.config['doubao_api_key']`.
    stream (bool): Whether to enable streaming output. Defaults to True.
    return_trace (bool): Whether to return trace information. Defaults to False.
    **kwargs: Additional arguments passed to the base class OnlineChatModuleBase.
"""
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='DOUBAO', api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoModule.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')


class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    """DoubaoEmbedding class inherits from OnlineEmbeddingModuleBase, encapsulating the functionality to call Doubao's online text embedding service.  
It supports remote text vector representation retrieval by specifying the service URL, model name, and API key.

Args:
    embed_url (Optional[str]): URL of the Doubao text embedding service, defaulting to the Beijing region endpoint.
    embed_model_name (Optional[str]): Name of the Doubao embedding model used, default is "doubao-embedding-text-240715".
    api_key (Optional[str]): API key for accessing the Doubao service. If not provided, it is read from lazyllm config.
"""
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 **kw):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name, **kw)


class DoubaoMultimodalEmbedding(OnlineEmbeddingModuleBase):
    """DoubaoMultimodalEmbedding class inherits from OnlineEmbeddingModuleBase, encapsulating the functionality to call Doubao's online multimodal (text + image) embedding service.  
It supports converting text and image inputs into a unified vector representation by specifying the service URL, model name, and API key, enabling remote retrieval of multimodal embeddings.

Args:
    embed_url (Optional[str]): URL of the Doubao multimodal embedding service, defaulting to the Beijing region endpoint.
    embed_model_name (Optional[str]): Name of the Doubao multimodal embedding model used, default is "doubao-embedding-vision-241215".
    api_key (Optional[str]): API key for accessing the Doubao service. If not provided, it is read from lazyllm config.
"""
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = 'doubao-embedding-vision-241215',
                 api_key: str = None):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{'text': input}]
        elif isinstance(input, list):
            # 验证输入格式，最多为1段文本+1张图片
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
        # 豆包多模态Embedding返回融合的单个embedding
        return response['data']['embedding']


class DoubaoMultiModal(OnlineMultiModalBase):
    """Doubao MultiModal module, inheriting from OnlineMultiModalBase, encapsulates the functionality to call Doubao's multimodal service.  
By specifying the API key, model name, and base service URL, it allows remote interaction with Doubao's API for multimodal data processing and feature extraction.

Args:
    api_key (Optional[str]): API key for accessing Doubao service. If not provided, it is read from lazyllm config.
    model_name (Optional[str]): Name of the Doubao multimodal model to use.
    base_url (str): Base URL of the Doubao service, defaulting to the Beijing region endpoint.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to OnlineMultiModalBase.
"""
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='DOUBAO', model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(
            base_url=base_url,
            api_key=api_key or lazyllm.config['doubao_api_key'],
        )


class DoubaoTextToImageModule(DoubaoMultiModal):
    """ByteDance Doubao Text-to-Image module supporting text to image generation.

Based on ByteDance Doubao multimodal model's text-to-image functionality, 
inherits from DoubaoMultiModal, providing high-quality text to image generation capability.

Args:
    api_key (str, optional): Doubao API key, defaults to None.
    model_name (str, optional): Model name, defaults to "doubao-seedream-3-0-t2i-250415".
    return_trace (bool, optional): Whether to return trace information, defaults to False.
    **kwargs: Other parameters passed to parent class.
"""
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model_name=model_name
                                  or DoubaoTextToImageModule.MODEL_NAME
                                  or lazyllm.config['doubao_text2image_model_name'],
                                  return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, size: str = '1024x1024', seed: int = -1, guidance_scale: float = 2.5,
                 watermark: bool = True, **kwargs):
        imagesResponse = self._client.images.generate(
            model=self._model_name,
            prompt=input,
            size=size,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
            **kwargs
        )
        return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                for result in imagesResponse.data]))
