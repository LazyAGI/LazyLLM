import lazyllm
from typing import Dict, List, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime

class DoubaoModule(OnlineChatModuleBase):
    MODEL_NAME = "doubao-1-5-pro-32k-250115"

    def __init__(self, model: str = None, base_url: str = "https://ark.cn-beijing.volces.com/api/v3/",
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series="DOUBAO", api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoModule.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ("You are Doubao, an AI assistant. Your task is to provide appropriate responses "
                "and support to users' questions and requests.")

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')


class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings",
                 embed_model_name: str = "doubao-embedding-text-240715",
                 api_key: str = None,
                 **kw):
        super().__init__("DOUBAO", embed_url, api_key or lazyllm.config["doubao_api_key"], embed_model_name, **kw)


class DoubaoMultimodalEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
                 embed_model_name: str = "doubao-embedding-vision-241215",
                 api_key: str = None):
        super().__init__("DOUBAO", embed_url, api_key or lazyllm.config["doubao_api_key"], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{"text": input}]
        elif isinstance(input, list):
            # 验证输入格式，最多为1段文本+1张图片
            if len(input) == 0:
                raise ValueError("Input list cannot be empty")
            if len(input) > 2:
                raise ValueError("Input list must contain at most 2 items (1 text and/or 1 image)")
        else:
            raise ValueError("Input must be either a string or a list of dictionaries")

        json_data = {
            "input": input,
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[float]:
        # 豆包多模态Embedding返回融合的单个embedding
        return response['data']['embedding']


class DoubaoMultiModal(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series="DOUBAO", model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(
            base_url=base_url,
            api_key=api_key or lazyllm.config['doubao_api_key'],
        )


class DoubaoTextToImageModule(DoubaoMultiModal):
    MODEL_NAME = "doubao-seedream-3-0-t2i-250415"

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model_name=model_name
                                  or DoubaoTextToImageModule.MODEL_NAME
                                  or lazyllm.config['doubao_text2image_model_name'],
                                  return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, size: str = "1024x1024", seed: int = -1, guidance_scale: float = 2.5,
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
