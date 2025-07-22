import requests
import lazyllm
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from .onlineMultiModalBase import OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths


class DoubaoModule(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series="DOUBAO", model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(
            base_url=base_url,
            api_key=api_key or lazyllm.config['doubao_api_key'],
        )


class DoubaoTextToImageModule(DoubaoModule):
    MODEL_NAME = "doubao-seedream-3-0-t2i-250415"

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoModule.__init__(self, api_key=api_key, model_name=model_name
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
