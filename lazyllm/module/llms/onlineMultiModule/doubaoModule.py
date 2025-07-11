from lazyllm.module.onlineMultiModule.onlineMultiModuleBase import OnlineMultiModuleBase
from volcenginesdkarkruntime import Ark
import requests
from typing import Union, Dict
import lazyllm
from lazyllm.components.utils.file_operate import bytes_to_file

class DoubaoModule(OnlineMultiModuleBase):
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModuleBase.__init__(self, model_series="DOUBAO", model_name=model_name,
                                       return_trace=return_trace, **kwargs)
        self._client = Ark(
            base_url=base_url,
            api_key=api_key or lazyllm.config['doubao_api_key'],
        )

class DoubaoTextToImageModule(DoubaoModule):
    MODEL_NAME = "doubao-seedream-3-0-t2i-250415"

    def __init__(self, api_key: str = None, model_name: str = None, stream: Union[bool, Dict[str, str]] = False,
                 return_trace: bool = False, **kwargs):
        DoubaoModule.__init__(self, api_key=api_key, model_name=model_name
                              or DoubaoTextToImageModule.MODEL_NAME
                              or lazyllm.config['doubao_text2image_model_name'],
                              stream=stream, return_trace=return_trace, **kwargs)
        self._format_output_files = bytes_to_file

    def _forward(self, input: str = None, **kwargs):
        imagesResponse = self._client.images.generate(
            model=self._model_name,
            prompt=input
        )
        return None, [requests.get(result.url).content for result in imagesResponse.data]
