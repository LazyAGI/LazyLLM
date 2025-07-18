import lazyllm
import requests
from typing import List
import os
from .onlineMultiModalBase import OnlineMultiModalBase
from lazyllm.thirdparty import zhipuai
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.formatter import encode_query_with_filepaths
class GLMModule(OnlineMultiModalBase):
    def __init__(self, model_name: str, api_key: str = None,
                 base_url: str = 'https://open.bigmodel.cn/api/paas/v4', return_trace: bool = False,
                 **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='GLM', model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = zhipuai.ZhipuAI(api_key=api_key or lazyllm.config['glm_api_key'], base_url=base_url)

class GLMSTTModule(GLMModule):
    MODEL_NAME = "glm-asr"

    def __init__(self, model_name: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        GLMModule.__init__(self, model_name=model_name or GLMSTTModule.MODEL_NAME
                           or lazyllm.config['glm_stt_model_name'], api_key=api_key,
                           return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):
        assert len(files) == 1, "GLMSTTModule only supports one file"
        assert os.path.exists(files[0]), f"File {files[0]} not found"
        transcriptResponse = self._client.audio.transcriptions.create(
            model=self._model_name,
            file=open(files[0], "rb"),
        )
        return transcriptResponse.text

class GLMTexToImageModule(GLMModule):
    MODEL_NAME = "cogview-4-250304"

    def __init__(self, model_name: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        GLMModule.__init__(self, model_name=model_name or GLMTexToImageModule.MODEL_NAME
                           or lazyllm.config['glm_text_to_image_model_name'], api_key=api_key,
                           return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, n: int = 1, size: str = '1024x1024', **kwargs):
        call_params = {
            'model': self._model_name,
            'prompt': input,
            'n': n,
            'size': size,
            **kwargs
        }
        response = self._client.images.generations(**call_params)
        return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                for result in response.data]))
