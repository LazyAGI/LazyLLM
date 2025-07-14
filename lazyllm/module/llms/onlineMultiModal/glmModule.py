import lazyllm
from zhipuai import ZhipuAI
from typing import List
import os
from .onlineMultiModalBase import OnlineMultiModalBase

class GLMModule(OnlineMultiModalBase):
    def __init__(self, model_series: str, model_name: str, api_key: str = None,
                 base_url: str = 'https://open.bigmodel.cn/api/paas/v4', return_trace: bool = False,
                 **kwargs):
        OnlineMultiModalBase.__init__(self, model_series=model_series, model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = ZhipuAI(api_key=api_key or lazyllm.config['glm_api_key'], base_url=base_url)

class GLMSTTModule(GLMModule):
    MODEL_NAME = "glm-asr"

    def __init__(self, model_name: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        GLMModule.__init__(self, model_series="GLM", model_name=model_name or GLMSTTModule.MODEL_NAME
                           or lazyllm.config['glm_stt_model_name'], api_key=api_key
                           or lazyllm.config['glm_api_key'], return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):
        assert len(files) == 1, "GLMSTTModule only supports one file"
        assert os.path.exists(files[0]), f"File {files[0]} not found"
        transcriptResponse = self._client.audio.transcriptions.create(
            model=self._model_name,
            file=open(files[0], "rb"),
            stream=False
        )
        return transcriptResponse.text, None
