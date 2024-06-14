import os
import lazyllm
from typing import Union, Dict, List
from .onlineChatModuleBase import OnlineChatModuleBase

class DoubaoModule(OnlineChatModuleBase):

    def __init__(self,
                 model: str,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 stream: bool = True,
                 return_trace: bool = False):
        super().__init__(model_type=self.__class__.__name__,
                         api_key=lazyllm.config['doubao_api_key'],
                         base_url=base_url,
                         model_name=model,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace)

    def _get_system_prompt(self):
        return "你是人工智能助手豆包。你的任务是针对用户的问题和要求提供适当的答复和支持。"

    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'chat/completions')

    def forward(self, __input: Union[Dict, str] = None, llm_chat_history: List[List[str]] = None, **kw):
        raise NotImplementedError("Individual user support is not friendly and is not supported yet")
