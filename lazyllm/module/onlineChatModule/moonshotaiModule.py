import os
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase

class MoonshotAIModule(OnlineChatModuleBase):

    def __init__(self,
                 base_url="https://api.moonshot.cn",
                 model="moonshot-v1-8k",
                 stream=True,
                 return_trace=False,
                 **kwargs):

        super().__init__(model_type=__class__.__name__,
                         api_key=lazyllm.config['moonshotai_api_key'],
                         base_url=base_url,
                         model_name=model,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace,
                         **kwargs)


    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'v1/chat/completions')
