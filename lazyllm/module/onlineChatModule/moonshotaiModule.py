import os
import lazyllm
from .onlineChatModuleBase import OnlineChatModuleBase

class MoonshotAIModule(OnlineChatModuleBase):

    def __init__(self,
                 base_url="https://api.moonshot.cn",
                 model="moonshot-v1-8k",
                 system_prompt="你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。\
                                你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，\
                                黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
                 stream=True,
                 return_trace=False,
                 **kwargs):

        super().__init__(model_type=__class__.__name__,
                         api_key=lazyllm.config['moonshotai_api_key'],
                         base_url=base_url,
                         model_name=model,
                         system_prompt=system_prompt,
                         stream=stream,
                         trainable_models=[],
                         return_trace=return_trace,
                         **kwargs)

    def _set_chat_url(self):
        self._url = os.path.join(self._base_url, 'v1/chat/completions')
