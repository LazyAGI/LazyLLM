from typing import Any, Dict
from .openaiModule import OpenAIModule
from .glmModule import GLMModule
from .moonshotaiModule import MoonshotAIModule
from .sensenovaModule import SenseNovaModule
from .qwenModule import QwenModule
from .doubaoModule import DoubaoModule
from .onlineChatModuleBase import OnlineChatModuleBase

class _ChatModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instatncheck__(self, __instance)


class OnlineChatModule(metaclass=_ChatModuleMeta):

    @staticmethod
    def _encapsulate_parameters(base_url: str,
                                model: str,
                                system_prompt: str,
                                stream: bool,
                                return_trace: bool,
                                **kwargs) -> Dict[str, Any]:
        params = {"stream": stream, "return_trace": return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        if system_prompt is not None:
            params['system_prompt'] = system_prompt
        params.update(kwargs)

        return params

    def __new__(self,
                source: str,
                base_url: str = None,
                model: str = None,
                system_prompt: str = None,
                stream: bool = True,
                return_trace: bool = False,
                **kwargs):
        params = OnlineChatModule._encapsulate_parameters(base_url, model, system_prompt, stream, return_trace, **kwargs)

        if source.lower() == "openai":
            return OpenAIModule(**params)
        elif source.lower() == "glm":
            return GLMModule(**params)
        elif source.lower() == "kimi":
            return MoonshotAIModule(**params)
        elif source.lower() == "sensenova":
            return SenseNovaModule(**params)
        elif source.lower() == "qwen":
            return QwenModule(**params)
        elif source.lower() == "doubao":
            if "model" not in params.keys():
                raise ValueError("Doubao model must be specified")
            return DoubaoModule(**params)
        else:
            raise ValueError("Unsupported source: {}".format(source))
