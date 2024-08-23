import lazyllm
from typing import Any, Dict
from .openaiModule import OpenAIModule
from .glmModule import GLMModule
from .kimiModule import KimiModule
from .sensenovaModule import SenseNovaModule
from .qwenModule import QwenModule
from .doubaoModule import DoubaoModule
from .onlineChatModuleBase import OnlineChatModuleBase

class _ChatModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineChatModule(metaclass=_ChatModuleMeta):
    MODELS = {'openai': OpenAIModule,
              'sensenova': SenseNovaModule,
              'glm': GLMModule,
              'kimi': KimiModule,
              'qwen': QwenModule,
              'doubao': DoubaoModule}

    @staticmethod
    def _encapsulate_parameters(base_url: str,
                                model: str,
                                stream: bool,
                                return_trace: bool,
                                **kwargs) -> Dict[str, Any]:
        params = {"stream": stream, "return_trace": return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)

        return params

    def __new__(self,
                model: str = None,
                source: str = None,
                base_url: str = None,
                stream: bool = True,
                return_trace: bool = False,
                **kwargs):
        if model in OnlineChatModule.MODELS.keys() and source is None: source, model = model, source

        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace, **kwargs)

        if source is None:
            for source in OnlineChatModule.MODELS.keys():
                if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f"No api_key is configured for any of the models {OnlineChatModule.MODELS.keys()}.")

        assert source in OnlineChatModule.MODELS.keys(), f"Unsupported source: {source}"
        return OnlineChatModule.MODELS[source](**params)
