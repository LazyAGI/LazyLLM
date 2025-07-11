import lazyllm
from typing import Any, Dict
from .qwenModule import QwenSTTModule, QwenTTSModule, QwenTextToImageModule
from .onlineMultiModuleBase import OnlineMultiModuleBase
from .doubaoModule import DoubaoTextToImageModule
from .openaiModule import OpenAISTTModule
from .glmModule import GLMSTTModule

class _OnlineMultiModuleMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineMultiModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineMultiModule(metaclass=_OnlineMultiModuleMeta):
    STT_MODELS = {
        'qwen': QwenSTTModule,
        'openai': OpenAISTTModule,
        'glm': GLMSTTModule
    }
    TTS_MODELS = {'qwen': QwenTTSModule}
    TEXT2IMAGE_MODELS = {
        'qwen': QwenTextToImageModule,
        'doubao': DoubaoTextToImageModule
    }

    @staticmethod
    def _encapsulate_parameters(base_url: str,
                                model: str,
                                return_trace: bool,
                                **kwargs) -> Dict[str, Any]:
        params = {"return_trace": return_trace}
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
                return_trace: bool = False,
                function: str = "stt",
                **kwargs):
        # 定义功能到模型映射
        FUNCTION_MODEL_MAP = {
            "stt": OnlineMultiModule.STT_MODELS,
            "tts": OnlineMultiModule.TTS_MODELS,
            "text2image": OnlineMultiModule.TEXT2IMAGE_MODELS,
        }

        if function not in FUNCTION_MODEL_MAP:
            raise ValueError(f"Invalid function: {function}")

        available_model = FUNCTION_MODEL_MAP[function]

        if model in available_model and source is None: source, model = model, source

        params = OnlineMultiModule._encapsulate_parameters(base_url, model, return_trace, **kwargs)

        if kwargs.get("skip_auth", False):
            source = source or "openai"
            if not base_url:
                raise KeyError("base_url must be set for local serving.")

        if source is None:
            if "api_key" in kwargs and kwargs["api_key"]:
                raise ValueError("No source is given but an api_key is provided.")
            for source in available_model:
                if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f"No api_key is configured for any of the models {available_model}.")

        assert source in available_model, f"Unsupported source: {source}"
        return available_model[source](**params)
