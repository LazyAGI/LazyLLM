from typing import Any, Dict

import lazyllm
from .openaiEmbed import OpenAIEmbedding
from .glmEmbed import GLMEmbedding
from .sensenovaEmbed import SenseNovaEmbedding
from .qwenEmbed import QwenEmbedding
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(metaclass=__EmbedModuleMeta):
    MODELS = {'openai': OpenAIEmbedding,
              'sensenova': SenseNovaEmbedding,
              'glm': GLMEmbedding,
              'qwen': QwenEmbedding}

    @staticmethod
    def _encapsulate_parameters(embed_url: str,
                                embed_model_name: str,
                                **kwargs) -> Dict[str, Any]:
        params = {}
        if embed_url is not None:
            params["embed_url"] = embed_url
        if embed_model_name is not None:
            params["embed_model_name"] = embed_model_name
        params.update(kwargs)
        return params

    def __new__(self,
                source: str = None,
                embed_url: str = None,
                embed_model_name: str = None,
                **kwargs):
        params = OnlineEmbeddingModule._encapsulate_parameters(embed_url, embed_model_name, **kwargs)

        if source is None:
            if "api_key" in kwargs and kwargs["api_key"]:
                raise ValueError("No source is given but an api_key is provided.")
            for source in OnlineEmbeddingModule.MODELS.keys():
                if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f"No api_key is configured for any of the models {OnlineEmbeddingModule.MODELS.keys()}.")

        assert source in OnlineEmbeddingModule.MODELS.keys(), f"Unsupported source: {source}"
        return OnlineEmbeddingModule.MODELS[source](**params)
