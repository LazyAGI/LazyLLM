from typing import Any, Dict
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

    @staticmethod
    def _encapsulate_parameters(embed_url: str,
                                embed_model_name: str) -> Dict[str, Any]:
        params = {}
        if embed_url is not None:
            params["embed_url"] = embed_url
        if embed_model_name is not None:
            params["embed_model_name"] = embed_model_name
        return params

    def __new__(self,
                source: str,
                embed_url: str = None,
                embed_model_name: str = None):
        params = OnlineEmbeddingModule._encapsulate_parameters(embed_url, embed_model_name)
        if source.lower() == "openai":
            return OpenAIEmbedding(**params)
        elif source.lower() == "glm":
            return GLMEmbedding(**params)
        elif source.lower() == "sensenova":
            return SenseNovaEmbedding(**params)
        elif source.lower() == "qwen":
            return QwenEmbedding(**params)
