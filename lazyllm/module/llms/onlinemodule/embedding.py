from typing import Any, Dict

import lazyllm
from .base import OnlineEmbeddingModuleBase
from .supplier.openai import OpenAIEmbedding
from .supplier.glm import GLMEmbedding, GLMReranking
from .supplier.sensenova import SenseNovaEmbedding
from .supplier.qwen import QwenEmbedding, QwenReranking
from .supplier.doubao import DoubaoEmbedding, DoubaoMultimodalEmbedding

class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(metaclass=__EmbedModuleMeta):
    EMBED_MODELS = {'openai': OpenAIEmbedding,
                    'sensenova': SenseNovaEmbedding,
                    'glm': GLMEmbedding,
                    'qwen': QwenEmbedding,
                    'doubao': DoubaoEmbedding}
    RERANK_MODELS = {'qwen': QwenReranking,
                     'glm': GLMReranking}

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

    @staticmethod
    def _check_available_source(available_models):
        for source in available_models.keys():
            if lazyllm.config[f'{source}_api_key']: break
        else:
            raise KeyError(f"No api_key is configured for any of the models {available_models.keys()}.")

        assert source in available_models.keys(), f"Unsupported source: {source}"
        return source

    def __new__(self,
                source: str = None,
                embed_url: str = None,
                embed_model_name: str = None,
                **kwargs):
        params = OnlineEmbeddingModule._encapsulate_parameters(embed_url, embed_model_name, **kwargs)

        if source is None and "api_key" in kwargs and kwargs["api_key"]:
            raise ValueError("No source is given but an api_key is provided.")

        if "type" in params:
            params.pop("type")
        if kwargs.get("type", "embed") == "embed":
            if source is None:
                source = OnlineEmbeddingModule._check_available_source(OnlineEmbeddingModule.EMBED_MODELS)
            if source == "doubao":
                if embed_model_name.startswith("doubao-embedding-vision"):
                    return DoubaoMultimodalEmbedding(**params)
                else:
                    return DoubaoEmbedding(**params)
            return OnlineEmbeddingModule.EMBED_MODELS[source](**params)
        elif kwargs.get("type") == "rerank":
            if source is None:
                source = OnlineEmbeddingModule._check_available_source(OnlineEmbeddingModule.RERANK_MODELS)
            return OnlineEmbeddingModule.RERANK_MODELS[source](**params)
        else:
            raise ValueError("Unknown type of online embedding module.")
