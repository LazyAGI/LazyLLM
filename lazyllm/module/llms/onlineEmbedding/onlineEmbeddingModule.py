from typing import Any, Dict

import lazyllm
from .openaiEmbed import OpenAIEmbedding
from .glmEmbed import GLMEmbedding, GLMReranking
from .sensenovaEmbed import SenseNovaEmbedding
from .qwenEmbed import QwenEmbedding, QwenReranking
from .doubaoEmbed import DoubaoEmbedding, DoubaoMultimodalEmbedding
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(metaclass=__EmbedModuleMeta):
    """Used to manage and create online Embedding service modules currently on the market, currently supporting openai, sensenova, glm, qwen, doubao.

Args:
    source (str): Specify the type of module to create. Options are  ``openai`` /  ``sensenova`` /  ``glm`` /  ``qwen`` / ``doubao``.
    embed_url (str): Specify the base link of the platform to be accessed. The default is the official link.
    embed_mode_name (str): Specify the model to access (Note that you need to use Model ID or Endpoint ID when using Doubao. For details on how to obtain it, see [Getting the Inference Access Point](https://www.volcengine.com/docs/82379/1099522). Before using the model, you must first activate the corresponding service on the Doubao platform.), default is ``text-embedding-ada-002(openai)`` / ``nova-embedding-stable(sensenova)`` / ``embedding-2(glm)`` / ``text-embedding-v1(qwen)`` / ``doubao-embedding-text-240715(doubao)``


Examples:
    >>> import lazyllm
    >>> m = lazyllm.OnlineEmbeddingModule(source="sensenova")
    >>> emb = m("hello world")
    >>> print(f"emb: {emb}")
    emb: [0.0010528564, 0.0063285828, 0.0049476624, -0.012008667, ..., -0.009124756, 0.0032043457, -0.051696777]
    """
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
