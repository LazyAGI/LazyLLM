from typing import Any, Dict

import lazyllm
from .base import OnlineEmbeddingModuleBase
from .base.utils import select_source_with_default_key
from .supplier.openai import OpenAIEmbedding, OpenAIReranking
from .supplier.glm import GLMEmbedding, GLMReranking
from .supplier.sensenova import SenseNovaEmbedding
from .supplier.qwen import QwenEmbedding, QwenReranking
from .supplier.doubao import DoubaoEmbedding, DoubaoMultimodalEmbedding
from .supplier.siliconflow import SiliconFlowEmbedding, SiliconFlowReranking
from .supplier.aiping import AipingEmbedding, AipingReranking

class __EmbedModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineEmbeddingModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineEmbeddingModule(metaclass=__EmbedModuleMeta):
    @staticmethod
    def _get_group(name: str):
        group = getattr(lazyllm.online, name)
        return {k: v for k, v in group.items() if k != 'base'}

    @staticmethod
    def _encapsulate_parameters(embed_url: str,
                                embed_model_name: str,
                                **kwargs) -> Dict[str, Any]:
        params = {}
        if embed_url is not None:
            params['embed_url'] = embed_url
        if embed_model_name is not None:
            params['embed_model_name'] = embed_model_name
        params.update(kwargs)
        return params

    def __new__(self,
                source: str = None,
                embed_url: str = None,
                embed_model_name: str = None,
                **kwargs):
        params = OnlineEmbeddingModule._encapsulate_parameters(embed_url, embed_model_name, **kwargs)

        if source is None and 'api_key' in kwargs and kwargs['api_key']:
            raise ValueError('No source is given but an api_key is provided.')

        if 'type' in params:
            params.pop('type')
        if kwargs.get('type', 'embed') == 'embed':
            embed_models = OnlineEmbeddingModule._get_group('embedding')
            source, default_key = select_source_with_default_key(embed_models, explicit_source=source)
            if default_key and not kwargs.get('api_key'):
                kwargs['api_key'] = default_key
            if source == 'doubao':
                if embed_model_name.startswith('doubao-embedding-vision'):
                    return DoubaoMultimodalEmbedding(**params)
                else:
                    return DoubaoEmbedding(**params)
            return embed_models[source](**params)
        elif kwargs.get('type') == 'rerank':
            rerank_models = OnlineEmbeddingModule._get_group('reranking')
            source, default_key = select_source_with_default_key(rerank_models, explicit_source=source)
            if default_key and not kwargs.get('api_key'):
                kwargs['api_key'] = default_key
            return rerank_models[source](**params)
        else:
            raise ValueError('Unknown type of online embedding module.')
